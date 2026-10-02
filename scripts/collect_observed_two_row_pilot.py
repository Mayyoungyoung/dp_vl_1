"""Independent RLBench-derived low-post, two-row feasibility pilot.

No model training, automatic retry, or inference-time oracle geometry. Exactly
one registered setup action and nine proposals per registered target. Legacy
versions use three targets; v4 uses the predeclared central target only. The
setup trajectory is separate from routes starting at the restored low entry.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time
import traceback

import numpy as np
from PIL import Image

# Existing collector helpers use script-local imports. Resolve that known
# directory explicitly for both CLI and pytest/module imports.
_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import collect_obstacle_reach as legacy
import two_row_anchor


PASSAGES = ("negative_y", "middle", "positive_y")
LOWER_POST_PROTOCOL = "observed_two_row_lower_posts_central_v4"


def registered_target_indices(config):
    expected = [1] if config["protocol"] == LOWER_POST_PROTOCOL else [0, 1, 2]
    if config.get("selected_target_indices", expected) != expected:
        raise ValueError("target selection differs from registered protocol")
    return expected


def summarize_reference_types(identifier, route_types, config):
    """Only accepted trajectories enter this list; unknown remains a reference."""
    known = {tuple(value) for value in route_types if value is not None}
    lateral = {value for value in known if all(item in PASSAGES for item in value)}
    include_over = config["protocol"] == LOWER_POST_PROTOCOL
    return dict(id=identifier, valid_references=len(route_types), distinct_classified=len(known),
        known_classified_sequences=sorted(known), distinct_lateral_sequences=len(lateral),
        known_lateral_sequences=sorted(lateral), unknown_valid_references=sum(v is None for v in route_types),
        has_more_than_four_valid_lateral_sequences=len(lateral)>4,
        feasibility_type_policy="lateral_and_over" if include_over else "lateral_only",
        preregistered_feasibility_met=(len(known) if include_over else len(lateral))>=5)


def geometry(config):
    centers = np.array([[x, y, config["post_base_z"] + config["post_size_xyz"][2] / 2]
                        for x, ys in zip(config["row_x"], config["post_y"]) for y in ys], dtype=float)
    return centers, np.tile(np.asarray(config["post_size_xyz"], dtype=float) / 2, (4, 1))


def validate_config(config):
    versions = {"observed_two_row_low_posts_v1":False, "observed_two_row_low_posts_rowpoints_v2":True,
                "observed_two_row_recorded_v1_anchor_rowpoints_v3":True, LOWER_POST_PROTOCOL:True}
    if config["protocol"] not in versions or config["parents"] != 1:
        raise ValueError("this version is a single registered feasibility parent")
    if bool(config.get("row_plane_guides",False)) != versions[config["protocol"]]:
        raise ValueError("row-plane guide policy differs from declared version")
    if (config["protocol"]=="observed_two_row_recorded_v1_anchor_rowpoints_v3") != ("initial_state_anchor" in config):
        raise ValueError("version and explicit v1 reconstruction policy differ")
    requested = 9 * len(registered_target_indices(config))
    if config["split"] != "DEV_COLLECTION" or config["requested_route_proposals"] != requested:
        raise ValueError("pilot role and proposal budget are fixed")
    if config["protocol"] == LOWER_POST_PROTOCOL:
        if (config.get("selected_target_indices") != [1] or
                config.get("feasibility_type_policy") != "lateral_and_over" or
                config.get("minimum_distinct_valid_types") != 5):
            raise ValueError("v4 requires explicit central target and its new type policy")
    if np.asarray(config["goal_xyz"]).shape != (3, 3) or np.asarray(config["post_y"]).shape != (2, 2):
        raise ValueError("three goals and two pairs of posts are required")
    if any(a >= b for a, b in zip(config["row_x"], config["row_x"][1:])):
        raise ValueError("rows must be in increasing x order")
    if not config["entry_xyz"][0] < config["row_x"][0] < config["row_x"][1] < min(g[0] for g in config["goal_xyz"]):
        raise ValueError("entry and goals must straddle both rows")
    centers, halves = geometry(config)
    if np.any(halves <= 0) or any(ys[0] >= ys[1] for ys in config["post_y"]):
        raise ValueError("positive sizes and ordered post pairs required")
    if config["canonicalization_passes"] != 2 or config["requested_setup_actions"] != 1:
        raise ValueError("fixed restore and setup protocols required")
    if config["reference_set_complete"] or config["all_solution_count"] is not None:
        raise ValueError("nine proposals are not an exhaustive solution set")
    return centers, halves


def proposed_waypoints(goal, sequence, config):
    """These guides are collection labels, never observation model inputs."""
    indices = [PASSAGES.index(name) for name in sequence]
    row1, row2 = config["row_x"]
    y1, y2 = [config["corridor_guide_y"][i] for i in indices]
    offset, z = config["row_guide_x_offset"], config["route_height"]
    middle = (row1 + row2) / 2
    points = [[row1-offset, y1, z]]
    if config.get("row_plane_guides",False):
        points.append([row1,y1,z])
    points.extend([[row1+offset,y1,z],[middle,y1,z],[middle,y2,z],[row2-offset,y2,z]])
    if config.get("row_plane_guides",False):
        points.append([row2,y2,z])
    points.extend([[row2+offset,y2,z],goal])
    return np.asarray(points,dtype=float)


def crossing_signature(xyz, config):
    """Actual full-route row crossings, not the guide IDs.

    Each row is crossed in one consistent corridor. Reversed/missing row
    order, transverse boundary ambiguity, or inconsistent backtracking is
    unknown. Above-row passages are explicitly 'over', not lateral openings.
    These are finite-object passage relations, not homotopy equivalence.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        return None
    rows = config["row_x"]
    if not xyz[0, 0] < rows[0] or not xyz[-1, 0] > rows[1]:
        return None
    margin = config["tip_clearance_m"]
    bottom, top = config["post_base_z"], config["post_base_z"] + config["post_size_xyz"][2]
    half_y = config["post_size_xyz"][1] / 2
    labels, events = [[], []], []
    for segment, (a, b) in enumerate(zip(xyz[:-1], xyz[1:])):
        segment_events = []
        for row, x in enumerate(rows):
            if not min(a[0], b[0]) <= x <= max(a[0], b[0]):
                continue
            if abs(b[0] - a[0]) <= 1e-12:
                points = [a, b]
                fraction = 0.
            else:
                fraction = (x-a[0])/(b[0]-a[0])
                points = [a + fraction*(b-a)]
            for point in points:
                if point[2] > top + margin:
                    label = "over"
                elif bottom + margin <= point[2] <= top - margin:
                    low, high = config["post_y"][row]
                    y = point[1]
                    if y < low-half_y-margin:
                        label = "negative_y"
                    elif low+half_y+margin < y < high-half_y-margin:
                        label = "middle"
                    elif y > high+half_y+margin:
                        label = "positive_y"
                    else:
                        return None
                else:
                    return None
                labels[row].append(label)
            segment_events.append((fraction, row))
        for _, row in sorted(segment_events):
            if not events or events[-1] != row:
                events.append(row)
    if events != [0, 1] or any(not row or len(set(row)) != 1 for row in labels):
        return None
    return tuple(row[0] for row in labels)


def strict_observation_difference(reference, current):
    values = {}
    for key in ("front_rgb", "front_depth", "gripper_pose", "gripper_open"):
        a, b = np.asarray(getattr(reference, key)), np.asarray(getattr(current, key))
        values[key + "_equal"] = bool(np.array_equal(a, b))
    for key in ("front_camera_intrinsics", "front_camera_extrinsics"):
        values[key + "_equal"] = bool(np.array_equal(reference.misc[key], current.misc[key]))
    return values


def save_observation_evidence(prefix, observation, world=None):
    """Validation evidence is deliberately separate from observation inputs."""
    Image.fromarray(observation.front_rgb).save(prefix.with_suffix(".png"))
    np.savez_compressed(prefix.with_suffix(".npz"), rgb=observation.front_rgb, depth=observation.front_depth,
        mask=observation.front_mask, gripper_pose=observation.gripper_pose, gripper_open=observation.gripper_open,
        camera_intrinsics=observation.misc["front_camera_intrinsics"], camera_extrinsics=observation.misc["front_camera_extrinsics"])
    if world is not None:
        write_json(prefix.with_suffix(".json"), world)


def state_sample(task):
    robot = task._robot
    return (np.asarray(robot.arm.get_tip().get_pose()),
            1. if robot.gripper.get_open_amount()[0] > .9 else 0.,
            np.asarray(robot.arm.get_joint_positions()), np.asarray(robot.gripper.get_joint_positions()))


def execute(task, waypoints, quaternion, gripper_shapes, external_shapes, trace, record, config):
    segments = record.setdefault("planning_segments", [])
    for index, waypoint in enumerate(waypoints):
        segment = dict(segment=index, goal_supervision_only=np.asarray(waypoint).tolist(),
                       get_path_calls=1, planning_status="running", planning_seconds=0.,
                       simulation_status="unattempted", simulation_seconds=0., simulated_steps=0)
        segments.append(segment)
        plan_started = time.perf_counter()
        try:
            path = task._robot.arm.get_path(np.asarray(waypoint).tolist(), quaternion=np.asarray(quaternion).tolist(),
                                            ignore_collisions=False)
            segment["planning_status"] = "success"
        except Exception as error:
            segment.update(planning_status="failed", planning_error=repr(error))
            raise
        finally:
            segment["planning_seconds"] = time.perf_counter()-plan_started
        simulation_started = time.perf_counter()
        segment["simulation_status"] = "running"
        try:
            done = False
            for _ in range(config["maximum_steps_per_segment"]):
                done = path.step()
                task._scene.step()
                trace.append(state_sample(task))
                record["simulated_steps"] += 1
                segment["simulated_steps"] += 1
                collision = legacy.robot_collision_check(task, gripper_shapes, external_shapes)
                if collision:
                    record["collision_pair"] = collision
                    raise RuntimeError("robot collision during simulated motion")
                if done:
                    break
            if not done:
                raise RuntimeError("segment simulation-step budget exhausted")
            segment["simulation_status"] = "success"
        except Exception as error:
            segment.update(simulation_status="failed", simulation_error=repr(error))
            raise
        finally:
            segment["simulation_seconds"] = time.perf_counter()-simulation_started


def add_execution_totals(counts, record):
    segments = record.get("planning_segments", [])
    for key in ("get_path_calls", "planning_seconds", "simulation_seconds"):
        counts[key] = counts.get(key, 0) + sum(segment[key] for segment in segments)


def save_trace(path, trace, extra=None):
    if not trace:
        return None
    arrays = dict(gripper_pose=np.asarray([s[0] for s in trace]), gripper_open=np.asarray([s[1] for s in trace]),
                  arm_joint_positions=np.asarray([s[2] for s in trace]), gripper_joint_positions=np.asarray([s[3] for s in trace]))
    arrays.update(extra or {})
    np.savez_compressed(path, **arrays)
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                pose_array_sha256=legacy.array_hash(arrays["gripper_pose"]), samples=len(trace))


def write_json(path, value):
    path.write_text(legacy.json_text(value, indent=2), encoding="utf-8")


def collect(config, config_path, output):
    from pyrep.const import ObjectType, PrimitiveShape
    from pyrep.objects.shape import Shape
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks.reach_target import ReachTarget
    from rlbench.const import colors
    centers, halves = validate_config(config)
    target_indices=registered_target_indices(config)
    requested_routes=config["requested_route_proposals"]
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    parent_id = "two_row_reach_%06d" % config["seed"]
    folder = output / parent_id
    folder.mkdir()
    source_files = [Path(__file__), Path(legacy.__file__), Path(legacy.native_snapshot.__code__.co_filename),Path(two_row_anchor.__file__)]
    anchored="initial_state_anchor" in config
    manifest = dict(config=config, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        sources_sha256={str(p.name): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        benchmark="RLBench-derived two-row low-post pilot", original_benchmark_result=False,
        rlbench_revision="02720bba4c73fe02eb75df946b8791b806028a9d", pyrep_revision="8f420be8064b1970aae18a9cfbc978dfb15747ef",
        input_contract=["RGB", "instruction", "depth", "camera", "current gripper pose/open"],
        supervision_only=["post geometry", "target coordinates", "route signatures", "guide points", "masks", "setup trajectory"],
        proposed_sequences=list(itertools.product(PASSAGES, repeat=2)),
        reference_set_complete=False, all_solution_count=None, full_robot_continuous_certificate=False,
        initial_preparation=("One reconstruction action from frozen v1 preparation terminal joints; no new setup IK; require every recorded world/velocity/RGB-D/calibration field exactly equal before any of 27 slots" if anchored else
            "One separately budgeted two-segment collision-checked setup; actual low-state snapshot is route start"),
        selected_target_indices=target_indices,physical_target_count=3,
        feasibility_type_policy="lateral_and_over" if config["protocol"]==LOWER_POST_PROTOCOL else "lateral_only",
        minimum_distinct_valid_types=5,old_dynamic_initial_state_pairing_claimed=False,
        planning_budget=dict(route_slots=requested_routes,maximum_calls_per_route=9 if config.get("row_plane_guides",False) else 7,
            setup_maximum_calls=0 if anchored else 2,failed_slots_stop_early=True,
            internal_search_scope="IK/OMPL configuration searches are not complete output trajectories and are not individually instrumented"),
        acceptance="strict full world/inventory/RGB/depth/camera/current restore; per-step arm/gripper collisions; 2cm tip segments; 3cm endpoint",
        type_definition="consistent actual crossings of both row-x planes in order; finite-height lateral corridors or explicit over; ambiguous unknown",
        camera_policy="initial and restore RGB-D enabled; trajectory state read directly; final RGB-D equivalence check",
        render_protocol="native_double_restart_explicit_render_warmup_v3", role="DEV_COLLECTION")
    write_json(output / "manifest.json", manifest)
    obsconfig = ObservationConfig()
    obsconfig.set_all(False)
    obsconfig.front_camera.rgb = obsconfig.front_camera.depth = obsconfig.front_camera.mask = True
    obsconfig.front_camera.depth_in_meters = obsconfig.front_camera.masks_as_one_channel = True
    obsconfig.front_camera.image_size = (config["image_size"], config["image_size"])
    obsconfig.gripper_pose = obsconfig.gripper_open = True
    obsconfig.task_low_dim_state = False
    env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=obsconfig, headless=True)
    counts = dict(requested_parents=1, requested_setup_actions=1, setup_actions_attempted=0, setup_successes=0,
                  requested_route_proposals=requested_routes, route_attempts=0, route_successes=0, classified_successes=0,
                  lateral_classified_successes=0, restore_passes=0, duplicate_types=0)
    setup = dict(parent_id=parent_id, phase="setup", success=False, simulated_steps=0)
    trace, initial, per_target, fatal_error, shutdown_error = [], None, [], None, None
    try:
        env.launch()
        env._pyrep.stop()
        task = env.get_task(ReachTarget)
        env._pyrep.stop()
        posts = []
        for index, (center, half) in enumerate(zip(centers, halves)):
            post = Shape.create(PrimitiveShape.CUBOID, (2*half).tolist(), static=True, respondable=True,
                                position=center.tolist(), color=[.32, .34, .38])
            post.set_name("derived_two_row_post_%d" % index)
            post.set_collidable(True)
            post.set_detectable(True)
            posts.append(post)
        env._pyrep.start()
        task._robot.arm.set_control_loop_enabled(True)
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        task.set_variation(0)
        reset_rng = np.random.get_state()
        setup_started = time.perf_counter()
        counts["setup_actions_attempted"] = 1
        try:
            _, nominal = task.reset()
            targets = [task._task.target, task._task.distractor0, task._task.distractor1]
            for target, position in zip(targets, config["goal_xyz"]):
                target.set_position(position)
            palette = np.asarray([item[1] for item in colors])
            target_colors = [int(np.linalg.norm(palette-target.get_color(), axis=1).argmin()) for target in targets]
            if len(set(target_colors)) != 3:
                raise RuntimeError("three target color instructions are not distinct")
            for post, center, half in zip(posts, centers, halves):
                bounds = np.asarray(post.get_bounding_box()).reshape(3, 2)
                if not (np.allclose(post.get_position(), center, atol=1e-6, rtol=0) and
                        np.allclose(bounds, np.stack([-half, half], axis=1), atol=1e-6, rtol=0)):
                    raise RuntimeError("physical post geometry differs from registered configuration")
            robot_handles = {obj.get_handle() for obj in task._robot.arm.get_objects_in_tree(exclude_base=False) +
                             task._robot.gripper.get_objects_in_tree(exclude_base=False)}
            gripper_shapes = [obj for obj in task._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE,
                              exclude_base=False) if obj.is_collidable()]
            external_shapes = [obj for obj in task._pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE,
                               exclude_base=False) if obj.is_collidable() and obj.get_handle() not in robot_handles]
            if not gripper_shapes or legacy.robot_collision_check(task, gripper_shapes, external_shapes):
                raise RuntimeError("initial robot collision or missing gripper collision shapes")
            before = task.get_observation()
            save_observation_evidence(folder / "before_preparation", before, legacy.full_audit(task, posts))
            if anchored:
                anchor=two_row_anchor.load_anchor(config)
                two_row_anchor.apply_recorded_task_state(task,posts,anchor)
                snapshot=legacy.full_snapshot(task,posts)
                snapshot["core"]["arm_joints"]=anchor["preparation_arm_joints"].tolist()
                snapshot["core"]["gripper_joints"]=anchor["preparation_gripper_joints"].tolist()
                setup.update(initialization="reconstruct_then_exact_readback",new_setup_ik_calls=0,
                             reconstruction_source=anchor["source"],canonicalization_steps=20)
            else:
                trace.append(state_sample(task))
                quaternion = trace[0][0][3:].copy()
                execute(task, config["preparation_xyz"], quaternion, gripper_shapes, external_shapes, trace, setup, config)
                setup["endpoint_error_m"] = float(np.linalg.norm(trace[-1][0][:3] - config["entry_xyz"]))
                if setup["endpoint_error_m"] > config["preparation_endpoint_tolerance_m"]:
                    raise RuntimeError("preparation did not reach registered entry")
                snapshot = legacy.full_snapshot(task, posts)
            legacy.canonical_restore(task, snapshot, config["canonicalization_passes"])
            initial = task.get_observation()
            reference = legacy.full_audit(task, posts)
            if anchored:
                trace.append(state_sample(task))
                gate=two_row_anchor.exact_readback(anchor,reference,initial)
                setup["v1_recorded_state_readback"]=gate
                write_json(folder/"v1_anchor_readback.json",gate)
                if not gate["passed"]:
                    raise RuntimeError("v1 recorded initial-state reconstruction differed; all 27 route slots remain unattempted")
            if legacy.robot_collision_check(task, gripper_shapes, external_shapes):
                raise RuntimeError("canonical low initial state is colliding")
            if np.linalg.norm(np.asarray(initial.gripper_pose[:3])-config["entry_xyz"]) > config["preparation_endpoint_tolerance_m"]:
                raise RuntimeError("canonical low state departed from registered entry")
            depth_audit = legacy.observed_box_depth_audit(initial, posts, centers, halves)
            visible = [int(np.sum(initial.front_mask == target.get_handle())) for target in targets]
            if min(visible) < config["minimum_visible_pixels"]:
                raise RuntimeError("target visibility insufficient: " + str(visible))
            if anchored:
                # Pinned PyRep returns an opaque CFFI pointer for config trees,
                # despite the bytes annotation. Never guess its buffer length.
                # The supported scene export is a native artifact; future load
                # still requires readback, and hidden dynamics are not certified.
                scene_path=folder/"verified_initial_scene.ttt"
                task._pyrep.export_scene(str(scene_path))
                setup["native_scene_export"]=dict(file=scene_path.name,sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
                    roundtrip_verified=False,configuration_tree_pointer_not_serialized=True)
            setup.update(success=True, target_visible_pixels=visible, post_depth_audit=depth_audit,
                         actual_entry_xyz=np.asarray(initial.gripper_pose[:3]),
                         target_colors=[colors[index][0] for index in target_colors])
            counts["setup_successes"] = 1
        except Exception as error:
            setup.update(error=repr(error), traceback=traceback.format_exc())
            try:
                save_observation_evidence(folder / "failed_setup", task.get_observation(), legacy.full_audit(task, posts))
            except Exception as diagnostic_error:
                setup["diagnostic_capture_error"] = repr(diagnostic_error)
        finally:
            setup["seconds"] = time.perf_counter() - setup_started
            setup["trajectory"] = save_trace(folder / "preparation_trace.npz", trace)
            add_execution_totals(counts, setup)
            write_json(folder / "setup.json", setup)
        if not setup["success"]:
            return
        goals = np.asarray([target.get_position() for target in targets])
        Image.fromarray(initial.front_rgb).save(folder / "front.png")
        np.savez_compressed(folder / "observation.npz", depth=initial.front_depth, gripper_pose=initial.gripper_pose,
            gripper_open=initial.gripper_open, camera_intrinsics=initial.misc["front_camera_intrinsics"],
            camera_extrinsics=initial.misc["front_camera_extrinsics"])
        np.savez_compressed(folder / "verification_only.npz", mask=initial.front_mask, obstacle_centers=centers,
                            obstacle_halfsizes=halves, target_centers=goals, target_visible_pixels=visible)
        write_json(folder / "restore_reference.json", reference)
        palette = np.asarray([item[1] for item in colors])
        for target_index in target_indices:
            target=targets[target_index]
            identifier = parent_id + "_target%d" % target_index
            color_index = int(np.linalg.norm(palette-target.get_color(), axis=1).argmin())
            instruction = "Move the gripper to touch the %s sphere while avoiding the gray posts." % colors[color_index][0]
            legacy.append_json(output / "observations.jsonl", dict(id=identifier, parent_id=parent_id, split=config["split"],
                image=parent_id+"/front.png", instruction=instruction))
            routes, types, seen, lateral = [], [], set(), set()
            for attempt, proposed in enumerate(itertools.product(PASSAGES, repeat=2)):
                tic = time.perf_counter()
                counts["route_attempts"] += 1
                record = dict(parent_id=parent_id, input_id=identifier, attempt=attempt, success=False, simulated_steps=0,
                              proposed_type_supervision_only=proposed, actual_route_type=None,
                              camera_flags=dict(rgb=obsconfig.front_camera.rgb, depth=obsconfig.front_camera.depth,
                                                mask=obsconfig.front_camera.mask, depth_in_meters=obsconfig.front_camera.depth_in_meters))
                route_trace = []
                try:
                    np.random.set_state(reset_rng)
                    task.reset()
                    legacy.canonical_restore(task, snapshot, config["canonicalization_passes"])
                    restored = task.get_observation()
                    restored_world = legacy.full_audit(task, posts)
                    difference = legacy.compare_restore(reference, restored_world, initial.front_rgb, restored.front_rgb)
                    observation_difference = strict_observation_difference(initial, restored)
                    record.update(restore=difference, observation_restore=observation_difference)
                    if (not difference["global_inventory_equal"] or difference["max_abs"] != 0 or
                            difference["rgb_max_difference"] != 0 or not all(observation_difference.values())):
                        save_observation_evidence(folder / ("failed_restore_%d_%d" % (target_index, attempt)), restored, restored_world)
                        raise RuntimeError("strict low-state world/inventory/RGB-D/camera/current restore failed")
                    if not all(record["camera_flags"].values()):
                        raise RuntimeError("RGB-D camera flags changed from registered policy")
                    counts["restore_passes"] += 1
                    route_trace.append(state_sample(task))
                    if not np.array_equal(route_trace[0][0], np.asarray(restored.gripper_pose)) or route_trace[0][1] != restored.gripper_open:
                        raise RuntimeError("direct state differs from Scene initial observation")
                    guides = proposed_waypoints(goals[target_index], proposed, config)
                    record["collection_guides_supervision_only"] = guides
                    execute(task, guides, route_trace[0][0][3:], gripper_shapes, external_shapes, route_trace, record, config)
                    xyz = np.asarray([s[0][:3] for s in route_trace])
                    final = task.get_observation()
                    if not np.array_equal(route_trace[-1][0], np.asarray(final.gripper_pose)) or route_trace[-1][1] != final.gripper_open:
                        raise RuntimeError("direct state differs from Scene final observation")
                    record["endpoint_error_m"] = float(np.linalg.norm(xyz[-1]-goals[target_index]))
                    record["length_m"] = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
                    record["tip_polyline_clear"] = legacy.tip_polyline_clear(xyz, centers, halves, config["tip_clearance_m"])
                    actual = crossing_signature(xyz, config)
                    sampled = legacy.resample(xyz, 24)
                    record["actual_route_type"] = actual
                    record["tip_polyline_24_clear"] = legacy.tip_polyline_clear(sampled, centers, halves, config["tip_clearance_m"])
                    if record["endpoint_error_m"] > config["endpoint_tolerance_m"] or not record["tip_polyline_clear"]:
                        raise RuntimeError("endpoint or complete tip-polyline clearance failed")
                    if not record["tip_polyline_24_clear"] or crossing_signature(sampled, config) != actual:
                        raise RuntimeError("H24 changed clearance or actual passage classification")
                    name = "target%d_route%d.npz" % (target_index, attempt)
                    record["trajectory"] = save_trace(folder/name, route_trace, dict(xyz_24=sampled, xyz_64=legacy.resample(xyz)))
                    record["duplicate_passage_type"] = None if actual is None else actual in seen
                    if actual is not None:
                        counts["classified_successes"] += 1
                        counts["duplicate_types"] += int(actual in seen)
                        seen.add(actual)
                        if all(item in PASSAGES for item in actual):
                            counts["lateral_classified_successes"] += 1
                            lateral.add(actual)
                    routes.append(parent_id+"/"+name)
                    types.append(actual)
                    counts["route_successes"] += 1
                    record["success"] = True
                except Exception as error:
                    record.update(error=repr(error), traceback=traceback.format_exc())
                    record["failed_partial_trajectory"] = save_trace(folder/("failed_target%d_attempt%d.npz" % (target_index, attempt)), route_trace)
                record["seconds"] = time.perf_counter()-tic
                add_execution_totals(counts, record)
                legacy.append_json(output/"attempts.jsonl", record)
                print(legacy.json_text({k:v for k,v in record.items() if k != "traceback"}), flush=True)
            legacy.append_json(output/"supervision.jsonl", dict(id=identifier, parent_id=parent_id, split=config["split"],
                task="rlbench_derived_two_row_reach", observation=parent_id+"/observation.npz", routes=routes,
                route_types=types, verification_only=parent_id+"/verification_only.npz", reference_set_complete=False,
                semantic_targets=dict(centers=goals.tolist(), target_index=target_index, tolerance=config["endpoint_tolerance_m"])))
            per_target.append(summarize_reference_types(identifier,types,config))
    except Exception as error:
        fatal_error = dict(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        try:
            env.shutdown()
        except Exception as error:
            shutdown_error = repr(error)
        counts["unattempted_route_proposals"] = requested_routes-counts["route_attempts"]
        status = "collector_error" if fatal_error or shutdown_error else ("collection_finished" if setup["success"] else "setup_failed")
        write_json(output/"summary.json", dict(**counts, per_target=per_target, elapsed_seconds=time.perf_counter()-started,
            setup_elapsed_seconds=setup.get("seconds"), status=status,
            fatal_error=fatal_error, shutdown_error=shutdown_error, full_robot_continuous_certificate=False,
            reference_set_complete=False, all_solution_count=None))
        write_json(output/"artifact_hashes.json", {str(path.relative_to(output)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*")) if path.is_file() and path.name != "artifact_hashes.json"})
        if shutdown_error and fatal_error is None:
            raise RuntimeError("simulator shutdown failed: " + shutdown_error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1]/"configs/observed_two_row_pilot_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.output.exists():
        parser.error("fresh output required; this feasibility version never overwrites or retries a proposal")
    collect(config, args.config, args.output)


if __name__ == "__main__":
    main()
