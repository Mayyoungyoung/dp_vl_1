"""RLBench-derived visible-target reaching around physical obstacle boxes.

Default pilot: one parent, three visible language targets, two proposed side
routes each. Opening types are obstacle-relative plane-crossing relations,
never clusters of trajectory perturbations. Simulator state/obstacle geometry,
collection guides and verification masks are supervision-only artifacts.
"""

import argparse
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from observation_collect_rlbench import (append_json, array_hash, audit_difference,
                                          native_snapshot, resample, world_audit)


MODE_NAMES = ("negative_x", "positive_x", "negative_y", "positive_y")


def tip_polyline_clear(paths, centers, halfsizes, clearance=.02):
    """Exact segment/AABB check for the TIP only, not continuous robot volume."""
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2 or paths.shape[1] != 3 or len(paths) < 2 or not np.isfinite(paths).all():
        return False
    for center, halfsize in zip(centers, halfsizes):
        lower, upper = np.asarray(center) - halfsize - clearance, np.asarray(center) + halfsize + clearance
        p0, delta = paths[:-1], np.diff(paths, axis=0)
        moving = np.abs(delta) > 1e-12
        outside = (~moving) & ((p0 < lower) | (p0 > upper))
        entry, leave = np.full_like(delta, -np.inf), np.full_like(delta, np.inf)
        np.divide(lower - p0, delta, out=entry, where=moving)
        np.divide(upper - p0, delta, out=leave, where=moving)
        near = np.minimum(entry, leave).max(axis=1)
        far = np.maximum(entry, leave).min(axis=1)
        collision = (near <= far) & (far >= 0) & (near <= 1) & ~outside.any(axis=1)
        if collision.any():
            return False
    return True


def crossing_signature(paths, centers, halfsizes, clearance=.02):
    """Classify all center-z-plane crossings per obstacle in descending order.

    +/-x apply within the obstacle's y-span; outside that span, +/-y apply.
    Every crossing must agree. Missing crossings, obstacle-interior crossings,
    or backtracking across different sides yield None, not artificial novelty.
    These are declared passage relations, not a general homotopy theorem.
    """
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim != 2 or len(paths) < 2 or paths.shape[1] != 3 or not np.isfinite(paths).all():
        return None
    signature = []
    for center, halfsize in zip(np.asarray(centers), np.asarray(halfsizes)):
        labels = []
        for a, b in zip(paths[:-1], paths[1:]):
            if not min(a[2], b[2]) <= center[2] <= max(a[2], b[2]):
                continue
            if abs(b[2] - a[2]) <= 1e-12:
                points = (a, b)
            else:
                points = (a + ((center[2] - a[2]) / (b[2] - a[2])) * (b - a),)
            for point in points:
                if point[1] < center[1] - halfsize[1] - clearance:
                    label = "negative_y"
                elif point[1] > center[1] + halfsize[1] + clearance:
                    label = "positive_y"
                elif point[0] < center[0] - halfsize[0] - clearance:
                    label = "negative_x"
                elif point[0] > center[0] + halfsize[0] + clearance:
                    label = "positive_x"
                else:
                    return None
                labels.append(label)
        if not labels or len(set(labels)) != 1:
            return None
        signature.append(labels[0])
    return tuple(signature)


def collection_waypoints(goal, centers, halfsizes, signature, side_margin=.10, vertical_margin=.07):
    points = []
    for center, halfsize, mode in zip(np.asarray(centers), np.asarray(halfsizes), signature):
        axis = 0 if mode.endswith("x") else 1
        sign = -1. if mode.startswith("negative") else 1.
        side = center.copy()
        side[axis] += sign * (halfsize[axis] + side_margin)
        above, below = side.copy(), side.copy()
        above[2] = center[2] + halfsize[2] + vertical_margin
        below[2] = center[2] - halfsize[2] - vertical_margin
        points.extend([above, below])
    points.append(np.asarray(goal))
    return points


def full_audit(task, obstacles):
    """Include root obstacles and the actual complete simulator inventory."""
    state = world_audit(task)
    for obstacle in obstacles:
        linear, angular = obstacle.get_velocity()
        state["_extra_" + obstacle.get_name()] = dict(
            pose=obstacle.get_pose(), color=obstacle.get_color(), velocity=list(linear) + list(angular),
            bounding_box=obstacle.get_bounding_box(),
            flags=[int(obstacle.is_collidable()), int(obstacle.is_respondable()), int(obstacle.is_dynamic())])
    inventory = sorted((obj.get_name(), int(obj.get_handle()), int(obj.get_type().value))
                       for obj in task._pyrep.get_objects_in_tree(exclude_base=False))
    return dict(state=state, inventory=inventory)


def full_snapshot(task, obstacles):
    return dict(core=native_snapshot(task),
                obstacles=[dict(shape=shape, tree=shape.get_configuration_tree(), color=shape.get_color(),
                                collidable=shape.is_collidable(), respondable=shape.is_respondable(),
                                dynamic=shape.is_dynamic()) for shape in obstacles])


def restore_full_snapshot(task, snapshot):
    """Established native restore, with root obstacles restored before start."""
    core, robot = snapshot["core"], task._robot
    robot.gripper.release()
    task._pyrep.stop()
    task._task.restore_state(core["task"])
    task._pyrep.set_configuration_tree(core["arm_tree"])
    task._pyrep.set_configuration_tree(core["gripper_tree"])
    robot.arm.set_joint_positions(core["arm_joints"], disable_dynamics=True)
    robot.gripper.set_joint_positions(core["gripper_joints"], disable_dynamics=True)
    robot.arm.set_joint_target_positions(core["arm_joints"])
    robot.gripper.set_joint_target_positions(core["gripper_joints"])
    robot.arm.set_joint_target_velocities([0.] * len(robot.arm.joints))
    robot.gripper.set_joint_target_velocities([0.] * len(robot.gripper.joints))
    for shape, color in core["shape_colors"]:
        shape.set_color(color)
    for entry in snapshot["obstacles"]:
        task._pyrep.set_configuration_tree(entry["tree"])
        entry["shape"].set_color(entry["color"])
        entry["shape"].set_collidable(entry["collidable"])
        entry["shape"].set_respondable(entry["respondable"])
        entry["shape"].set_dynamic(entry["dynamic"])
    task._pyrep.start()
    robot.arm.set_control_loop_enabled(True)
    for _ in range(10):
        task._scene.step()


def compare_restore(reference, current, original_rgb, current_rgb):
    same_inventory = reference["inventory"] == current["inventory"]
    difference = audit_difference(reference["state"], current["state"]) if same_inventory else dict(max_abs=None)
    return dict(**difference, global_inventory_equal=same_inventory,
                rgb_max_difference=int(np.abs(original_rgb.astype(np.int16) - current_rgb.astype(np.int16)).max()))


def robot_collision_check(task, gripper_shapes, external_shapes):
    if task._robot.arm.check_arm_collision():
        return "arm_environment"
    # Arm's collision collection need not include the fingers. Explicitly check
    # collidable gripper links against external bodies, excluding robot links.
    for gripper_shape in gripper_shapes:
        for external in external_shapes:
            if gripper_shape.check_collision(external):
                return "%s:%s" % (gripper_shape.get_name(), external.get_name())
    return None


def self_test():
    centers, halfsizes = np.array([[0., 0., 1.]]), np.array([[.1, .1, .05]])
    for side in ("negative_x", "positive_x", "negative_y", "positive_y"):
        path = np.vstack([[0., 0., 1.4], collection_waypoints([0., 0., .6], centers, halfsizes, (side,))])
        assert tip_polyline_clear(path, centers, halfsizes)
        assert crossing_signature(path, centers, halfsizes) == (side,)
        assert crossing_signature(path[::-1], centers, halfsizes) == (side,)
    assert not tip_polyline_clear(np.array([[0., 0., 1.4], [0., 0., .6]]), centers, halfsizes)
    assert crossing_signature(np.array([[0., 0., 1.4], [0., 0., .6]]), centers, halfsizes) is None
    # Safe waypoints do not make a segment through the expanded box safe.
    assert not tip_polyline_clear(np.array([[-.2, 0., 1.], [.2, 0., 1.]]), centers, halfsizes)
    print(json.dumps(dict(pure_geometry_self_test="passed", simulator_launched=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parents", type=int, default=1)
    parser.add_argument("--dev-parents", type=int, default=1)
    parser.add_argument("--seed", type=int, default=271000)
    parser.add_argument("--obstacles", type=int, choices=(1, 2), default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--side-margin", type=float, default=.10)
    parser.add_argument("--tip-clearance", type=float, default=.02)
    parser.add_argument("--endpoint-tolerance", type=float, default=.03)
    parser.add_argument("--restore-atol", type=float, default=0.)
    parser.add_argument("--minimum-target-pixels", type=int, default=10)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output is None or args.output.exists() or args.parents < 1 or not 0 <= args.dev_parents <= args.parents:
        parser.error("new --output directory, positive parents and valid dev-parent count required")
    from pyrep.const import ObjectType, PrimitiveShape
    from pyrep.objects.shape import Shape
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.const import colors
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks.reach_target import ReachTarget
    config = ObservationConfig()
    config.set_all(False)
    config.front_camera.rgb = True
    config.front_camera.depth = True
    config.front_camera.depth_in_meters = True
    config.front_camera.mask = True  # Verification only; never exported as input.
    config.front_camera.masks_as_one_channel = True
    config.front_camera.image_size = (args.image_size, args.image_size)
    config.gripper_pose = config.gripper_open = True
    config.task_low_dim_state = False
    env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=config, headless=True)
    args.output.mkdir(parents=True)
    proposals = list(itertools.product(("negative_x", "positive_x"), repeat=args.obstacles))
    manifest = dict(benchmark="RLBench-derived obstacle-passage multi-target reaching v1", original_benchmark_result=False,
                    rlbench_revision="02720bba4c73fe02eb75df946b8791b806028a9d", pyrep_revision="8f420be8064b1970aae18a9cfbc978dfb15747ef",
                    source_sha256=__import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
                    input_contract=["RGB", "instruction", "depth", "camera", "current gripper pose/open"],
                    supervision_only=["physical obstacle truth", "target coordinates", "route labels", "future trajectories", "guide points", "segmentation masks"],
                    type_definition="all crossings of each obstacle center-z plane agree on +/-x or +/-y; y-outside-span takes precedence; ordered high-to-low obstacles",
                    proposed_type_sequences=proposals, reference_set_complete=False, number_of_all_continuous_solutions=None,
                    acceptance=dict(endpoint_tolerance_m=args.endpoint_tolerance, tip_polyline_clearance_m=args.tip_clearance,
                                    all_simulated_steps_arm_environment_collision=False, all_simulated_steps_gripper_external_collision=False,
                                    initial_state_max_abs_tolerance=args.restore_atol, initial_rgb_max_difference=0,
                                    global_object_inventory_must_match=True),
                    continuous_whole_robot_collision_certified=False, parents_requested=args.parents,
                    dev_parents=args.dev_parents, seed=args.seed, obstacles=args.obstacles)
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    total = successes = restores = classified = duplicate_types = parents_collected = 0
    env.launch()
    try:
        # Root shapes are created while stopped so native stop/start retains
        # them. They never change Task.get_state's task-tree object count.
        env._pyrep.stop()
        task = env.get_task(ReachTarget)
        sizes = [np.array([.14, .18, .06])] * args.obstacles
        obstacles = []
        for index, size in enumerate(sizes):
            shape = Shape.create(PrimitiveShape.CUBOID, size.tolist(), static=True, respondable=True,
                                 position=[0., 0., -5. - index], color=[.32, .34, .38])
            shape.set_name("derived_passage_obstacle_%d" % index)
            shape.set_collidable(True)
            shape.set_detectable(True)
            obstacles.append(shape)
        env._pyrep.start()
        task._robot.arm.set_control_loop_enabled(True)
        for parent in range(args.parents):
            parent_id = "obstacle_reach_%06d" % (args.seed + parent)
            split = "TRAIN" if parent < args.parents - args.dev_parents else "DEV_MODEL"
            folder = args.output / parent_id
            folder.mkdir()
            np.random.seed(args.seed + parent)
            task.set_variation(parent % task.variation_count())
            random_state = np.random.get_state()
            try:
                _, initial = task.reset()
                targets = [task._task.target, task._task.distractor0, task._task.distractor1]
                goals = np.asarray([target.get_position() for target in targets])
                start_xyz = np.asarray(initial.gripper_pose[:3])
                gap = start_xyz[2] - goals[:, 2].max()
                if gap < (.28 if args.obstacles == 1 else .55):
                    raise RuntimeError("insufficient vertical workspace for the declared physical obstacle layout")
                center_xy = .65 * start_xyz[:2] + .35 * goals[:, :2].mean(axis=0)
                heights = np.linspace(start_xyz[2], goals[:, 2].max(), args.obstacles + 2)[1:-1]
                centers = np.array([np.r_[center_xy, height] for height in heights])
                halfsizes = np.asarray(sizes) / 2
                for shape, center in zip(obstacles, centers):
                    shape.set_position(center.tolist())
                    shape.set_orientation([0., 0., 0.])
                snapshot = full_snapshot(task, obstacles)
                restore_full_snapshot(task, snapshot)
                initial = task.get_observation()
                reference = full_audit(task, obstacles)
                visible = [int(np.sum(initial.front_mask == target.get_handle())) for target in targets]
                if min(visible) < args.minimum_target_pixels:
                    raise RuntimeError("target visibility check failed: pixels=" + str(visible))
                robot_objects = task._robot.arm.get_objects_in_tree(exclude_base=False) + task._robot.gripper.get_objects_in_tree(exclude_base=False)
                robot_handles = {obj.get_handle() for obj in robot_objects}
                gripper_shapes = [obj for obj in task._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE, exclude_base=False) if obj.is_collidable()]
                external_shapes = [obj for obj in task._pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE, exclude_base=False)
                                   if obj.is_collidable() and obj.get_handle() not in robot_handles]
                collision = robot_collision_check(task, gripper_shapes, external_shapes)
                if collision:
                    raise RuntimeError("initial full robot collision: " + collision)
                if not gripper_shapes:
                    raise RuntimeError("no collidable gripper shapes were found for explicit collision validation")
            except Exception as exc:
                record = dict(parent_id=parent_id, phase="parent_setup", success=False, error=repr(exc), traceback=traceback.format_exc())
                append_json(args.output / "attempts.jsonl", record)
                print(json.dumps({key: value for key, value in record.items() if key != "traceback"}), flush=True)
                continue
            parents_collected += 1
            Image.fromarray(initial.front_rgb).save(folder / "front.png")
            np.savez_compressed(folder / "observation.npz", depth=initial.front_depth,
                                gripper_pose=initial.gripper_pose, gripper_open=initial.gripper_open,
                                camera_intrinsics=initial.misc["front_camera_intrinsics"], camera_extrinsics=initial.misc["front_camera_extrinsics"])
            np.savez_compressed(folder / "verification_only.npz", mask=initial.front_mask, obstacle_centers=centers,
                                obstacle_halfsizes=halfsizes, target_centers=goals, target_visible_pixels=visible)
            (folder / "restore_reference.json").write_text(json.dumps(reference), encoding="utf-8")
            palette = np.asarray([color[1] for color in colors])
            for target_index, target in enumerate(targets):
                goal = goals[target_index]
                color_index = int(np.linalg.norm(palette - target.get_color(), axis=1).argmin())
                identifier = parent_id + "_target%d" % target_index
                instruction = "Move the gripper to touch the %s sphere while avoiding the gray obstacle%s." % (colors[color_index][0], "s" if args.obstacles > 1 else "")
                append_json(args.output / "observations.jsonl", dict(id=identifier, parent_id=parent_id, split=split,
                                                                     image=parent_id + "/front.png", instruction=instruction))
                routes, route_types, accepted_types = [], [], set()
                for attempt, proposed in enumerate(proposals):
                    total += 1
                    tic = time.perf_counter()
                    record = dict(parent_id=parent_id, input_id=identifier, attempt=attempt, success=False,
                                  proposed_type_supervision_only=proposed, actual_route_type=None, simulated_steps=0)
                    poses, opens = [], []
                    try:
                        np.random.set_state(random_state)
                        task.reset()
                        restore_full_snapshot(task, snapshot)
                        restored = task.get_observation()
                        difference = compare_restore(reference, full_audit(task, obstacles), initial.front_rgb, restored.front_rgb)
                        record["restore"] = difference
                        if not difference["global_inventory_equal"] or difference["max_abs"] > args.restore_atol or difference["rgb_max_difference"] != 0:
                            raise RuntimeError("full initial inventory/state/RGB equality failed")
                        restores += 1
                        arm = task._robot.arm
                        tip = np.asarray(arm.get_tip().get_pose())
                        poses.append(tip)
                        opens.append(restored.gripper_open)
                        waypoints = collection_waypoints(goal, centers, halfsizes, proposed, args.side_margin,
                                                         .07 if args.obstacles == 1 else .035)
                        record["collection_guides_supervision_only"] = [point.tolist() for point in waypoints[:-1]]
                        for waypoint in waypoints:
                            path = arm.get_path(waypoint.tolist(), quaternion=tip[3:].tolist(), ignore_collisions=False)
                            done = False
                            for step in range(1000):
                                done = path.step()
                                task._scene.step()
                                observed = task.get_observation()
                                poses.append(np.asarray(observed.gripper_pose))
                                opens.append(observed.gripper_open)
                                record["simulated_steps"] += 1
                                collision = robot_collision_check(task, gripper_shapes, external_shapes)
                                if collision:
                                    record["collision_pair"] = collision
                                    raise RuntimeError("robot collision during simulated motion")
                                if done:
                                    break
                            if not done:
                                raise RuntimeError("segment exceeded 1000 simulation steps")
                        xyz = np.asarray(poses)[:, :3]
                        record["endpoint_error_m"] = float(np.linalg.norm(xyz[-1] - goal))
                        record["length_m"] = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
                        record["tip_polyline_clear"] = bool(tip_polyline_clear(xyz, centers, halfsizes, args.tip_clearance))
                        if record["endpoint_error_m"] > args.endpoint_tolerance or not record["tip_polyline_clear"]:
                            raise RuntimeError("endpoint or continuous tip-polyline clearance failed")
                        actual = crossing_signature(xyz, centers, halfsizes, args.tip_clearance)
                        record["actual_route_type"] = actual
                        duplicate = actual in accepted_types if actual is not None else None
                        if actual is not None:
                            classified += 1
                            duplicate_types += int(duplicate)
                            accepted_types.add(actual)
                        filename = parent_id + "/target%d_route%d.npz" % (target_index, attempt)
                        np.savez_compressed(args.output / filename, gripper_pose=np.asarray(poses), gripper_open=np.asarray(opens), xyz_64=resample(np.asarray(poses)))
                        routes.append(filename)
                        route_types.append(actual)
                        successes += 1
                        record.update(success=True, duplicate_passage_type=duplicate, route_file=filename,
                                      trajectory_sha256=array_hash(np.asarray(poses)))
                    except Exception as exc:
                        record.update(error=repr(exc), traceback=traceback.format_exc())
                        if poses:
                            filename = parent_id + "/failed_target%d_attempt%d.npz" % (target_index, attempt)
                            np.savez_compressed(args.output / filename, gripper_pose=np.asarray(poses), gripper_open=np.asarray(opens))
                            record["failed_partial_route"] = filename
                    record["seconds"] = time.perf_counter() - tic
                    append_json(args.output / "attempts.jsonl", record)
                    print(json.dumps({key: value for key, value in record.items() if key != "traceback"}), flush=True)
                append_json(args.output / "supervision.jsonl", dict(id=identifier, parent_id=parent_id, split=split,
                    observation=parent_id + "/observation.npz", routes=routes, route_types=route_types,
                    task="rlbench_derived_obstacle_multitarget_reach", reference_set_complete=False,
                    verification_only=parent_id + "/verification_only.npz",
                    semantic_targets=dict(centers=goals.tolist(), target_index=target_index, tolerance=args.endpoint_tolerance)))
    finally:
        env.shutdown()
    summary = dict(status="collection_finished", parents_requested=args.parents, parents_collected=parents_collected,
                   attempts=total, successes=successes, restore_passes=restores, classified_successes=classified,
                   duplicate_passage_type_successes=duplicate_types, elapsed_seconds=time.perf_counter() - started,
                   all_solution_count=None, continuous_whole_robot_collision_certified=False)
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
