"""RLBench-derived visible-target reaching around physical obstacle boxes.

Default v2 pilot: one parent, three visible language targets, four proposed side
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

from observation_collect_rlbench import (array_hash, audit_difference,
                                          native_snapshot, resample, world_audit)


MODE_NAMES = ("negative_x", "positive_x", "negative_y", "positive_y")
RANDOMIZED_V3 = dict(
    region_centers=[[.08, .18, .84], [.28, .02, .84], [.44, -.05, .83]],
    region_halfwidths=[[.04, .05, .012], [.045, .045, .012], [.04, .05, .012]],
    box_xy_halfwidth=[.03, .04], box_z_offset_from_initial_tip=[-.27, -.23],
    box_size_lower=[.12, .15, .05], box_size_upper=[.17, .21, .07],
    object_to_region_mapping="independent random permutation of three target identities",
    layout_prng="numpy.random.default_rng(parent_seed + 80000)")


def randomized_layout(parent_seed, start_xyz):
    """Declared v3 spatial distribution; labels never become model inputs."""
    rng = np.random.default_rng(parent_seed + 80000)
    offsets = rng.uniform(-1., 1., (3, 3)) * np.asarray(RANDOMIZED_V3["region_halfwidths"])
    region_positions = np.asarray(RANDOMIZED_V3["region_centers"]) + offsets
    permutation = rng.permutation(3)
    goals = region_positions[permutation]
    base_xy = .65 * np.asarray(start_xyz)[:2] + .35 * goals[:, :2].mean(axis=0)
    center_xy = base_xy + rng.uniform(-1., 1., 2) * RANDOMIZED_V3["box_xy_halfwidth"]
    height = start_xyz[2] + rng.uniform(*RANDOMIZED_V3["box_z_offset_from_initial_tip"])
    sizes = rng.uniform(RANDOMIZED_V3["box_size_lower"], RANDOMIZED_V3["box_size_upper"])
    return dict(goals=goals, centers=np.array([np.r_[center_xy, height]]), sizes=np.array([sizes]),
                target_region_indices=permutation, region_positions=region_positions,
                parent_seed=parent_seed, layout_prng_seed=parent_seed + 80000)


def set_box_size(shape, desired):
    """Scale a persistent box in place; never change the object inventory."""
    bounds = np.asarray(shape.get_bounding_box()).reshape(3, 2)
    current = bounds[:, 1] - bounds[:, 0]
    desired = np.asarray(desired)
    if not np.allclose(current, desired, atol=1e-7, rtol=0):
        shape.scale_object(*(desired / current).tolist())
    actual = np.asarray(shape.get_bounding_box()).reshape(3, 2)
    if not np.allclose(actual[:, 1] - actual[:, 0], desired, atol=1e-6, rtol=0):
        raise RuntimeError("physical obstacle dimensions differ from declared dimensions")
    if not np.allclose(actual.sum(axis=1), 0, atol=1e-6, rtol=0):
        raise RuntimeError("physical obstacle local bounding box is not centered")
    return actual[:, 1] - actual[:, 0]


def observed_box_depth_audit(observation, obstacles, centers, halfsizes):
    """Validation-only check for stale rendered meshes after simulator scale.

    Signed camera intrinsics and optical-axis metric depth reconstruct each
    actually observed obstacle pixel. Geometry/masks are never model inputs.
    """
    intrinsics = np.asarray(observation.misc["front_camera_intrinsics"])
    extrinsics = np.asarray(observation.misc["front_camera_extrinsics"])
    audits = []
    for obstacle, center, halfsize in zip(obstacles, centers, halfsizes):
        yy, xx = np.where(observation.front_mask == obstacle.get_handle())
        if len(xx) < 10:
            raise RuntimeError("physical obstacle is not sufficiently visible for RGB-D consistency validation")
        pixels = np.stack([xx, yy, np.ones_like(xx)], axis=-1)
        camera_xyz = (pixels @ np.linalg.inv(intrinsics).T) * observation.front_depth[yy, xx, None]
        world_xyz = camera_xyz @ extrinsics[:3, :3].T + extrinsics[:3, 3]
        distance = np.linalg.norm(np.maximum(np.abs(world_xyz - center) - halfsize, 0.), axis=-1)
        record = dict(object_name=obstacle.get_name(), visible_pixels=len(xx),
                      maximum_observed_distance_outside_box_m=float(distance.max()),
                      points_outside_one_cm=int(np.sum(distance > .01)))
        audits.append(record)
        if not np.isfinite(distance).all() or record["points_outside_one_cm"]:
            raise RuntimeError("rendered obstacle depth is inconsistent with physical box: " + json_text(record))
    return audits


def json_ready(value):
    """One strict serialization boundary for every collector JSON artifact."""
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.generic):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def json_text(value, indent=None):
    return json.dumps(json_ready(value), ensure_ascii=False, allow_nan=False, indent=indent)


def append_json(path, value):
    text = json_text(value)
    with Path(path).open("a", encoding="utf-8") as stream:
        stream.write(text + "\n")


def serialization_check(folder):
    """Exercise every JSON schema using saved simulator arrays, without a sim.

    Success/failure records below are serialization fixtures, not fabricated
    collection results. Only a compact preflight report is emitted.
    """
    folder = Path(folder)
    with np.load(folder / "observation.npz", allow_pickle=False) as archive:
        observed = {key: archive[key] for key in archive.files}
    with np.load(folder / "verification_only.npz", allow_pickle=False) as archive:
        verification = {key: archive[key] for key in archive.files}
    center, halfsize = verification["obstacle_centers"][0], verification["obstacle_halfsizes"][0]
    reference = dict(state={"_robot": dict(gripper_pose=observed["gripper_pose"], gripper_open=observed["gripper_open"]),
                           "_extra_obstacle": dict(pose=np.r_[center, [0., 0., 0., 1.]], color=np.array([.32, .34, .38]),
                               bounding_box=np.ravel(np.stack([-halfsize, halfsize], axis=1)),
                               velocity=np.zeros(6), flags=np.array([True, True, False]))},
                     inventory=[("fixture_obstacle", np.int64(1), np.int32(0))])
    restore = dict(max_abs=np.float64(0.), worst_field=None, same_object_inventory=np.bool_(True),
                   global_inventory_equal=np.bool_(True), rgb_max_difference=np.int16(0))
    success = dict(parent_id=folder.name, attempt=np.int64(0), success=np.bool_(True), restore=restore,
                   actual_route_type=("negative_x",), collection_guides_supervision_only=verification["obstacle_centers"],
                   endpoint_error_m=np.float32(0.), length_m=np.float64(1.), tip_polyline_clear=np.bool_(True),
                   tip_polyline_24_clear=np.bool_(True), duplicate_passage_type=np.bool_(False), trajectory_sha256="fixture")
    failure = dict(parent_id=folder.name, attempt=np.int64(0), success=np.bool_(False), error="serialization fixture",
                   restore=restore, failed_partial_route=Path("fixture.npz"), traceback="fixture")
    supervision = dict(id=folder.name + "_target0", parent_id=folder.name, split="DEV_MODEL", routes=["fixture.npz"],
                        route_types=[("negative_x",), None], reference_set_complete=False,
                        semantic_targets=dict(centers=verification["target_centers"], target_index=np.int64(0), tolerance=np.float32(.03)))
    summary = dict(status="serialization_fixture_only", parents_requested=np.int64(1), successes=np.int64(0),
                   elapsed_seconds=np.float64(0), all_solution_count=None, continuous_whole_robot_collision_certified=False)
    manifest = json.loads((folder.parent / "manifest.json").read_text())
    observation_manifest = dict(id=folder.name + "_target0", parent_id=folder.name, split="DEV_MODEL",
                                image=folder.name + "/front.png", instruction="Serialization fixture instruction.")
    fixtures = dict(manifest=manifest, observations=observation_manifest, restore_reference=reference, attempt_success=success,
                    attempt_failure=failure, supervision=supervision, summary=summary,
                    all_actual_observation_arrays=observed, all_actual_verification_arrays=verification,
                    randomized_layout_sampling=randomized_layout(271100, observed["gripper_pose"][:3]),
                    canonical_restore_diagnostic=dict(difference_from_original=restore, audit=reference),
                    observation_validation=dict(canonicalization_passes=2, physical_obstacle_observed_depth=[dict(
                        object_name="fixture_obstacle", visible_pixels=np.int64(1477),
                        maximum_observed_distance_outside_box_m=np.float32(.002), points_outside_one_cm=np.int64(0))]))
    hashes = {}
    import hashlib
    import tempfile
    with tempfile.TemporaryDirectory() as temporary:
        # Exercise both JSON and the same JSONL writer used by the real job.
        destination = Path(temporary) / "schemas.jsonl"
        for name, fixture in fixtures.items():
            payload = json_text(fixture)
            json.loads(payload)
            append_json(destination, fixture)
            hashes[name] = hashlib.sha256(payload.encode()).hexdigest()
        assert len(destination.read_text().splitlines()) == len(fixtures)
    print(json_text(dict(serialization_preflight="passed", simulator_launched=False, source=str(folder),
                         schemas_checked=sorted(fixtures), fixture_hashes=hashes,
                         actual_visible_target_pixels=verification["target_visible_pixels"])))


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
    from pyrep.backend import sim
    from pyrep.const import ObjectType
    state = world_audit(task)
    for obstacle in obstacles:
        linear, angular = obstacle.get_velocity()
        fields = dict(
            pose=obstacle.get_pose(), color=obstacle.get_color(), velocity=list(linear) + list(angular),
            bounding_box=obstacle.get_bounding_box(),
            flags=[int(obstacle.is_collidable()), int(obstacle.is_respondable()), int(obstacle.is_dynamic())])
        state["_extra_" + obstacle.get_name()] = {key: np.asarray(value).tolist() for key, value in fields.items()}
    # PyRep's object-wrapper iterator silently skips unsupported object types
    # (e.g. lights). Raw simulator handles preserve the ENTIRE inventory.
    handles = sim.simGetObjectsInTree(sim.sim_handle_scene, ObjectType.ALL.value, 0)
    inventory = sorted((sim.simGetObjectName(handle), int(handle), int(sim.simGetObjectType(handle)))
                       for handle in handles)
    # Include global cameras/lights and every robot link, not only the task
    # subtree. This diagnoses image changes that coarse task state misses.
    for name, handle, object_type in inventory:
        state["_global_pose_" + name] = dict(pose=sim.simGetObjectPosition(handle, -1) +
                                           sim.simGetObjectQuaternion(handle, -1))
    return dict(state=state, inventory=inventory)


def full_snapshot(task, obstacles):
    return dict(core=native_snapshot(task),
                obstacles=[dict(shape=shape, tree=shape.get_configuration_tree(), color=shape.get_color(),
                                collidable=shape.is_collidable(), respondable=shape.is_respondable(),
                                dynamic=shape.is_dynamic(),
                                size=np.diff(np.asarray(shape.get_bounding_box()).reshape(3, 2), axis=1).ravel())
                           for shape in obstacles])


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
        set_box_size(entry["shape"], entry["size"])
        task._pyrep.set_configuration_tree(entry["tree"])
        entry["shape"].set_color(entry["color"])
        entry["shape"].set_collidable(entry["collidable"])
        entry["shape"].set_respondable(entry["respondable"])
        entry["shape"].set_dynamic(entry["dynamic"])
    task._pyrep.start()
    robot.arm.set_control_loop_enabled(True)
    for _ in range(10):
        task._scene.step()


def canonical_restore(task, snapshot, passes):
    # CoppeliaSim 4.1/PyRep can serve a stale RGB/depth mesh on the first native
    # restart after scaling. A fixed warm-up pass precedes the canonical pass
    # in v3, identically for reference and EVERY candidate. No adaptive choice
    # of a convenient reference or relaxed equality tolerance is allowed.
    for index in range(passes):
        restore_full_snapshot(task, snapshot)
        if index < passes - 1:
            # The renderer must actually execute to refresh the scaled mesh;
            # two physics restarts without an intervening render are not a
            # warm-up. This observation is discarded on EVERY invocation.
            task.get_observation()


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
        sampled = resample(path, count=24)
        assert tip_polyline_clear(sampled, centers, halfsizes)
        assert crossing_signature(sampled, centers, halfsizes) == (side,)
    assert not tip_polyline_clear(np.array([[0., 0., 1.4], [0., 0., .6]]), centers, halfsizes)
    assert crossing_signature(np.array([[0., 0., 1.4], [0., 0., .6]]), centers, halfsizes) is None
    # Safe waypoints do not make a segment through the expanded box safe.
    assert not tip_polyline_clear(np.array([[-.2, 0., 1.], [.2, 0., 1.]]), centers, halfsizes)
    samples = [randomized_layout(271100 + index, np.array([.28, .01, 1.47])) for index in range(128)]
    assert len({json_text(sample) for sample in samples}) == 128
    assert len({tuple(sample["target_region_indices"]) for sample in samples}) == 6
    for index, sample in enumerate(samples):
        assert json_text(sample) == json_text(randomized_layout(271100 + index, np.array([.28, .01, 1.47])))
        delta = np.abs(sample["region_positions"] - RANDOMIZED_V3["region_centers"])
        assert np.all(delta <= RANDOMIZED_V3["region_halfwidths"])
        assert np.all(sample["sizes"] >= RANDOMIZED_V3["box_size_lower"])
        assert np.all(sample["sizes"] <= RANDOMIZED_V3["box_size_upper"])
        assert np.array_equal(sample["goals"], sample["region_positions"][sample["target_region_indices"]])
    from types import SimpleNamespace
    mock_box = SimpleNamespace(get_handle=lambda: 7, get_name=lambda: "test_box")
    observed = SimpleNamespace(front_mask=np.full((4, 4), 7), front_depth=np.ones((4, 4)),
        misc=dict(front_camera_intrinsics=np.array([[-8., 0., 1.5], [0., -8., 1.5], [0., 0., 1.]]),
                  front_camera_extrinsics=np.eye(4)))
    assert observed_box_depth_audit(observed, [mock_box], [[0., 0., 1.]], [[.25, .25, .02]])[0]["points_outside_one_cm"] == 0
    observed.front_depth[0, 0] = 3.
    try:
        observed_box_depth_audit(observed, [mock_box], [[0., 0., 1.]], [[.25, .25, .02]])
    except RuntimeError as error:
        assert "rendered obstacle depth" in str(error)
    else:
        raise AssertionError("stale/wrong depth must be rejected by the geometry auditor")
    events = []
    original_restore = globals()["restore_full_snapshot"]
    try:
        globals()["restore_full_snapshot"] = lambda task, snapshot: events.append("native_restore_10steps")
        mock_task = SimpleNamespace(get_observation=lambda: events.append("explicit_render_discarded"))
        canonical_restore(mock_task, {}, passes=2)
        assert events == ["native_restore_10steps", "explicit_render_discarded", "native_restore_10steps"]
    finally:
        globals()["restore_full_snapshot"] = original_restore
    print(json_text(dict(pure_geometry_self_test="passed", simulator_launched=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parents", type=int, default=1)
    parser.add_argument("--dev-parents", type=int, default=1)
    parser.add_argument("--seed", type=int, default=271000)
    parser.add_argument("--obstacles", type=int, choices=(1, 2), default=1)
    parser.add_argument("--target-layout", choices=("natural", "safe_lowered", "safe_low_all_v2", "safe_randomized_v3"), default="safe_low_all_v2")
    parser.add_argument("--passages", nargs="+", choices=MODE_NAMES,
                        help="predeclared proposal sides; v2 defaults to +/-x and +/-y, older layouts to +/-x")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--side-margin", type=float, default=.10)
    parser.add_argument("--tip-clearance", type=float, default=.02)
    parser.add_argument("--endpoint-tolerance", type=float, default=.03)
    parser.add_argument("--restore-atol", type=float, default=0.)
    parser.add_argument("--minimum-target-pixels", type=int, default=10)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--serialization-check", type=Path, help="saved parent folder; checks actual array types without starting simulation")
    parser.add_argument("--diagnostic-restores", type=int, default=0,
                        help="save additional canonical restores before attempts; never replace the original reference")
    parser.add_argument("--canonicalization-passes", type=int, choices=(1, 2),
                        help="v3 defaults to two fixed native restores (mesh warm-up then canonical); older layouts use one")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.serialization_check:
        serialization_check(args.serialization_check)
        return
    if args.output is None or args.output.exists() or args.parents < 1 or not 0 <= args.dev_parents <= args.parents:
        parser.error("new --output directory, positive parents and valid dev-parent count required")
    if args.target_layout == "safe_randomized_v3" and args.obstacles != 1:
        parser.error("safe_randomized_v3 currently declares exactly one physical obstacle")
    if args.diagnostic_restores < 0:
        parser.error("diagnostic restore count must be nonnegative")
    if args.canonicalization_passes is None:
        args.canonicalization_passes = 2 if args.target_layout == "safe_randomized_v3" else 1
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
    sides = tuple(args.passages) if args.passages else (MODE_NAMES if args.target_layout in ("safe_low_all_v2", "safe_randomized_v3") else MODE_NAMES[:2])
    if len(set(sides)) != len(sides):
        parser.error("duplicate proposal side")
    proposals = list(itertools.product(sides, repeat=args.obstacles))
    layout_descriptions = dict(natural="unchanged original ReachTarget sphere placement; equally spaced vertical obstacle centers",
        safe_lowered="preserved v1: sphere centers near [.08,.18,.84], [.28,.02,.96], [.44,-.05,.83] with seeded offsets; equally spaced vertical obstacle centers",
        safe_low_all_v2="new v2: middle sphere lowered from .96 to .84m, other centers preserved; single box centered .25m below initial tip; default four side proposals",
        safe_randomized_v3="new v3: independently randomized continuous target locations in three declared regions, identity-region permutation, and variable box center/size; four side proposals")
    manifest = dict(benchmark="RLBench-derived obstacle-passage multi-target reaching", original_benchmark_result=False,
                    setting_version=args.target_layout,
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
                    joint_state_traces_recorded=True,
                    diagnostic_restores=args.diagnostic_restores, diagnostic_restores_do_not_replace_reference=True,
                    canonicalization_passes=args.canonicalization_passes,
                    render_protocol_version=("native_double_restart_explicit_render_warmup_v3" if args.canonicalization_passes == 2 else
                                             "native_single_restart_v1"),
                    render_warmup_observations_discarded=max(0, args.canonicalization_passes - 1),
                    renderer_consistency_check="all segmentation-labelled box depth pixels within 1cm of the physical AABB; validation only",
                    dev_parents=args.dev_parents, seed=args.seed, obstacles=args.obstacles,
                    target_layout=args.target_layout,
                    target_layout_description=layout_descriptions[args.target_layout],
                    randomized_layout_distribution=RANDOMIZED_V3 if args.target_layout == "safe_randomized_v3" else None,
                    not_a_matched_comparison_to_other_layout_versions=True)
    (args.output / "manifest.json").write_text(json_text(manifest, indent=2), encoding="utf-8")
    started = time.perf_counter()
    total = successes = restores = classified = duplicate_types = parents_collected = parent_setup_failures = 0
    env.launch()
    try:
        # Root shapes are created while stopped so native stop/start retains
        # them. They never change Task.get_state's task-tree object count.
        env._pyrep.stop()
        task = env.get_task(ReachTarget)
        # TaskEnvironment.__init__ starts simulation internally after loading
        # the task. Stop AGAIN before adding persistent root obstacle objects.
        env._pyrep.stop()
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
            layout_metadata = None
            initial = None
            try:
                _, initial = task.reset()
                targets = [task._task.target, task._task.distractor0, task._task.distractor1]
                start_xyz = np.asarray(initial.gripper_pose[:3])
                if args.target_layout == "safe_randomized_v3":
                    layout_metadata = randomized_layout(args.seed + parent, start_xyz)
                    for target, position in zip(targets, layout_metadata["goals"]):
                        target.set_position(position.tolist())
                    # Scaling is in-place and every native restore checks the
                    # actual dimensions; no hidden object creation/deletion.
                    sizes = [set_box_size(shape, size) for shape, size in zip(obstacles, layout_metadata["sizes"])]
                if args.target_layout in ("safe_lowered", "safe_low_all_v2"):
                    # Original ReachTarget may place a sphere almost at the
                    # initial wrist height, leaving no room for this independent
                    # obstacle setting. Explicit derived layouts retain object
                    # identities/colors and change only their physical poses.
                    layout_rng = np.random.default_rng(args.seed + parent + 80000)
                    middle_height = .96 if args.target_layout == "safe_lowered" else .84
                    safe_positions = np.array([[.08, .18, .84], [.28, .02, middle_height], [.44, -.05, .83]])
                    safe_positions += layout_rng.uniform([-.012, -.012, -.008], [.012, .012, .008], (3, 3))
                    for target, position in zip(targets, safe_positions):
                        target.set_position(position.tolist())
                goals = np.asarray([target.get_position() for target in targets])
                gap = start_xyz[2] - goals[:, 2].max()
                if gap < (.28 if args.obstacles == 1 else .55):
                    raise RuntimeError("insufficient vertical workspace for the declared physical obstacle layout")
                center_xy = .65 * start_xyz[:2] + .35 * goals[:, :2].mean(axis=0)
                heights = (np.array([start_xyz[2] - .25]) if args.obstacles == 1 and args.target_layout == "safe_low_all_v2" else
                           np.linspace(start_xyz[2], goals[:, 2].max(), args.obstacles + 2)[1:-1])
                centers = (layout_metadata["centers"] if layout_metadata is not None else
                           np.array([np.r_[center_xy, height] for height in heights]))
                halfsizes = np.asarray(sizes) / 2
                for shape, center in zip(obstacles, centers):
                    shape.set_position(center.tolist())
                    shape.set_orientation([0., 0., 0.])
                snapshot = full_snapshot(task, obstacles)
                canonical_restore(task, snapshot, args.canonicalization_passes)
                initial = task.get_observation()
                for shape, center, halfsize in zip(obstacles, centers, halfsizes):
                    actual_bounds = np.asarray(shape.get_bounding_box()).reshape(3, 2)
                    if not np.allclose(actual_bounds, np.stack([-halfsize, halfsize], axis=1), atol=1e-6, rtol=0):
                        raise RuntimeError("initial restored physical box size differs from validation geometry")
                    if not np.allclose(shape.get_position(), center, atol=1e-6, rtol=0):
                        raise RuntimeError("initial restored physical box center differs from validation geometry")
                reference = full_audit(task, obstacles)
                depth_audit = observed_box_depth_audit(initial, obstacles, centers, halfsizes)
                for diagnostic in range(args.diagnostic_restores):
                    restore_full_snapshot(task, snapshot)
                    diagnostic_observation = task.get_observation()
                    diagnostic_audit = full_audit(task, obstacles)
                    diagnostic_prefix = "restore_diagnostic_%02d" % diagnostic
                    Image.fromarray(diagnostic_observation.front_rgb).save(folder / (diagnostic_prefix + ".png"))
                    np.savez_compressed(folder / (diagnostic_prefix + ".npz"),
                        rgb=diagnostic_observation.front_rgb, depth=diagnostic_observation.front_depth,
                        mask=diagnostic_observation.front_mask,
                        camera_intrinsics=diagnostic_observation.misc["front_camera_intrinsics"],
                        camera_extrinsics=diagnostic_observation.misc["front_camera_extrinsics"])
                    (folder / (diagnostic_prefix + ".json")).write_text(json_text(dict(
                        difference_from_original=compare_restore(reference, diagnostic_audit, initial.front_rgb,
                                                                 diagnostic_observation.front_rgb),
                        audit=diagnostic_audit)), encoding="utf-8")
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
                parent_setup_failures += 1
                record = dict(parent_id=parent_id, split=split, phase="parent_setup", success=False,
                              requested_layout_supervision_only=layout_metadata,
                              error=repr(exc), traceback=traceback.format_exc())
                if initial is not None:
                    Image.fromarray(initial.front_rgb).save(folder / "failed_setup_front.png")
                    np.savez_compressed(folder / "failed_setup_observation.npz", rgb=initial.front_rgb,
                        depth=initial.front_depth, mask=initial.front_mask,
                        camera_intrinsics=initial.misc["front_camera_intrinsics"],
                        camera_extrinsics=initial.misc["front_camera_extrinsics"])
                    record["failed_setup_observation"] = parent_id + "/failed_setup_observation.npz"
                append_json(args.output / "attempts.jsonl", record)
                print(json_text({key: value for key, value in record.items() if key != "traceback"}), flush=True)
                continue
            parents_collected += 1
            (folder / "observation_validation.json").write_text(json_text(dict(
                canonicalization_passes=args.canonicalization_passes,
                physical_obstacle_observed_depth=depth_audit)), encoding="utf-8")
            if layout_metadata is not None:
                layout_metadata.update(actual_target_centers=goals, actual_obstacle_centers=centers,
                                       actual_obstacle_sizes=np.asarray(sizes), target_visible_pixels=visible,
                                       target_colors=[target.get_color() for target in targets],
                                       parent_id=parent_id, split=split)
                (folder / "layout_sampling_supervision_only.json").write_text(json_text(layout_metadata, indent=2), encoding="utf-8")
            Image.fromarray(initial.front_rgb).save(folder / "front.png")
            np.savez_compressed(folder / "observation.npz", depth=initial.front_depth,
                                gripper_pose=initial.gripper_pose, gripper_open=initial.gripper_open,
                                camera_intrinsics=initial.misc["front_camera_intrinsics"], camera_extrinsics=initial.misc["front_camera_extrinsics"])
            np.savez_compressed(folder / "verification_only.npz", mask=initial.front_mask, obstacle_centers=centers,
                                obstacle_halfsizes=halfsizes, target_centers=goals, target_visible_pixels=visible)
            (folder / "restore_reference.json").write_text(json_text(reference), encoding="utf-8")
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
                    poses, opens, arm_joint_trace, gripper_joint_trace = [], [], [], []
                    try:
                        np.random.set_state(random_state)
                        task.reset()
                        canonical_restore(task, snapshot, args.canonicalization_passes)
                        restored = task.get_observation()
                        restored_audit = full_audit(task, obstacles)
                        difference = compare_restore(reference, restored_audit, initial.front_rgb, restored.front_rgb)
                        record["restore"] = difference
                        if not difference["global_inventory_equal"] or difference["max_abs"] > args.restore_atol or difference["rgb_max_difference"] != 0:
                            diagnostic_prefix = "failed_restore_target%d_attempt%d" % (target_index, attempt)
                            Image.fromarray(restored.front_rgb).save(folder / (diagnostic_prefix + ".png"))
                            np.savez_compressed(folder / (diagnostic_prefix + ".npz"), rgb=restored.front_rgb,
                                depth=restored.front_depth, mask=restored.front_mask,
                                camera_intrinsics=restored.misc["front_camera_intrinsics"],
                                camera_extrinsics=restored.misc["front_camera_extrinsics"])
                            (folder / (diagnostic_prefix + ".json")).write_text(json_text(restored_audit), encoding="utf-8")
                            record["failed_restore_diagnostic_prefix"] = parent_id + "/" + diagnostic_prefix
                            raise RuntimeError("full initial inventory/state/RGB equality failed")
                        restores += 1
                        arm = task._robot.arm
                        tip = np.asarray(arm.get_tip().get_pose())
                        direct_open = 1.0 if task._robot.gripper.get_open_amount()[0] > .9 else 0.0
                        record["direct_state_initial_equivalence"] = dict(
                            pose_max_abs=float(np.abs(tip - np.asarray(restored.gripper_pose)).max()),
                            open_abs=abs(direct_open - float(restored.gripper_open)))
                        if any(record["direct_state_initial_equivalence"].values()):
                            raise RuntimeError("direct current-state read differs from Scene observation")
                        poses.append(tip)
                        opens.append(restored.gripper_open)
                        arm_joint_trace.append(np.asarray(arm.get_joint_positions()))
                        gripper_joint_trace.append(np.asarray(task._robot.gripper.get_joint_positions()))
                        waypoints = collection_waypoints(goal, centers, halfsizes, proposed, args.side_margin,
                                                         .07 if args.obstacles == 1 else .035)
                        record["collection_guides_supervision_only"] = [point.tolist() for point in waypoints[:-1]]
                        for waypoint in waypoints:
                            path = arm.get_path(waypoint.tolist(), quaternion=tip[3:].tolist(), ignore_collisions=False)
                            done = False
                            for step in range(1000):
                                done = path.step()
                                task._scene.step()
                                # Read the identical state fields directly;
                                # rendering RGB/depth/masks at every physics
                                # step is unnecessary for trajectory labels.
                                poses.append(np.asarray(arm.get_tip().get_pose()))
                                opens.append(1.0 if task._robot.gripper.get_open_amount()[0] > .9 else 0.0)
                                arm_joint_trace.append(np.asarray(arm.get_joint_positions()))
                                gripper_joint_trace.append(np.asarray(task._robot.gripper.get_joint_positions()))
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
                        final_observation = task.get_observation()
                        record["direct_state_final_equivalence"] = dict(
                            pose_max_abs=float(np.abs(np.asarray(poses[-1]) - np.asarray(final_observation.gripper_pose)).max()),
                            open_abs=abs(float(opens[-1]) - float(final_observation.gripper_open)))
                        if any(record["direct_state_final_equivalence"].values()):
                            raise RuntimeError("direct final-state read differs from Scene observation")
                        record["endpoint_error_m"] = float(np.linalg.norm(xyz[-1] - goal))
                        record["length_m"] = float(np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum())
                        record["tip_polyline_clear"] = bool(tip_polyline_clear(xyz, centers, halfsizes, args.tip_clearance))
                        if record["endpoint_error_m"] > args.endpoint_tolerance or not record["tip_polyline_clear"]:
                            raise RuntimeError("endpoint or continuous tip-polyline clearance failed")
                        actual = crossing_signature(xyz, centers, halfsizes, args.tip_clearance)
                        record["actual_route_type"] = actual
                        xyz_24 = resample(np.asarray(poses), count=24)
                        record["tip_polyline_24_clear"] = bool(tip_polyline_clear(xyz_24, centers, halfsizes, args.tip_clearance))
                        if not record["tip_polyline_24_clear"] or crossing_signature(xyz_24, centers, halfsizes, args.tip_clearance) != actual:
                            raise RuntimeError("24-point reference resampling changed clearance or passage classification")
                        duplicate = actual in accepted_types if actual is not None else None
                        if actual is not None:
                            classified += 1
                            duplicate_types += int(duplicate)
                            accepted_types.add(actual)
                        filename = parent_id + "/target%d_route%d.npz" % (target_index, attempt)
                        np.savez_compressed(args.output / filename, gripper_pose=np.asarray(poses), gripper_open=np.asarray(opens),
                                            arm_joint_positions=np.asarray(arm_joint_trace), gripper_joint_positions=np.asarray(gripper_joint_trace),
                                            xyz_64=resample(np.asarray(poses)), xyz_24=xyz_24)
                        routes.append(filename)
                        route_types.append(actual)
                        successes += 1
                        record.update(success=True, duplicate_passage_type=duplicate, route_file=filename,
                                      trajectory_sha256=array_hash(np.asarray(poses)))
                    except Exception as exc:
                        record.update(error=repr(exc), traceback=traceback.format_exc())
                        if poses:
                            filename = parent_id + "/failed_target%d_attempt%d.npz" % (target_index, attempt)
                            np.savez_compressed(args.output / filename, gripper_pose=np.asarray(poses), gripper_open=np.asarray(opens),
                                                arm_joint_positions=np.asarray(arm_joint_trace), gripper_joint_positions=np.asarray(gripper_joint_trace))
                            record["failed_partial_route"] = filename
                    record["seconds"] = time.perf_counter() - tic
                    append_json(args.output / "attempts.jsonl", record)
                    print(json_text({key: value for key, value in record.items() if key != "traceback"}), flush=True)
                append_json(args.output / "supervision.jsonl", dict(id=identifier, parent_id=parent_id, split=split,
                    observation=parent_id + "/observation.npz", routes=routes, route_types=route_types,
                    task="rlbench_derived_obstacle_multitarget_reach", reference_set_complete=False,
                    verification_only=parent_id + "/verification_only.npz",
                    semantic_targets=dict(centers=goals.tolist(), target_index=target_index, tolerance=args.endpoint_tolerance)))
    finally:
        env.shutdown()
    summary = dict(status="collection_finished", parents_requested=args.parents, parents_collected=parents_collected,
                   parent_setup_failures=parent_setup_failures,
                   attempts=total, successes=successes, restore_passes=restores, classified_successes=classified,
                   duplicate_passage_type_successes=duplicate_types, elapsed_seconds=time.perf_counter() - started,
                   all_solution_count=None, continuous_whole_robot_collision_certified=False)
    (args.output / "summary.json").write_text(json_text(summary, indent=2), encoding="utf-8")
    print(json_text(summary), flush=True)


if __name__ == "__main__":
    main()
