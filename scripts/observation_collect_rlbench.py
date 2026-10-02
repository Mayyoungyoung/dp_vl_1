"""Audited RLBench pilot: restore one initial state before each route attempt.

This changes only free-space approach collection; outputs are explicitly
RLBench-derived pilot data. It is NOT a multi-route benchmark or validated route
type inventory. No simulator state or guides are exported in model-input JSONL.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image


def world_audit(task):
    """All task/robot poses + joints/velocities: restore evidence, never input."""
    from pyrep.const import ObjectType
    objects = task._task.get_base().get_objects_in_tree(exclude_base=False)
    values = {}
    for obj in objects:
        record = {'pose': obj.get_pose()}
        if obj.get_type() == ObjectType.JOINT:
            record['joint_position'] = [obj.get_joint_position()]
        if obj.get_type() == ObjectType.SHAPE:
            record['color'] = obj.get_color()
            velocity = obj.get_velocity()
            record['velocity'] = list(velocity[0]) + list(velocity[1])
        values[obj.get_name()] = {key: np.asarray(value).tolist() for key, value in record.items()}
    robot = task._robot
    values['_robot'] = {'arm_joints': robot.arm.get_joint_positions(),
                         'arm_velocity': robot.arm.get_joint_velocities(),
                         'gripper_joints': robot.gripper.get_joint_positions(),
                         'gripper_pose': robot.arm.get_tip().get_pose()}
    values['_robot'] = {key: np.asarray(value).tolist() for key, value in values['_robot'].items()}
    return values


def audit_difference(reference, current):
    if reference.keys() != current.keys():
        return {'max_abs': float('inf'), 'same_object_inventory': False}
    difference = 0.0
    worst_field = None
    for name in reference:
        for field, ref in reference[name].items():
            value = float(np.max(np.abs(np.asarray(ref) - np.asarray(current[name][field]))))
            if value > difference:
                difference, worst_field = value, name + '.' + field
    return {'max_abs': difference, 'worst_field': worst_field, 'same_object_inventory': True}


def native_snapshot(task):
    robot = task._robot
    from pyrep.const import ObjectType
    return {'task': task._task.get_state(),
            'arm_tree': robot.arm.get_configuration_tree(),
            'gripper_tree': robot.gripper.get_configuration_tree(),
            'arm_joints': robot.arm.get_joint_positions(),
            'gripper_joints': robot.gripper.get_joint_positions(),
            'shape_colors': [(obj, obj.get_color()) for obj in task._task.get_base().get_objects_in_tree()
                             if obj.get_type() == ObjectType.SHAPE]}


def restore_native_snapshot(task, snapshot):
    """Restore after task.reset resets Python condition state.

    Canonical initial state is the saved configuration with zero dynamic joint
    velocities and current-position motor targets. It is audited after this same
    operation both for reference capture and every candidate attempt.
    """
    robot = task._robot
    robot.gripper.release()
    # Reset the physics engine and simulation clock as well as configurations.
    # Configuration trees alone do not capture contact/solver warm-start state.
    task._pyrep.stop()
    task._task.restore_state(snapshot['task'])
    task._pyrep.set_configuration_tree(snapshot['arm_tree'])
    task._pyrep.set_configuration_tree(snapshot['gripper_tree'])
    robot.arm.set_joint_positions(snapshot['arm_joints'], disable_dynamics=True)
    robot.gripper.set_joint_positions(snapshot['gripper_joints'], disable_dynamics=True)
    robot.arm.set_joint_target_positions(snapshot['arm_joints'])
    robot.gripper.set_joint_target_positions(snapshot['gripper_joints'])
    robot.arm.set_joint_target_velocities([0.0] * len(robot.arm.joints))
    robot.gripper.set_joint_target_velocities([0.0] * len(robot.gripper.joints))
    for shape, color in snapshot['shape_colors']:
        shape.set_color(color)
    task._pyrep.start()
    # Simulation restart restores motor flags from its start snapshot too.
    robot.arm.set_control_loop_enabled(True)
    # Identical settling schedule refreshes measured velocities and rendering.
    for _ in range(10):
        task._scene.step()


def array_hash(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def resample(path, count=64):
    path = np.asarray(path)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(path[:, :3], axis=0), axis=1))]
    if distance[-1] < 1e-8:
        return np.repeat(path[:1, :3], count, axis=0)
    return np.stack([np.interp(np.linspace(0, distance[-1], count), distance, path[:, dim]) for dim in range(3)], -1)


def append_json(path, value):
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False) + '\n')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', default=['reach_target', 'pick_and_lift', 'push_button', 'take_lid_off_saucepan'])
    p.add_argument('--parents-per-task', type=int, default=1)
    p.add_argument('--seed', type=int, default=260100)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--restore-atol', type=float, default=1e-5)
    p.add_argument('--render-warmup', action='store_true',
                   help='discard a rendered frame between two identical native restores')
    p.add_argument('--strict-rgb', action='store_true',
                   help='reject any initial RGB difference in addition to state/language checks')
    a = p.parse_args()
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    config = ObservationConfig()
    config.set_all(False)
    config.front_camera.rgb = True
    config.front_camera.depth = True
    config.front_camera.depth_in_meters = True
    config.front_camera.image_size = (a.image_size, a.image_size)
    config.gripper_pose = True
    config.gripper_open = True
    config.joint_positions = True
    config.task_low_dim_state = False
    env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=config, headless=True)
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'attempts.jsonl').exists():
        raise ValueError('Use a new output directory; pilot may not silently duplicate parents')
    metadata = {'source': 'RLBench-derived free-approach pilot', 'rlbench_revision': '02720bba4c73fe02eb75df946b8791b806028a9d',
                'pyrep_revision': '8f420be8064b1970aae18a9cfbc978dfb15747ef',
                'coppeliasim': '4.1.0 Ubuntu20.04', 'split': 'DEV_COLLECTION',
                'restore_protocol': 'native_state_with_optional_double_restore_v2',
                'strict_rgb': a.strict_rgb, 'render_warmup': a.render_warmup,
                'requested_tasks': a.tasks, 'parents_per_task': a.parents_per_task,
                'attempts_per_parent': a.attempts, 'seed': a.seed,
                'task_low_dim_state_exported': False, 'route_type_labels': 'unverified; do not count Euclidean variation as unique route types',
                'known_limitations': ['small feasibility pilot', 'state equality must pass before every attempt',
                    'whole-trajectory continuous collision certification is not implemented',
                    'same-image different-target instruction pairs are not yet implemented']}
    (a.output / 'manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    started = time.perf_counter()
    total, successes, restore_passes, near_duplicates = 0, 0, 0, 0
    env.launch()
    try:
        for task_index, name in enumerate(a.tasks):
            module = importlib.import_module('rlbench.tasks.' + name)
            task_class = getattr(module, ''.join(word.title() for word in name.split('_')))
            # Models imported during a running simulation may be removed when
            # stop() restores the scene. Import each task before start instead.
            env._pyrep.stop()
            task = env.get_task(task_class)
            task._robot.arm.set_control_loop_enabled(True)
            for parent in range(a.parents_per_task):
                parent_id = '%s_%06d' % (name, a.seed + parent)
                folder = a.output / parent_id
                folder.mkdir()
                seed = a.seed + task_index * 10000 + parent
                np.random.seed(seed)
                task.set_variation(parent % task.variation_count())
                rng_before = np.random.get_state()
                try:
                    descriptions, initial = task.reset()
                    snapshot = native_snapshot(task)
                    restore_native_snapshot(task, snapshot)
                    if a.render_warmup:
                        task.get_observation()  # Discard the first renderer frame.
                        restore_native_snapshot(task, snapshot)
                    initial = task.get_observation()
                    reference = world_audit(task)
                except Exception as e:
                    append_json(a.output / 'attempts.jsonl', {'parent_id': parent_id, 'phase': 'initial_reset', 'success': False, 'error': str(e)})
                    continue
                rgb = initial.front_rgb
                Image.fromarray(rgb).save(folder / 'front.png')
                camera = {key: np.asarray(value).tolist() for key, value in initial.misc.items()
                          if key.startswith('front_camera_')}
                np.savez_compressed(folder / 'observation.npz', depth=initial.front_depth,
                    gripper_pose=initial.gripper_pose, gripper_open=initial.gripper_open,
                    camera_intrinsics=initial.misc['front_camera_intrinsics'],
                    camera_extrinsics=initial.misc['front_camera_extrinsics'])
                (folder / 'camera.json').write_text(json.dumps(camera, indent=2), encoding='utf-8')
                (folder / 'restore_reference.json').write_text(json.dumps(reference), encoding='utf-8')
                # All paraphrases retain the same parent/split; they are not new scenes.
                for language_id, instruction in enumerate(descriptions):
                    append_json(a.output / 'observations.jsonl', {'id': parent_id + '_lang%d' % language_id,
                        'parent_id': parent_id, 'split': 'DEV_COLLECTION', 'image': parent_id + '/front.png', 'instruction': instruction})
                accepted = []
                for attempt in range(a.attempts):
                    total += 1
                    record = {'parent_id': parent_id, 'attempt': attempt, 'success': False,
                              'route_type': None, 'guide_is_collection_only': True}
                    tic = time.perf_counter()
                    try:
                        np.random.set_state(rng_before)
                        restored_descriptions, restored = task.reset()
                        restore_native_snapshot(task, snapshot)
                        if a.render_warmup:
                            task.get_observation()
                            restore_native_snapshot(task, snapshot)
                        restored = task.get_observation()
                        difference = audit_difference(reference, world_audit(task))
                        image_delta = int(np.max(np.abs(restored.front_rgb.astype(np.int16) - rgb.astype(np.int16))))
                        record['restore'] = dict(difference, rgb_max_difference=image_delta,
                                                  language_equal=restored_descriptions == descriptions)
                        if (difference['max_abs'] > a.restore_atol or restored_descriptions != descriptions
                                or (a.strict_rgb and image_delta != 0)):
                            raise RuntimeError('Initial state restore verification failed')
                        restore_passes += 1
                        poses = [restored.gripper_pose]
                        events = [restored.gripper_open]
                        robot = task._robot
                        robot.arm.set_control_loop_enabled(True)
                        # A free-space pre-approach is a collection proposal only.
                        # Start from the verified state for EACH attempt; never get_demos(K).
                        if attempt > 0:
                            tip = np.asarray(robot.arm.get_tip().get_pose())
                            first = task._task.get_waypoints()[0]._waypoint.get_position()
                            guide = (tip[:3] + np.asarray(first)) * 0.5
                            guide += np.array([0.0, 0.10 if attempt % 2 else -0.10, 0.08])
                            record['collection_guide'] = guide.tolist()
                            path = robot.arm.get_path(guide.tolist(), quaternion=tip[3:].tolist(), ignore_collisions=False)
                            done = False
                            steps = 0
                            while not done:
                                done = path.step()
                                task._scene.step()
                                if robot.arm.check_arm_collision():
                                    raise RuntimeError('Collision during free-space prefix')
                                obs = task.get_observation()
                                poses.append(obs.gripper_pose)
                                events.append(obs.gripper_open)
                                steps += 1
                                if steps > 1000:
                                    raise RuntimeError('Prefix exceeded step budget')
                            record['free_prefix_steps'] = steps
                        demo = task._scene.get_demo()
                        poses.extend(obs.gripper_pose for obs in demo)
                        events.extend(obs.gripper_open for obs in demo)
                        success, _ = task._task.success()
                        if not success:
                            raise RuntimeError('Original task success condition did not pass')
                        route = np.asarray(poses)
                        normalized = resample(route)
                        duplicate = any(float(np.max(np.linalg.norm(normalized - old, axis=1))) < 0.01 for old in accepted)
                        record.update(success=True, steps=len(poses), near_duplicate=duplicate,
                                      trajectory_sha256=array_hash(route), full_motion_collision_checked=False)
                        np.savez_compressed(folder / ('route_%02d.npz' % attempt), gripper_pose=route, gripper_open=np.asarray(events))
                        accepted.append(normalized)
                        successes += 1
                        near_duplicates += int(duplicate)
                    except Exception as e:
                        record['error'] = type(e).__name__ + ': ' + str(e)
                        record['traceback'] = traceback.format_exc()
                    record['seconds'] = time.perf_counter()-tic
                    append_json(a.output / 'attempts.jsonl', record)
                    print(json.dumps({k:v for k,v in record.items() if k != 'traceback'}), flush=True)
    finally:
        env.shutdown()
    summary = {'attempts': total, 'successes': successes, 'restore_passes': restore_passes,
               'near_duplicate_successes': near_duplicates, 'elapsed_seconds': time.perf_counter()-started,
               'status': 'pilot_finished', 'unique_valid_route_types': None}
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
