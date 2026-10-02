"""Same-image, three-language-target RLBench-derived reach collection.

ReachTarget's visible colored spheres provide three independent language goals.
The simulator scene, image and robot initial state stay fixed. Original RLBench
success is not used: acceptance is explicit Cartesian endpoint tolerance after
collision-checked planned free motion. This does not label route topology.
"""
import argparse
import json
from pathlib import Path
import time
import traceback
import numpy as np
from PIL import Image
from observation_collect_rlbench import (world_audit, audit_difference, array_hash,
    resample, append_json, native_snapshot, restore_native_snapshot)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--parents', type=int, default=4)
    p.add_argument('--seed', type=int, default=261000)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--dev-parents', type=int, default=1)
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--restore-atol', type=float, default=1e-5)
    p.add_argument('--endpoint-tolerance', type=float, default=0.03)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError('Output must be new to prevent duplicate or mixed pilot data')
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks.reach_target import ReachTarget
    from rlbench.const import colors
    config = ObservationConfig()
    config.set_all(False)
    config.front_camera.rgb = True
    config.front_camera.depth = True
    config.front_camera.depth_in_meters = True
    config.front_camera.image_size = (args.image_size, args.image_size)
    config.gripper_pose = True
    config.gripper_open = True
    config.task_low_dim_state = False
    env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=config, headless=True)
    args.output.mkdir(parents=True)
    manifest = {'benchmark': 'RLBench-derived same-image multi-target reaching pilot v1',
        'rlbench_revision': '02720bba4c73fe02eb75df946b8791b806028a9d',
        'pyrep_revision': '8f420be8064b1970aae18a9cfbc978dfb15747ef',
        'original_benchmark_result': False, 'task_low_dim_state': False,
        'acceptance': {'final_tip_distance_lte_m': args.endpoint_tolerance, 'simulated_step_arm_collision': False},
        'route_type_labels': None, 'continuous_segment_collision_certified': False,
        'same_image_different_target': True, 'parent_count_requested': args.parents,
        'train_parent_count_requested': args.parents-args.dev_parents, 'dev_model_parent_count_requested': args.dev_parents,
        'seed': args.seed, 'attempts_per_target': args.attempts,
        'input_contract': ['RGB', 'instruction', 'depth', 'camera', 'current gripper pose/open'],
        'supervision_only': ['target_coordinates', 'future_path', 'collection_guide', 'acceptance']}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    total = successes = restores = duplicates = parents_collected = 0
    start = time.perf_counter()
    env.launch()
    try:
        task = env.get_task(ReachTarget)
        task._robot.arm.set_control_loop_enabled(True)
        for parent in range(args.parents):
            parent_id = 'derived_reach_%06d' % (args.seed + parent)
            split = 'TRAIN' if parent < args.parents-args.dev_parents else 'DEV_MODEL'
            folder = args.output / parent_id
            folder.mkdir()
            np.random.seed(args.seed + parent)
            task.set_variation(parent % task.variation_count())
            rng = np.random.get_state()
            try:
                _, initial = task.reset()
                snapshot = native_snapshot(task)
                restore_native_snapshot(task, snapshot)
                initial = task.get_observation()
                reference = world_audit(task)
            except Exception as e:
                append_json(args.output / 'attempts.jsonl', {'parent_id': parent_id, 'phase': 'initial_reset', 'success': False, 'error': str(e)})
                continue
            parents_collected += 1
            Image.fromarray(initial.front_rgb).save(folder / 'front.png')
            camera = {key: np.asarray(value).tolist() for key, value in initial.misc.items() if key.startswith('front_camera_')}
            np.savez_compressed(folder / 'observation.npz', depth=initial.front_depth,
                gripper_pose=initial.gripper_pose, gripper_open=initial.gripper_open,
                camera_intrinsics=np.asarray(initial.misc['front_camera_intrinsics']),
                camera_extrinsics=np.asarray(initial.misc['front_camera_extrinsics']))
            (folder / 'camera.json').write_text(json.dumps(camera, indent=2), encoding='utf-8')
            (folder / 'restore_reference.json').write_text(json.dumps(reference), encoding='utf-8')
            objects = [task._task.target, task._task.distractor0, task._task.distractor1]
            semantic_centers = [np.asarray(obj.get_position()).tolist() for obj in objects]
            rgb_palette = np.asarray([color[1] for color in colors])
            for target_index, obj in enumerate(objects):
                goal = np.asarray(obj.get_position())
                color_index = int(np.argmin(np.linalg.norm(rgb_palette-np.asarray(obj.get_color()), axis=1)))
                color_name = colors[color_index][0]
                input_id = parent_id + '_target%d' % target_index
                instruction = 'Move the gripper to touch the %s sphere.' % color_name
                append_json(args.output / 'observations.jsonl', {'id': input_id, 'parent_id': parent_id,
                    'split': split, 'image': parent_id + '/front.png', 'instruction': instruction})
                accepted, route_files = [], []
                for attempt in range(args.attempts):
                    total += 1
                    tic = time.perf_counter()
                    record = {'parent_id': parent_id, 'input_id': input_id, 'attempt': attempt,
                              'success': False, 'original_rlbench_success_used': False,
                              'route_type': None, 'target_coordinates_supervision_only': goal.tolist()}
                    try:
                        np.random.set_state(rng)
                        _, restored = task.reset()
                        restore_native_snapshot(task, snapshot)
                        restored = task.get_observation()
                        difference = audit_difference(reference, world_audit(task))
                        delta = int(np.max(np.abs(initial.front_rgb.astype(np.int16)-restored.front_rgb.astype(np.int16))))
                        record['restore'] = dict(difference, rgb_max_difference=delta)
                        if difference['max_abs'] > args.restore_atol or delta != 0:
                            raise RuntimeError('Same initial state/image verification failed')
                        restores += 1
                        arm = task._robot.arm
                        tip = np.asarray(arm.get_tip().get_pose())
                        poses = [tip]
                        segments = []
                        if attempt > 0:
                            guide = (tip[:3] + goal) * 0.5 + np.array([0, 0.10 if attempt % 2 else -0.10, 0.08])
                            segments.append(guide)
                            record['collection_guide_supervision_only'] = guide.tolist()
                        segments.append(goal)
                        for waypoint in segments:
                            path = arm.get_path(waypoint.tolist(), quaternion=tip[3:].tolist(), ignore_collisions=False)
                            done = False
                            for step in range(1000):
                                done = path.step()
                                task._scene.step()
                                if arm.check_arm_collision():
                                    raise RuntimeError('Arm collision during derived reach')
                                poses.append(np.asarray(arm.get_tip().get_pose()))
                                if done:
                                    break
                            if not done:
                                raise RuntimeError('Path step budget exhausted')
                        distance = float(np.linalg.norm(poses[-1][:3]-goal))
                        record['endpoint_error_m'] = distance
                        if distance > args.endpoint_tolerance:
                            raise RuntimeError('Derived endpoint acceptance failed')
                        normalized = resample(np.asarray(poses))
                        duplicate = any(float(np.max(np.linalg.norm(normalized-old, axis=1))) < .01 for old in accepted)
                        accepted.append(normalized)
                        route_file = parent_id + '/target%d_route%d.npz' % (target_index, attempt)
                        np.savez_compressed(args.output / route_file, gripper_pose=np.asarray(poses),
                                             gripper_open=np.full(len(poses), restored.gripper_open), xyz_64=normalized)
                        route_files.append(route_file)
                        record.update(success=True, near_duplicate=duplicate, trajectory_sha256=array_hash(np.asarray(poses)),
                                      steps=len(poses), route_file=route_file)
                        successes += 1
                        duplicates += int(duplicate)
                    except Exception as e:
                        record['error'] = type(e).__name__ + ': ' + str(e)
                        record['traceback'] = traceback.format_exc()
                    record['seconds'] = time.perf_counter()-tic
                    append_json(args.output / 'attempts.jsonl', record)
                    print(json.dumps({k:v for k,v in record.items() if k != 'traceback'}), flush=True)
                append_json(args.output / 'supervision.jsonl', {'id': input_id, 'parent_id': parent_id,
                    'split': split, 'routes': route_files, 'task': 'rlbench_derived_multitarget_reach',
                    'semantic_targets': {'centers': semantic_centers, 'target_index': target_index,
                                         'tolerance': args.endpoint_tolerance},
                    'observation': parent_id + '/observation.npz', 'reference_set_complete': False})
    finally:
        env.shutdown()
    summary = {'requested_parents': args.parents, 'parents_collected': parents_collected,
        'attempts': total, 'successes': successes, 'restore_passes': restores,
        'near_duplicate_successes': duplicates, 'elapsed_seconds': time.perf_counter()-start,
        'unique_valid_route_types': None, 'status': 'derived_pilot_finished'}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
