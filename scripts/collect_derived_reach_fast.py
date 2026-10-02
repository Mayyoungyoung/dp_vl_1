"""Natural-layout RLBench-derived reach data with audited direct state reads.

Three visible spheres provide three instructions for one strictly restored RGB
scene. Three free-motion proposals per instruction have UNKNOWN route types.
RGB is rendered only for canonical/restore/final observation audits, never for
each trajectory sample. Every simulated step checks arm and gripper collision.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
import numpy as np
from PIL import Image

import observation_collect_rlbench as core
import collect_obstacle_reach as collision_helpers
from observation_collect_rlbench import native_snapshot, restore_native_snapshot, resample, array_hash
from collect_obstacle_reach import full_audit, compare_restore, robot_collision_check, append_json, json_text


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def direct_state(task):
    """Exactly the gripper_pose/open definitions in pinned Scene.get_observation."""
    pose = np.asarray(task._robot.arm.get_tip().get_pose())
    opened = 1.0 if task._robot.gripper.get_open_amount()[0] > .9 else 0.0
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError('nonfinite or malformed direct tip pose')
    return pose, opened


def check_direct_equivalence(pose, opened, official):
    difference = dict(pose_max_abs=float(np.abs(pose-np.asarray(official.gripper_pose)).max()),
                      open_abs=abs(opened-float(official.gripper_open)))
    if any(difference.values()):
        raise RuntimeError('direct state and official Scene observation differ: '+json_text(difference))
    return difference


def collision_objects(task):
    from pyrep.const import ObjectType
    robot = task._robot
    robot_objects = robot.arm.get_objects_in_tree(exclude_base=False)+robot.gripper.get_objects_in_tree(exclude_base=False)
    robot_handles = {obj.get_handle() for obj in robot_objects}
    gripper_shapes = [obj for obj in robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE, exclude_base=False) if obj.is_collidable()]
    external_shapes = [obj for obj in task._pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE, exclude_base=False)
                       if obj.is_collidable() and obj.get_handle() not in robot_handles]
    if not gripper_shapes:
        raise RuntimeError('no collidable gripper shapes found')
    return gripper_shapes, external_shapes


def self_test():
    pose = np.array([.1, .2, .3, 0., 0., 0., 1.])
    for opening, expected in ((1., 1.), (.9, 0.), (.8, 0.)):
        tip = SimpleNamespace(get_pose=lambda:pose.copy())
        robot = SimpleNamespace(arm=SimpleNamespace(get_tip=lambda:tip),
                                gripper=SimpleNamespace(get_open_amount=lambda:[opening, opening]))
        actual_pose, actual_open = direct_state(SimpleNamespace(_robot=robot))
        assert np.array_equal(actual_pose, pose) and actual_open == expected
        assert check_direct_equivalence(actual_pose, actual_open, SimpleNamespace(gripper_pose=pose, gripper_open=expected)) == dict(pose_max_abs=0., open_abs=0.)
    try:
        check_direct_equivalence(pose+1e-9, 1., SimpleNamespace(gripper_pose=pose, gripper_open=1.))
    except RuntimeError:
        pass
    else:
        raise AssertionError('nonzero mismatch must never be accepted')
    print(json_text(dict(status='direct_state_unit_checks_passed', live_simulator_equivalence='pending real pilot')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--parents', type=int, default=256)
    parser.add_argument('--seed', type=int, default=262000)
    parser.add_argument('--dev-parents', type=int, default=64)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test(); return
    if args.output is None or args.parents < 1 or not 0 <= args.dev_parents <= args.parents:
        parser.error('new output and valid parent/split counts required')
    if args.output.exists():
        raise ValueError('output must be new; never overwrite or mix a prior collection')
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks.reach_target import ReachTarget
    from rlbench.const import colors
    config = ObservationConfig(); config.set_all(False)
    config.front_camera.rgb = config.front_camera.depth = True
    config.front_camera.depth_in_meters = True
    config.front_camera.image_size = (args.image_size, args.image_size)
    config.gripper_pose = config.gripper_open = True
    config.task_low_dim_state = False
    env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=config, headless=True)
    args.output.mkdir(parents=True)
    source_hashes = {str(Path(path).resolve()):file_hash(path) for path in (__file__, core.__file__, collision_helpers.__file__)}
    manifest = dict(benchmark='RLBench-derived natural-layout same-image three-target reaching development data v2',
        original_benchmark_result=False, rlbench_revision='02720bba4c73fe02eb75df946b8791b806028a9d',
        pyrep_revision='8f420be8064b1970aae18a9cfbc978dfb15747ef', requested_parents=args.parents,
        requested_train_parents=args.parents-args.dev_parents, requested_dev_model_parents=args.dev_parents,
        parent_seed_start=args.seed, proposed_routes_per_target=3, targets_per_parent=3,
        parent_layout='original ReachTarget natural randomized layout; no added or relocated objects',
        splits='all development; no locked evaluation claims', restore_max_abs=0., restore_rgb_max_difference=0,
        acceptance=dict(final_tip_distance_lte_m=.03, all_simulated_step_arm_collision=False, all_simulated_step_gripper_external_collision=False),
        route_type_labels=None, reference_set_complete=False, continuous_whole_robot_collision_certified=False,
        input_contract=['RGB', 'instruction', 'depth', 'camera', 'current gripper pose/open'],
        supervision_only=['target coordinates', 'future trajectories', 'joint traces', 'collection guides', 'state audits', 'acceptance'],
        source_sha256=source_hashes, direct_state_definition='pinned Scene pose=get_tip().get_pose(); open=get_open_amount()[0] > 0.9',
        original_free_collector_already_used_direct_tip=True,
        change='add per-step gripper reads, exact official initial/final equivalence, explicit gripper collision and failed trajectories')
    (args.output/'manifest.json').write_text(json_text(manifest, indent=2))
    counts = dict(parents_requested=args.parents, parents_collected=0, parent_setup_failures=0, attempts=0,
                  proposals_not_attempted=0, successes=0, restore_passes=0, near_duplicate_successes=0,
                  direct_initial_equivalence_passes=0, direct_final_equivalence_passes=0)
    started = time.perf_counter()
    def save_progress(status):
        summary = dict(counts, status=status, elapsed_seconds=time.perf_counter()-started,
                       proposed_total=args.parents*9, route_types=None, continuous_whole_robot_collision_certified=False)
        temporary = args.output/'summary.tmp'
        temporary.write_text(json_text(summary, indent=2))
        temporary.replace(args.output/'summary.json')
        return summary
    env.launch()
    try:
        env._pyrep.stop()
        task = env.get_task(ReachTarget)
        task._robot.arm.set_control_loop_enabled(True)
        palette = np.asarray([color[1] for color in colors])
        for parent in range(args.parents):
            parent_seed = args.seed+parent
            parent_id = 'derived_reach_%06d' % parent_seed
            split = 'TRAIN' if parent < args.parents-args.dev_parents else 'DEV_MODEL'
            folder = args.output/parent_id; folder.mkdir()
            parent_record = dict(parent_id=parent_id, parent_index=parent, seed=parent_seed, split=split,
                                 variation=parent % task.variation_count(), proposed_count=9, setup_success=False)
            np.random.seed(parent_seed); task.set_variation(parent_record['variation'])
            random_state = np.random.get_state()
            try:
                task.reset()
                snapshot = native_snapshot(task)
                restore_native_snapshot(task, snapshot)
                initial = task.get_observation()
                reference = full_audit(task, [])
                targets = [task._task.target, task._task.distractor0, task._task.distractor1]
                goals = np.asarray([obj.get_position() for obj in targets])
                gripper_shapes, external_shapes = collision_objects(task)
                collision = robot_collision_check(task, gripper_shapes, external_shapes)
                if collision:
                    raise RuntimeError('initial full robot collision: '+collision)
                pose, opened = direct_state(task)
                check_direct_equivalence(pose, opened, initial)
            except Exception as exc:
                counts['parent_setup_failures'] += 1
                parent_record.update(error=repr(exc), traceback=traceback.format_exc())
                append_json(args.output/'parents.jsonl', parent_record)
                for target_index in range(3):
                    for attempt in range(3):
                        counts['proposals_not_attempted'] += 1
                        append_json(args.output/'attempts.jsonl', dict(parent_id=parent_id,
                            input_id=parent_id+'_target%d' % target_index, attempt=attempt, success=False,
                            attempted=False, phase='parent_setup_failed', error=repr(exc), route_type=None))
                print(json_text(parent_record), flush=True)
                save_progress('running'); continue
            counts['parents_collected'] += 1
            parent_record.update(setup_success=True, gripper_collision_shapes=len(gripper_shapes), external_collision_shapes=len(external_shapes))
            append_json(args.output/'parents.jsonl', parent_record)
            Image.fromarray(initial.front_rgb).save(folder/'front.png')
            np.savez_compressed(folder/'observation.npz', depth=initial.front_depth, gripper_pose=initial.gripper_pose,
                gripper_open=initial.gripper_open, camera_intrinsics=initial.misc['front_camera_intrinsics'],
                camera_extrinsics=initial.misc['front_camera_extrinsics'])
            (folder/'restore_reference.json').write_text(json_text(reference))
            for target_index, obj in enumerate(targets):
                goal = goals[target_index]
                color_index = int(np.linalg.norm(palette-obj.get_color(), axis=1).argmin())
                input_id = parent_id+'_target%d' % target_index
                instruction = 'Move the gripper to touch the %s sphere.' % colors[color_index][0]
                append_json(args.output/'observations.jsonl', dict(id=input_id, parent_id=parent_id, split=split,
                            image=parent_id+'/front.png', instruction=instruction))
                routes, accepted = [], []
                for attempt in range(3):
                    counts['attempts'] += 1
                    tic = time.perf_counter()
                    record = dict(parent_id=parent_id, input_id=input_id, attempt=attempt, attempted=True,
                                  success=False, route_type=None, simulated_steps=0, original_benchmark_result=False)
                    poses, opens, joints, gripper_joints = [], [], [], []
                    try:
                        np.random.set_state(random_state); task.reset()
                        restore_native_snapshot(task, snapshot)
                        restored = task.get_observation()
                        restored_audit = full_audit(task, [])
                        difference = compare_restore(reference, restored_audit, initial.front_rgb, restored.front_rgb)
                        record['restore'] = difference
                        if not difference['global_inventory_equal'] or difference['max_abs'] != 0 or difference['rgb_max_difference'] != 0:
                            diagnostic = folder/('failed_restore_target%d_attempt%d' % (target_index, attempt))
                            Image.fromarray(restored.front_rgb).save(diagnostic.with_suffix('.png'))
                            diagnostic.with_suffix('.json').write_text(json_text(restored_audit))
                            record['restore_failure_evidence'] = str(diagnostic.relative_to(args.output))
                            raise RuntimeError('strict full state/inventory/RGB restoration failed')
                        counts['restore_passes'] += 1
                        arm, gripper = task._robot.arm, task._robot.gripper
                        tip, opened = direct_state(task)
                        record['direct_state_initial_equivalence'] = check_direct_equivalence(tip, opened, restored)
                        counts['direct_initial_equivalence_passes'] += 1
                        poses.append(tip); opens.append(opened)
                        joints.append(np.asarray(arm.get_joint_positions())); gripper_joints.append(np.asarray(gripper.get_joint_positions()))
                        waypoints = []
                        if attempt > 0:
                            guide = (tip[:3]+goal)*.5+np.array([0., .10 if attempt == 1 else -.10, .08])
                            waypoints.append(guide)
                            record['collection_guide_supervision_only'] = guide.tolist()
                        waypoints.append(goal)
                        for waypoint in waypoints:
                            path = arm.get_path(waypoint.tolist(), quaternion=tip[3:].tolist(), ignore_collisions=False)
                            done = False
                            for _ in range(1000):
                                done = path.step(); task._scene.step()
                                pose, opened = direct_state(task)
                                poses.append(pose); opens.append(opened)
                                joints.append(np.asarray(arm.get_joint_positions())); gripper_joints.append(np.asarray(gripper.get_joint_positions()))
                                record['simulated_steps'] += 1
                                collision = robot_collision_check(task, gripper_shapes, external_shapes)
                                if collision:
                                    record['collision_pair'] = collision
                                    raise RuntimeError('robot collision during simulated motion')
                                if done:
                                    break
                            if not done:
                                raise RuntimeError('segment exceeded 1000 physical steps')
                        final = task.get_observation()
                        record['direct_state_final_equivalence'] = check_direct_equivalence(poses[-1], opens[-1], final)
                        counts['direct_final_equivalence_passes'] += 1
                        record['endpoint_error_m'] = float(np.linalg.norm(poses[-1][:3]-goal))
                        if record['endpoint_error_m'] > .03:
                            raise RuntimeError('3cm endpoint acceptance failed')
                        normalized = resample(np.asarray(poses))
                        duplicate = any(np.linalg.norm(normalized-old, axis=1).max() < .01 for old in accepted)
                        accepted.append(normalized)
                        filename = parent_id+'/target%d_route%d.npz' % (target_index, attempt)
                        np.savez_compressed(args.output/filename, gripper_pose=np.asarray(poses), gripper_open=np.asarray(opens),
                            arm_joint_positions=np.asarray(joints), gripper_joint_positions=np.asarray(gripper_joints), xyz_64=normalized)
                        routes.append(filename)
                        counts['successes'] += 1; counts['near_duplicate_successes'] += int(duplicate)
                        record.update(success=True, near_duplicate=bool(duplicate), route_file=filename,
                                      steps=len(poses), trajectory_sha256=array_hash(np.asarray(poses)))
                    except Exception as exc:
                        record.update(error=repr(exc), traceback=traceback.format_exc())
                        if poses:
                            # Audit the last retained sample even for planning/
                            # collision failures whenever a live state exists.
                            try:
                                final = task.get_observation()
                                record['direct_state_failed_final_equivalence'] = check_direct_equivalence(poses[-1], opens[-1], final)
                            except Exception as audit_exc:
                                record['failed_final_audit_error'] = repr(audit_exc)
                            filename = parent_id+'/failed_target%d_attempt%d.npz' % (target_index, attempt)
                            np.savez_compressed(args.output/filename, gripper_pose=np.asarray(poses), gripper_open=np.asarray(opens),
                                arm_joint_positions=np.asarray(joints), gripper_joint_positions=np.asarray(gripper_joints))
                            record['failed_partial_route'] = filename
                    record['seconds'] = time.perf_counter()-tic
                    append_json(args.output/'attempts.jsonl', record)
                    print(json_text({key:value for key,value in record.items() if key != 'traceback'}), flush=True)
                append_json(args.output/'supervision.jsonl', dict(id=input_id, parent_id=parent_id, split=split,
                    observation=parent_id+'/observation.npz', routes=routes, route_types=None,
                    task='rlbench_derived_multitarget_reach_fast_v2', reference_set_complete=False,
                    semantic_targets=dict(centers=goals.tolist(), target_index=target_index, tolerance=.03)))
            save_progress('running')
    except BaseException:
        save_progress('interrupted_or_failed')
        raise
    finally:
        env.shutdown()
    summary = save_progress('collection_finished')
    print(json_text(summary), flush=True)


if __name__ == '__main__':
    main()
