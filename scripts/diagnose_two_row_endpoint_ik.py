"""Bounded endpoint-IK diagnostic; no candidate routes or training references.

Each query starts at the same newly created in-process native snapshot. The
ignore-collision arm is diagnostic only. Existing v1/v2/v3 data is untouched.
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

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import collect_observed_two_row_pilot as pilot

legacy = pilot.legacy


def rotation(q):
    q = np.asarray(q, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError('finite nonzero xyzw quaternion required')
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def vertical_quaternion(q):
    """Local +Z points world down; preserve projected real local +X heading.

    With R[:,2]=(0,0,-1), its xyzw quaternion has z=w=0. The formula follows
    Rxx=2*x*x-1, Ryx=2*x*y and fixes a deterministic quaternion sign.
    """
    heading = rotation(q)[:, 0].copy()
    heading[2] = 0.
    size = np.linalg.norm(heading)
    if size < 1e-8:
        raise ValueError('tip local X has no defined horizontal heading')
    heading /= size
    angle = np.arctan2(heading[1], heading[0]) / 2
    result = np.array([np.cos(angle), np.sin(angle), 0., 0.])
    if result[np.argmax(np.abs(result))] < 0:
        result = -result
    assert np.allclose(rotation(result)[:, 2], [0, 0, -1], atol=1e-12, rtol=0)
    return result


def query_plan(config):
    if config['protocol'] != 'two_row_endpoint_ik_24_v1':
        raise ValueError('unknown diagnostic protocol')
    geom = config['geometry']
    pilot.validate_config(geom)
    if (config['requested_endpoint_queries'] != 24 or config['trials'] != 128 or
            config['max_configs'] != 1 or config['max_time_ms'] != 10 or
            config['distance_threshold'] != .65 or config['maximum_wall_seconds'] != 180 or
            config['attachment_axis_cosine_min'] != .99):
        raise ValueError('registered budgets/axis gate changed')
    return [dict(query_id=index, row=row, passage=pilot.PASSAGES[passage],
                 xyz=[geom['row_x'][row], geom['corridor_guide_y'][passage], geom['route_height']],
                 orientation=orientation, ignore_collisions=ignore)
            for index, (row, passage, orientation, ignore) in enumerate(itertools.product(
                range(2), range(3), ('original', 'world_vertical'), (False, True)))]


def measured_axis_evidence(tip_pose, attachment_position):
    direction = np.asarray(tip_pose[:3])-np.asarray(attachment_position)
    distance = float(np.linalg.norm(direction))
    if distance < 1e-8:
        raise ValueError('attachment and tip do not define an approach direction')
    axes = rotation(tip_pose[3:])
    return dict(tip_pose=np.asarray(tip_pose), attachment_position=np.asarray(attachment_position),
                tip_local_axes_world=axes, attachment_to_tip_unit=direction/distance,
                attachment_to_tip_length_m=distance,
                attachment_to_tip_dot_local_positive_z=float(np.dot(direction/distance, axes[:, 2])),
                original_local_z_tilt_from_world_down_degrees=float(np.degrees(np.arccos(np.clip(-axes[2, 2], -1, 1)))),
                vertical_quaternion_xyzw=vertical_quaternion(tip_pose[3:]),
                vertical_local_axes_world=rotation(vertical_quaternion(tip_pose[3:])))


def audited_world(task, posts):
    audit = legacy.full_audit(task, posts)
    audit['state']['_diagnostic_targets'] = {}
    for name, part in [('arm', task._robot.arm), ('gripper', task._robot.gripper)]:
        for field in ('joint_target_positions', 'joint_target_velocities'):
            audit['state']['_diagnostic_targets'][name+'_'+field] = getattr(part, 'get_'+field)()
    return audit


def strict_restore(task, posts, snapshot, reference, initial, passes):
    start = time.perf_counter()
    legacy.canonical_restore(task, snapshot, passes)
    observation = task.get_observation()
    current = audited_world(task, posts)
    world = legacy.compare_restore(reference, current, initial.front_rgb, observation.front_rgb)
    obs = pilot.strict_observation_difference(initial, observation)
    passed = (world.get('max_abs') == 0 and world['global_inventory_equal'] and
              world['rgb_max_difference'] == 0 and all(obs.values()))
    return dict(passed=bool(passed), world=world, observation=obs, canonicalization_steps=passes*10,
                seconds=time.perf_counter()-start), current, observation


def bounded_ik(arm, sim, query, quaternion, config, record):
    """Instrument the exact low-level random-configuration search count."""
    original = sim.simGetConfigForTipPose
    record['lowlevel_config_search_calls'] = 0
    def counted(*args, **kwargs):
        if record['lowlevel_config_search_calls'] >= config['trials']:
            raise RuntimeError('low-level search budget exceeded')
        record['lowlevel_config_search_calls'] += 1
        return original(*args, **kwargs)
    sim.simGetConfigForTipPose = counted
    started = time.perf_counter()
    try:
        return arm.solve_ik_via_sampling(query['xyz'], quaternion=quaternion.tolist(),
            ignore_collisions=query['ignore_collisions'], trials=config['trials'],
            max_configs=config['max_configs'], distance_threshold=config['distance_threshold'],
            max_time_ms=config['max_time_ms'])
    finally:
        record['ik_seconds'] = time.perf_counter()-started
        sim.simGetConfigForTipPose = original


def run(config, config_path, output):
    # Import simulator only in the authorized execution entrypoint, never tests.
    from pyrep.backend import sim
    from pyrep.const import ObjectType, PrimitiveShape
    from pyrep.objects.shape import Shape
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks.reach_target import ReachTarget
    plan = query_plan(config); geom = config['geometry']; centers, halves = pilot.geometry(geom)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter(); records = []; fatal = None; shutdown = None
    setup = dict(success=False, simulated_steps=0, requested_setup_actions=1)
    trace = []; env = None; setup_started = time.perf_counter()
    sources = [Path(__file__), Path(pilot.__file__), Path(legacy.__file__),
               Path(legacy.native_snapshot.__code__.co_filename)]
    pilot.write_json(output/'manifest.json', dict(config=config,
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        sources_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        registered_queries=plan, role='DEV_COLLECTION', route_proposals=0, new_route_references=0,
        setup_get_path_calls_maximum=2, endpoint_ik_calls_maximum=24,
        lowlevel_config_search_calls_maximum=24*128, nominal_sum_lowlevel_time_limits_ms=24*128*10,
        simulator_random_stream_paired=False, all_solution_count=None,
        ignore_collision_results_are_positive_references=False,
        canonicalization_steps_per_restore=20, endpoint_placement_physics_steps=0,
        wall_deadline_policy='Before each query; finish any already-started bounded query and mandatory restore before stopping',
        purpose='Endpoint feasibility diagnosis only; no path search after common setup'))
    try:
        obsconfig = ObservationConfig(); obsconfig.set_all(False)
        obsconfig.front_camera.rgb = obsconfig.front_camera.depth = obsconfig.front_camera.mask = True
        obsconfig.front_camera.depth_in_meters = obsconfig.front_camera.masks_as_one_channel = True
        obsconfig.front_camera.image_size = (geom['image_size'], geom['image_size'])
        obsconfig.gripper_pose = obsconfig.gripper_open = True
        obsconfig.task_low_dim_state = False
        env = Environment(MoveArmThenGripper(JointVelocity(), Discrete()), obs_config=obsconfig, headless=True)
        env.launch(); env._pyrep.stop(); task = env.get_task(ReachTarget); env._pyrep.stop()
        posts = []
        for index, (center, half) in enumerate(zip(centers, halves)):
            post = Shape.create(PrimitiveShape.CUBOID, (2*half).tolist(), static=True, respondable=True,
                                position=center.tolist(), color=[.32, .34, .38])
            post.set_name('derived_two_row_post_%d'%index); post.set_collidable(True); post.set_detectable(True)
            posts.append(post)
        env._pyrep.start(); task._robot.arm.set_control_loop_enabled(True)
        random.seed(geom['seed']); np.random.seed(geom['seed']); task.set_variation(0); task.reset()
        targets = [task._task.target, task._task.distractor0, task._task.distractor1]
        for target, xyz in zip(targets, geom['goal_xyz']): target.set_position(xyz)
        robot_handles = {obj.get_handle() for obj in task._robot.arm.get_objects_in_tree(exclude_base=False) +
                         task._robot.gripper.get_objects_in_tree(exclude_base=False)}
        gripper_shapes = [o for o in task._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE,
                          exclude_base=False) if o.is_collidable()]
        external = [o for o in task._pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE,
                    exclude_base=False) if o.is_collidable() and o.get_handle() not in robot_handles]
        if not gripper_shapes or legacy.robot_collision_check(task, gripper_shapes, external):
            raise RuntimeError('initial collision or missing gripper shapes')
        pilot.save_observation_evidence(output/'before_preparation', task.get_observation(), audited_world(task, posts))
        trace.append(pilot.state_sample(task))
        pilot.execute(task, geom['preparation_xyz'], trace[0][0][3:], gripper_shapes, external, trace, setup, geom)
        snapshot = legacy.full_snapshot(task, posts)
        legacy.canonical_restore(task, snapshot, geom['canonicalization_passes'])
        initial = task.get_observation(); reference = audited_world(task, posts)
        if legacy.robot_collision_check(task, gripper_shapes, external): raise RuntimeError('canonical initial collision')
        if np.linalg.norm(np.asarray(initial.gripper_pose[:3])-geom['entry_xyz']) > geom['preparation_endpoint_tolerance_m']:
            raise RuntimeError('setup entry mismatch')
        setup['depth_audit'] = legacy.observed_box_depth_audit(initial, posts, centers, halves)
        setup['visible_pixels'] = [int(np.sum(initial.front_mask==o.get_handle())) for o in targets]
        if min(setup['visible_pixels']) < geom['minimum_visible_pixels']: raise RuntimeError('target visibility insufficient')
        axis = measured_axis_evidence(initial.gripper_pose, reference['state']['_global_pose_Panda_attachment']['pose'][:3])
        pilot.write_json(output/'tip_axis_evidence.json', axis)
        if axis['attachment_to_tip_dot_local_positive_z'] < config['attachment_axis_cosine_min']:
            raise RuntimeError('measured attachment-to-tip does not verify local positive Z approach axis')
        arm = task._robot.arm
        setup['joint_intervals'] = arm.get_joint_intervals()
        setup['success'] = True; setup['seconds'] = time.perf_counter()-setup_started
        pilot.save_observation_evidence(output/'common_initial', initial, reference)
        initial_q = np.asarray(initial.gripper_pose[3:]); vertical_q = np.asarray(axis['vertical_quaternion_xyzw'])
        arm_source = Path(arm.solve_ik_via_sampling.__code__.co_filename)
        pilot.write_json(output/'runtime_sources.json', dict(arm_py=str(arm_source),
            arm_py_sha256=hashlib.sha256(arm_source.read_bytes()).hexdigest()))
        for query in plan:
            if time.perf_counter()-started > config['maximum_wall_seconds']:
                raise RuntimeError('whole diagnostic wall deadline reached; remaining queries unattempted')
            record = dict(**query, ik_attempted=False, status='restore_pending')
            records.append(record); query_start = time.perf_counter()
            try:
                before, actual, observation = strict_restore(task, posts, snapshot, reference, initial, geom['canonicalization_passes'])
                record['before_restore'] = before
                if not before['passed']:
                    pilot.save_observation_evidence(output/('failed_restore_%02d'%query['query_id']), observation, actual)
                    raise RuntimeError('common initial strict restoration failed')
                record['before_robot'] = actual['state']['_robot']
                quaternion = initial_q if query['orientation']=='original' else vertical_q
                record['requested_quaternion_xyzw'] = quaternion
                record['ik_attempted'] = True
                try:
                    configs = np.asarray(bounded_ik(arm, sim, query, quaternion, config, record))
                    record['returned_configs'] = configs
                    if configs.shape != (1, 7) or not np.isfinite(configs).all():
                        raise RuntimeError('IK returned invalid config shape or values')
                    # No physics step or path interpolation. A subsequent native
                    # restore removes this temporary forward-kinematics probe.
                    arm.set_joint_positions(configs[0].tolist(), disable_dynamics=False)
                    actual_joints = np.asarray(arm.get_joint_positions())
                    pose = np.asarray(arm.get_tip().get_pose())
                    record['endpoint_joint_readback'] = actual_joints
                    record['endpoint_joint_readback_max_abs'] = float(np.abs(actual_joints-configs[0]).max())
                    record['endpoint_pose_readback'] = pose
                    record['endpoint_position_error_m'] = float(np.linalg.norm(pose[:3]-query['xyz']))
                    record['endpoint_rotation_error_degrees'] = float(np.degrees(np.arccos(np.clip((np.trace(rotation(quaternion).T@rotation(pose[3:]))-1)/2, -1, 1))))
                    record['endpoint_collision'] = legacy.robot_collision_check(task, gripper_shapes, external)
                    record['endpoint_tip_2cm_clear'] = legacy.tip_polyline_clear(np.stack([pose[:3], pose[:3]]), centers, halves, geom['tip_clearance_m'])
                    record['status'] = 'configuration_returned_diagnostic_only'
                except Exception as error:
                    record.update(status='solver_or_readback_error_no_reachability_proof', error=repr(error), traceback=traceback.format_exc())
                after, actual, observation = strict_restore(task, posts, snapshot, reference, initial, geom['canonicalization_passes'])
                record['after_restore'] = after
                if not after['passed']:
                    pilot.save_observation_evidence(output/('failed_postrestore_%02d'%query['query_id']), observation, actual)
                    raise RuntimeError('post-query strict restoration failed')
            finally:
                record['wall_seconds'] = time.perf_counter()-query_start
                legacy.append_json(output/'endpoint_queries.jsonl', record)
    except Exception as error:
        fatal = repr(error)
        pilot.write_json(output/'failure.json', dict(error=fatal, traceback=traceback.format_exc()))
    finally:
        setup.setdefault('seconds', time.perf_counter()-setup_started)
        setup['trajectory'] = pilot.save_trace(output/'preparation_trace.npz', trace)
        for key in ('get_path_calls','planning_seconds','simulation_seconds'):
            setup[key]=sum(segment[key] for segment in setup.get('planning_segments',[]))
        pilot.write_json(output/'setup.json', setup)
        if env is not None:
            try: env.shutdown()
            except Exception as error: shutdown = repr(error)
        attempted = sum(r['ik_attempted'] for r in records)
        pilot.write_json(output/'summary.json', dict(status='setup_failed' if not setup['success'] else ('diagnostic_stopped' if fatal or shutdown else 'diagnostic_finished'),
            requested_endpoint_queries=24, endpoint_queries_attempted=attempted, endpoint_queries_unattempted=24-attempted,
            configuration_returns=sum(r.get('status')=='configuration_returned_diagnostic_only' for r in records),
            lowlevel_config_search_calls=sum(r.get('lowlevel_config_search_calls',0) for r in records),
            setup_get_path_calls=sum(s['get_path_calls'] for s in setup.get('planning_segments',[])),
            route_proposals=0, accepted_route_references=0, elapsed_seconds=time.perf_counter()-started,
            fatal_error=fatal, shutdown_error=shutdown, arbitrary_hidden_state_equivalence=None))
        pilot.write_json(output/'artifact_hashes.json', {p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--config',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); run(json.loads(args.config.read_text()),args.config,args.output)


if __name__=='__main__': main()
