"""One bounded, new static-start data initialization experiment.

All four existing v5 development geometries are audited. Only geometry 283102
may receive its 27 registered proposals. No task.reset/episode validation,
setup path, fallback, geometry resampling, or old dynamic-state claim.
"""
import argparse
from contextlib import contextmanager
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import collect_observed_two_row_pilot as pilot
import collect_two_row_layout4 as batch
import diagnose_two_row_endpoint_ik as endpoint
import diagnose_two_row_static_contacts as static
legacy=pilot.legacy


def validate_registration(config):
    expected=dict(protocol='two_row_canonical_init_bounded_v6',role='DEV_COLLECTION',
        parent_seeds=[283100,283101,283102,283103],route_parent_seed=283102,
        requested_layout_audits=4,requested_route_proposals=27,selected_target_indices=[0,1,2],
        maximum_explicit_get_path_calls=243,setup_path_budget=0,initialization_ik_budget=0,
        fallback_attempts=0,geometry_resampling_attempts=0,canonicalization_passes=2,
        old_dynamic_state_reproduced=False,setup_trajectory_executed=False)
    if any(config.get(k)!=v for k,v in expected.items()):raise ValueError('bounded canonical-start registration changed')
    return expected


def checked_json(path,expected):
    if batch.digest(path)!=expected:raise ValueError('frozen source SHA mismatch: '+str(path))
    return json.loads(Path(path).read_text())


def load_sources(config):
    validate_registration(config)
    source=Path(config['source_v5']);reg=checked_json(source/'registration.json',config['registration_sha256'])
    anchor=checked_json(Path(config['source_v4'])/config['anchor_file'],config['anchor_sha256'])
    robot=anchor['state']['_robot']
    q=np.asarray(robot['arm_joints'],float);g=np.asarray(robot['gripper_joints'],float)
    if q.shape!=(7,) or g.shape!=(2,) or not np.isfinite(np.r_[q,g]).all():raise ValueError('invalid frozen Panda joints')
    if not np.array_equal(q,config['canonical_arm_joints']) or not np.array_equal(g,config['canonical_gripper_joints']):
        raise ValueError('registered joint values differ from accepted v4 initial state')
    plans=reg['parent_plan']
    if [p['config']['seed'] for p in plans]!=config['parent_seeds']:raise ValueError('source parent order changed')
    for p in plans:
        c=p['config'];pilot.validate_config(c)
        if c['post_size_xyz']!=[.035,.035,.14] or c['selected_target_indices']!=[0,1,2]:raise ValueError('source v5 geometry/proposals changed')
        filename='raw/%d/%s/before_preparation.json'%(c['seed'],p['parent_id'])
        p['saved_world']=checked_json(source/filename,config['v5_before_preparation_sha256'][str(c['seed'])])
    return plans,q,g


@contextmanager
def phase_guard(arm_class,sim,phase,counts):
    """Zero planning during initialization/restore; count all exposed entries."""
    patches=[]
    def wrap(owner,name):
        if not hasattr(owner,name):return
        original=getattr(owner,name)
        def counted(*args,**kwargs):
            key=phase['name']+':'+name;counts[key]=counts.get(key,0)+1
            if phase['name']!='route':raise RuntimeError('planning forbidden during '+phase['name']+': '+name)
            return original(*args,**kwargs)
        patches.append((owner,name,original));setattr(owner,name,counted)
    try:
        for name in ('get_path','get_linear_path','get_nonlinear_path','solve_ik','solve_ik_via_sampling','solve_ik_via_jacobian'):
            wrap(arm_class,name)
        for name in ('simGetConfigForTipPose','simCheckIkGroup','generateIkPath'):wrap(sim,name)
        yield
    finally:
        for owner,name,original in reversed(patches):setattr(owner,name,original)


@contextmanager
def monitor_settling(task,gripper_shapes,external_shapes,counters):
    original=task._scene.step
    def checked_step():
        original();counters['settling_steps']=counters.get('settling_steps',0)+1
        hit=legacy.robot_collision_check(task,gripper_shapes,external_shapes)
        if hit:
            counters['settling_collision']=hit
            raise RuntimeError('collision during fixed canonical settling: '+hit)
    task._scene.step=checked_step
    try:yield
    finally:task._scene.step=original


def restore_check(task,posts,saved,gripper_shapes,external_shapes):
    steps={}
    try:
        with monitor_settling(task,gripper_shapes,external_shapes,steps):
            check,world,obs=endpoint.strict_restore(task,posts,saved['snapshot'],saved['reference'],saved['initial'],2)
    except Exception as error:
        error.restore_evidence=dict(passed=False,settling_audit=steps,error=repr(error))
        raise
    check['settling_audit']=steps
    if not check['passed']:
        error=RuntimeError('new within-parent native restore is not exactly equal');error.restore_evidence=check
        raise error
    return check,world,obs


def route_acceptance(xyz,goals,target,c):
    centers,halves=pilot.geometry(c);sampled=legacy.resample(xyz,24)
    actual=pilot.crossing_signature(xyz,c)
    fields=dict(endpoint_error_m=float(np.linalg.norm(xyz[-1]-goals[target])),
        length_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
        tip_polyline_clear=legacy.tip_polyline_clear(xyz,centers,halves,c['tip_clearance_m']),
        actual_route_type=actual,tip_polyline_24_clear=legacy.tip_polyline_clear(sampled,centers,halves,c['tip_clearance_m']),
        h24_route_type=pilot.crossing_signature(sampled,c))
    passed=(fields['endpoint_error_m']<=c['endpoint_tolerance_m'] and fields['tip_polyline_clear'] and
            fields['tip_polyline_24_clear'] and fields['h24_route_type']==actual)
    return fields,sampled,bool(passed)


def collect_routes(task,posts,targets,saved,plan,output,phase,counters,gripper_shapes,external_shapes):
    from rlbench.const import colors
    c=plan['config'];parent=plan['parent_id'];folder=output/parent;initial=saved['initial']
    goals=np.asarray([target.get_position() for target in targets]);centers,halves=pilot.geometry(c)
    Image.fromarray(initial.front_rgb).save(folder/'front.png')
    np.savez_compressed(folder/'observation.npz',depth=initial.front_depth,gripper_pose=initial.gripper_pose,
        gripper_open=initial.gripper_open,camera_intrinsics=initial.misc['front_camera_intrinsics'],
        camera_extrinsics=initial.misc['front_camera_extrinsics'])
    np.savez_compressed(folder/'verification_only.npz',mask=initial.front_mask,obstacle_centers=centers,
        obstacle_halfsizes=halves,target_centers=goals)
    palette=np.asarray([item[1] for item in colors]);per_target=[]
    for target in (0,1,2):
        identifier=parent+'_target%d'%target
        color_index=int(np.linalg.norm(palette-targets[target].get_color(),axis=1).argmin())
        instruction='Move the gripper to touch the %s sphere while avoiding the gray posts.'%colors[color_index][0]
        legacy.append_json(output/'observations.jsonl',dict(id=identifier,parent_id=parent,split=c['split'],image=parent+'/front.png',instruction=instruction))
        routes=[];types=[];seen=set()
        for attempt,proposed in enumerate(itertools.product(pilot.PASSAGES,repeat=2)):
            tic=time.perf_counter();trace=[];counters['route_attempts']+=1
            legacy.append_json(output/'slot_ledger.jsonl',dict(event='started',parent_id=parent,input_id=identifier,attempt=attempt))
            record=dict(parent_id=parent,input_id=identifier,attempt=attempt,success=False,simulated_steps=0,
                proposed_type_supervision_only=proposed,actual_route_type=None,
                camera_flags={key:bool(getattr(task._obsconfig.front_camera,key)) for key in ('rgb','depth','mask','depth_in_meters')})
            try:
                if not all(record['camera_flags'].values()):raise RuntimeError('registered RGB-D camera flags changed')
                phase['name']='restore'
                check,world,obs=restore_check(task,posts,saved,gripper_shapes,external_shapes)
                record['strict_restore']=check;counters['strict_route_restores']+=1
                trace.append(pilot.state_sample(task))
                if not np.array_equal(trace[0][0],obs.gripper_pose) or trace[0][1]!=obs.gripper_open:raise RuntimeError('direct initial state differs')
                guides=pilot.proposed_waypoints(goals[target],proposed,c);record['collection_guides_supervision_only']=guides
                if len(guides)!=9:raise RuntimeError('registered route planning budget changed')
                phase['name']='route'
                pilot.execute(task,guides,trace[0][0][3:],gripper_shapes,external_shapes,trace,record,c)
                phase['name']='audit';final=task.get_observation()
                if not np.array_equal(trace[-1][0],final.gripper_pose) or trace[-1][1]!=final.gripper_open:raise RuntimeError('direct terminal state differs')
                xyz=np.asarray([s[0][:3] for s in trace]);fields,h24,passed=route_acceptance(xyz,goals,target,c);record.update(fields)
                if not passed:raise RuntimeError('unchanged endpoint/raw/H24 clearance or type acceptance failed')
                name='target%d_route%d.npz'%(target,attempt)
                record['trajectory']=pilot.save_trace(folder/name,trace,dict(xyz_24=h24,xyz_64=legacy.resample(xyz)))
                actual=fields['actual_route_type'];record['duplicate_passage_type']=None if actual is None else actual in seen
                if actual is not None:seen.add(actual)
                record['success']=True;counters['accepted_routes']+=1
                routes.append(parent+'/'+name);types.append(actual)
            except Exception as error:
                record.update(error=repr(error),traceback=traceback.format_exc(),restore_failure=getattr(error,'restore_evidence',None))
                record['failed_partial_trajectory']=pilot.save_trace(folder/('failed_target%d_attempt%d.npz'%(target,attempt)),trace)
                if phase['name']=='restore':
                    # A failed state gate closes the remaining budget, without retries.
                    record['fatal_restore_gate']=True
            finally:
                phase['name']='audit';record['seconds']=time.perf_counter()-tic
                pilot.add_execution_totals(counters,record);legacy.append_json(output/'attempts.jsonl',record)
                legacy.append_json(output/'slot_ledger.jsonl',dict(event='completed',parent_id=parent,input_id=identifier,attempt=attempt))
                print(legacy.json_text(record),flush=True)
            if record.get('fatal_restore_gate'):raise RuntimeError('strict route restore failed; remaining slots unattempted')
        legacy.append_json(output/'supervision.jsonl',dict(id=identifier,parent_id=parent,split=c['split'],
            task='rlbench_derived_two_row_reach',observation=parent+'/observation.npz',routes=routes,route_types=types,
            verification_only=parent+'/verification_only.npz',reference_set_complete=False,
            route_config='route_configs/'+parent+'.json',
            semantic_targets=dict(centers=goals.tolist(),target_index=target,tolerance=c['endpoint_tolerance_m'])))
        per_target.append(pilot.summarize_reference_types(identifier,types,c))
    return per_target


def run_collection(config,config_path,output,source_loader):
    """Shared physical worker; each public entry validates its own registration."""
    from pyrep.backend import sim
    from pyrep.const import ObjectType,PrimitiveShape
    from pyrep.objects.shape import Shape
    from pyrep.robots.arms.arm import Arm
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import JointVelocity
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment
    from rlbench import environment as env_module
    from rlbench.observation_config import ObservationConfig
    from rlbench.tasks.reach_target import ReachTarget
    output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();env=None;fatal=None;shutdown=None;records=[];saved={};per_target=[]
    counters=dict(route_attempts=0,accepted_routes=0,strict_route_restores=0);guard_counts={};phase=dict(name='initialization')
    pilot.write_json(output/'manifest.json',dict(config=config,config_sha256=batch.digest(config_path),
        sources_sha256={Path(p).name:batch.digest(p) for p in (__file__,pilot.__file__,legacy.__file__,endpoint.__file__,static.__file__,
            batch.__file__,legacy.native_snapshot.__code__.co_filename)},
        benchmark='RLBench-derived canonical static start',role=config['role'],
        geometry_groups_unchanged=config['protocol']=='two_row_canonical_init_bounded_v6',original_dynamic_state_reproduced=False,executed_setup_trajectory=False,
        native_snapshot_scope='New same-process trees only; exported scene is not a verified dynamic roundtrip',
        input_contract=['RGB','instruction','depth','camera','current gripper pose/open'],
        labels_only=['target/post coordinates','guide sequences','route types','masks'],
        runtime_mesh_arrays_available=False,full_robot_continuous_certificate=False,
        reference_set_complete=False,all_solution_count=None))
    try:
        plans,q,g=source_loader(config);assets=static.verify_assets(config,env_module.DIR_PATH)
        (output/'route_configs').mkdir()
        for p in plans:pilot.write_json(output/'route_configs'/(p['parent_id']+'.json'),p['config'])
        arm_path=Path(Arm.solve_ik_via_sampling.__code__.co_filename)
        if batch.digest(arm_path)!=config['original_arm_py_sha256']:raise ValueError('pinned Arm source changed')
        pilot.write_json(output/'model_assets.json',dict(assets=assets,arm_py_sha256=batch.digest(arm_path)))
        with phase_guard(Arm,sim,phase,guard_counts):
            obsconfig=ObservationConfig();obsconfig.set_all(False)
            obsconfig.front_camera.rgb=obsconfig.front_camera.depth=obsconfig.front_camera.mask=True
            obsconfig.front_camera.depth_in_meters=obsconfig.front_camera.masks_as_one_channel=True
            obsconfig.front_camera.image_size=(224,224);obsconfig.gripper_pose=obsconfig.gripper_open=True
            env=Environment(MoveArmThenGripper(JointVelocity(),Discrete()),obs_config=obsconfig,headless=True);env.launch()
            model=ReachTarget(env._pyrep,env._robot);env._scene.load(model);env._scene.init_task()
            task=SimpleNamespace(_task=model,_robot=env._robot,_scene=env._scene,_pyrep=env._pyrep,_obsconfig=obsconfig,get_observation=env._scene.get_observation)
            targets=[model.target,model.distractor0,model.distractor1];posts=[]
            for index in range(4):
                post=Shape.create(PrimitiveShape.CUBOID,[.035,.035,.14],static=True,respondable=True,color=[.32,.34,.38])
                post.set_name('derived_two_row_post_%d'%index);post.set_collidable(True);post.set_detectable(True);posts.append(post)
            robot_handles={o.get_handle() for part in (task._robot.arm,task._robot.gripper) for o in part.get_objects_in_tree(exclude_base=False)}
            gripper_shapes=[o for o in task._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False) if o.is_collidable()]
            external_shapes=[o for o in task._pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False) if o.is_collidable() and o.get_handle() not in robot_handles]
            if not gripper_shapes:raise RuntimeError('missing gripper collision shapes')
            for plan in plans:
                c=plan['config'];parent=plan['parent_id'];folder=output/parent;folder.mkdir()
                clock=time.perf_counter();record=dict(parent_id=parent,geometry_group=parent,role=c['split'],
                    requested_route_slots=27 if c['seed']==config['route_parent_seed'] else 0,passed=False,initialization='new static joint start')
                records.append(record);phase['name']='initialization';random.seed(c['seed']);np.random.seed(c['seed'])
                try:
                    task._pyrep.stop();centers,halves=pilot.geometry(c)
                    for post,center in zip(posts,centers):post.set_position(center.tolist())
                    for name,target in zip(('target','distractor0','distractor1'),targets):
                        values=plan['saved_world']['state'][name];target.set_pose(values['pose']);target.set_color(values['color'])
                    record['actual_target_colors']=[target.get_color() for target in targets]
                    if len({tuple(value) for value in record['actual_target_colors']})!=3:raise RuntimeError('three target colors not distinct')
                    for part,joints in ((task._robot.arm,q),(task._robot.gripper,g)):
                        part.set_joint_positions(joints.tolist(),disable_dynamics=False)
                        part.set_joint_target_positions(joints.tolist());part.set_joint_target_velocities([0.]*len(joints))
                        if not np.array_equal(part.get_joint_positions(),joints):raise RuntimeError('canonical q readback differs')
                    record['initial_collision']=legacy.robot_collision_check(task,gripper_shapes,external_shapes)
                    if record['initial_collision']:raise RuntimeError('registered static q collides in this layout')
                    snapshot=legacy.full_snapshot(task,posts)
                    with monitor_settling(task,gripper_shapes,external_shapes,record):legacy.canonical_restore(task,snapshot,2)
                    initial=task.get_observation();reference=endpoint.audited_world(task,posts)
                    record['actual_initial_fk_pose']=initial.gripper_pose
                    record['canonical_entry_error_m']=float(np.linalg.norm(initial.gripper_pose[:3]-np.asarray(config['canonical_entry_xyz'])))
                    if record['canonical_entry_error_m']>.01:raise RuntimeError('new canonical FK departed from declared low entry')
                    actual_centers=np.asarray([post.get_position() for post in posts])
                    actual_halves=np.asarray([np.diff(np.asarray(post.get_bounding_box()).reshape(3,2),axis=1).ravel()/2 for post in posts])
                    if not np.allclose(actual_centers,centers,atol=1e-6,rtol=0) or not np.allclose(actual_halves,halves,atol=1e-6,rtol=0):raise RuntimeError('post geometry differs')
                    goals=np.asarray([target.get_position() for target in targets])
                    if not np.allclose(goals,c['goal_xyz'],atol=1e-6,rtol=0):raise RuntimeError('target geometry differs')
                    record['actual_geometry_1mm_sha256']=batch.scene_geometry_hash(actual_centers,actual_halves,goals,.001)
                    record['post_depth_audit']=legacy.observed_box_depth_audit(initial,posts,centers,halves)
                    record['target_visible_pixels']=[int(np.sum(initial.front_mask==target.get_handle())) for target in targets]
                    if min(record['target_visible_pixels'])<10:raise RuntimeError('insufficient target visibility')
                    saved[parent]=dict(snapshot=snapshot,initial=initial,reference=reference)
                    phase['name']='restore';check,_,_=restore_check(task,posts,saved[parent],gripper_shapes,external_shapes)
                    record['strict_restore']=check
                    pilot.save_observation_evidence(folder/'new_initial',initial,reference)
                    scene_path=folder/'new_initial.ttt';task._pyrep.export_scene(str(scene_path))
                    record['native_scene_export']=dict(file=scene_path.name,sha256=batch.digest(scene_path),roundtrip_verified=False)
                    record['passed']=True
                except Exception as error:
                    record.update(error=repr(error),traceback=traceback.format_exc(),restore_failure=getattr(error,'restore_evidence',None));saved.pop(parent,None)
                    try:pilot.save_observation_evidence(folder/'failed_initialization',task.get_observation(),endpoint.audited_world(task,posts))
                    except Exception as diagnostic:record['diagnostic_capture_error']=repr(diagnostic)
                finally:
                    phase['name']='audit';record['elapsed_seconds']=time.perf_counter()-clock
                    pilot.write_json(folder/'initialization_audit.json',record);legacy.append_json(output/'layout_audits.jsonl',record)
            selected=next(p for p in plans if p['config']['seed']==config['route_parent_seed']);parent=selected['parent_id']
            if parent in saved:
                # Reuse the very same in-process snapshot audited before the fourth layout.
                phase['name']='restore';check,_,_=restore_check(task,posts,saved[parent],gripper_shapes,external_shapes)
                pilot.write_json(output/'selected_snapshot_return.json',check)
                random.seed(config['route_parent_seed']);np.random.seed(config['route_parent_seed'])
                per_target=collect_routes(task,posts,targets,saved[parent],selected,output,phase,counters,gripper_shapes,external_shapes)
    except Exception as error:
        fatal=repr(error);pilot.write_json(output/'failure.json',dict(error=fatal,traceback=traceback.format_exc(),restore_failure=getattr(error,'restore_evidence',None)))
    finally:
        if env is not None:
            try:env.shutdown()
            except Exception as error:shutdown=repr(error)
        summary=dict(status='error' if fatal or shutdown else ('collection_finished' if counters['route_attempts']==27 else 'initialization_gate_closed'),
            requested_layout_audits=config['requested_layout_audits'],completed_layout_audits=len(records),passed_layout_audits=sum(r['passed'] for r in records),
            requested_route_proposals=27,unattempted_route_proposals=27-counters['route_attempts'],
            **counters,per_target=per_target,planning_api_entry_counts=guard_counts,elapsed_seconds=time.perf_counter()-started,
            old_dynamic_initial_state_reproduced=False,executed_setup_trajectory=False,fatal_error=fatal,shutdown_error=shutdown,
            actual_geometry_duplicate_groups=batch.duplicate_groups(records,'actual_geometry_1mm_sha256'),
            internal_entry_counts_are_not_candidate_counts=True)
        pilot.write_json(output/'summary.json',summary)
        pilot.write_json(output/'mechanical_receipt.json',dict(protocol=config['protocol'],role=config['role'],
            requested_route_proposals=27,started_slots=counters['route_attempts'],
            completed_slots=sum(row['event']=='completed' for row in batch.read_rows(output/'slot_ledger.jsonl')),
            unattempted_slots=27-counters['route_attempts'],runtime_error=bool(fatal or shutdown),
            elapsed_seconds=summary['elapsed_seconds'],
            layouts=[dict(parent_id=r['parent_id'],actual_geometry_1mm_sha256=r.get('actual_geometry_1mm_sha256'),
                initial_observation_saved=(output/r['parent_id']/'new_initial.json').exists()) for r in records],
            no_route_validity_or_success_metrics=True))
        pilot.write_json(output/'artifact_hashes.json',{p.relative_to(output).as_posix():batch.digest(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
    return summary


def run(config,config_path,output):
    validate_registration(config)
    return run_collection(config,config_path,output,load_sources)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);summary=run(json.loads(args.config.read_text()),args.config,args.output)
    return 1 if summary.get('fatal_error') or summary.get('shutdown_error') else 0


if __name__=='__main__':sys.exit(main())
