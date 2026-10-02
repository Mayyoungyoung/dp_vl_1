"""Six frozen joint configurations in a corresponding static scene.

This is explicitly NOT reconstruction of the original query instant. No IK,
path generation, task reset, or physics stepping is allowed. Original labels
are read-only; matrices identify possible collision bodies in the new scene.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image

_SCRIPTS=str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:sys.path.insert(0,_SCRIPTS)
import diagnose_two_row_endpoint_ik as endpoint

pilot=endpoint.pilot
legacy=endpoint.legacy
QUERY_IDS=[5,7,13,15,21,23]
SOURCE_FILES=['manifest.json','endpoint_queries.jsonl','common_initial.json','common_initial.npz','common_initial.png']


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(4*1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def load_frozen(config):
    if config['protocol']!='two_row_static_collision_six_v1' or config['query_ids']!=QUERY_IDS:
        raise ValueError('six fixed recorded collision configurations required')
    if set(config['source_files_sha256'])!=set(SOURCE_FILES):raise ValueError('freeze all five input files')
    source=Path(config['source_dataset'])
    actual={name:digest(source/name) for name in SOURCE_FILES}
    if actual!=config['source_files_sha256']:raise ValueError('frozen input SHA mismatch')
    manifest=json.loads((source/'manifest.json').read_text())
    if manifest['config']['protocol']!='two_row_endpoint_ik_24_v1':raise ValueError('wrong original diagnostic')
    rows=[json.loads(line) for line in (source/'endpoint_queries.jsonl').read_text().splitlines()]
    if len(rows)!=24 or [r['query_id'] for r in rows]!=list(range(24)):raise ValueError('original 24-query ledger incomplete')
    selected=[rows[index] for index in QUERY_IDS]
    for row in selected:
        if (row['status']!='configuration_returned_diagnostic_only' or not row['ignore_collisions'] or
                row['endpoint_collision']!='arm_environment' or row['endpoint_joint_readback_max_abs']!=0):
            raise ValueError('query is not the recorded colliding configuration')
        q=np.asarray(row['returned_configs'])
        if q.shape!=(1,7) or not np.isfinite(q).all():raise ValueError('invalid frozen joint vector')
    world=json.loads((source/'common_initial.json').read_text())
    with np.load(source/'common_initial.npz',allow_pickle=False) as z:
        observation=SimpleNamespace(front_rgb=z['rgb'].copy(),front_depth=z['depth'].copy(),
            gripper_pose=z['gripper_pose'].copy(),gripper_open=z['gripper_open'].copy(),
            misc={'front_camera_intrinsics':z['camera_intrinsics'].copy(),'front_camera_extrinsics':z['camera_extrinsics'].copy()})
    if not np.array_equal(observation.front_rgb,np.asarray(Image.open(source/'common_initial.png'))):
        raise ValueError('frozen PNG and NPZ RGB disagree')
    return dict(root=source,hashes=actual,geometry=manifest['config']['geometry'],queries=selected,world=world,observation=observation)


def verify_assets(config, installed_package):
    archive=Path(config['rlbench_archive'])
    if digest(archive)!=config['rlbench_archive_sha256']:raise ValueError('pinned RLBench archive SHA mismatch')
    result={}
    with tarfile.open(archive,'r:gz') as tar:
        for relative in ('task_design.ttt','task_ttms/reach_target.ttm'):
            suffix='/rlbench/'+relative
            matches=[m for m in tar.getmembers() if m.isfile() and m.name.endswith(suffix)]
            if len(matches)!=1:raise ValueError('model archive member missing or ambiguous: '+relative)
            value=hashlib.sha256()
            with tar.extractfile(matches[0]) as handle:
                for chunk in iter(lambda:handle.read(4*1024*1024),b''):value.update(chunk)
            expected=value.hexdigest()
            installed=Path(installed_package)/relative
            actual=digest(installed)
            if expected!=actual:raise ValueError('installed model differs from pinned archive: '+relative)
            result[relative]=dict(installed_path=str(installed),member=matches[0].name,sha256=actual,archive_match=True)
    return result


def world_difference(reference,current):
    missing=[];changed=[]
    for name in sorted(set(reference['state'])|set(current['state'])):
        if name not in reference['state'] or name not in current['state']:
            missing.append(name);continue
        a,b=reference['state'][name],current['state'][name]
        for field in sorted(set(a)|set(b)):
            if field not in a or field not in b:missing.append(name+'.'+field);continue
            old,new=np.asarray(a[field],dtype=float),np.asarray(b[field],dtype=float)
            if old.shape!=new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
                missing.append(name+'.'+field+' shape/nonfinite');continue
            difference=float(np.abs(old-new).max())
            if difference:changed.append(dict(name=name,field=field,max_abs_difference=difference))
    inventory_equal=[list(x) for x in reference['inventory']]==[list(x) for x in current['inventory']]
    return dict(exact=bool(inventory_equal and not missing and not changed),inventory_equal=inventory_equal,
                missing_fields=missing,changed_fields=changed,max_abs=max((x['max_abs_difference'] for x in changed),default=0.))


def observation_difference(reference,current):
    arrays={name:(getattr(reference,name),getattr(current,name)) for name in ('front_rgb','front_depth','gripper_pose','gripper_open')}
    arrays.update({name:(reference.misc[name],current.misc[name]) for name in ('front_camera_intrinsics','front_camera_extrinsics')})
    result={}
    for name,(a,b) in arrays.items():
        a,b=np.asarray(a),np.asarray(b);finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        result[name]=dict(exact=bool(np.array_equal(a,b) and finite),
            max_abs_difference=float(np.abs(a.astype(float)-b.astype(float)).max()) if a.shape==b.shape and finite else None)
    return result


def collision_body_group(name,post_names,robot_names,task_names):
    if name in post_names:return 'registered_post'
    if name in robot_names:return 'robot_body'
    if name in task_names:return 'task_body'
    return 'other_world_body'


def collision_matrix(arm,robot_shapes,all_shapes,post_names,robot_names,task_names):
    rows=[]
    for body in all_shapes:
        collection_hit=bool(arm.check_arm_collision(body))
        pairs=[];pair_checks=0
        if collection_hit:
            for link in robot_shapes:
                if link.get_handle()!=body.get_handle():
                    pair_checks+=1
                    if link.check_collision(body):pairs.append(link.get_name())
        rows.append(dict(body=body.get_name(),body_handle=body.get_handle(),body_collidable=bool(body.is_collidable()),
            body_group=collision_body_group(body.get_name(),post_names,robot_names,task_names),
            arm_collection_collision=collection_hit,intersecting_robot_shapes=pairs,
            robot_pair_checks_attempted=pair_checks,
            link_membership_in_arm_collection_not_individually_verified=True))
    return rows


@contextmanager
def zero_motion_guard(arm_class,sim,counts):
    patches=[]
    def replace(owner,name,kind):
        original=getattr(owner,name)
        def forbidden(*args,**kwargs):
            counts[kind]+=1
            raise RuntimeError('forbidden '+kind+' call in static-only replay: '+name)
        patches.append((owner,name,original));setattr(owner,name,forbidden)
    try:
        for name in ('get_path','get_linear_path','get_nonlinear_path'):
            replace(arm_class,name,'path_attempts')
        for name in ('solve_ik','solve_ik_via_sampling','solve_ik_via_jacobian'):
            replace(arm_class,name,'ik_attempts')
        for name in ('simGetConfigForTipPose','simCheckIkGroup','generateIkPath'):
            replace(sim,name,'ik_attempts')
        replace(sim,'simStartSimulation','simulation_start_attempts')
        original_step=sim.simExtStep
        def ui_only(*args,**kwargs):
            counts['ui_updates']+=1
            return original_step(False)
        # PyRep.launch/shutdown service their UI via simExtStep. Explicitly
        # pass False, so no physics step can run even during setup/cleanup.
        patches.append((sim,'simExtStep',original_step));sim.simExtStep=ui_only
        yield
    finally:
        for owner,name,original in reversed(patches):setattr(owner,name,original)


def mesh_provenance(shapes):
    rows=[]
    for shape in shapes:
        vertices,indices,normals=shape.get_mesh_data()
        arrays={}
        for name,values in [('vertices',vertices),('indices',indices),('normals',normals)]:
            values=np.asarray(values)
            arrays[name]=dict(shape=list(values.shape),dtype=str(values.dtype),sha256=hashlib.sha256(values.tobytes()).hexdigest())
        rows.append(dict(name=shape.get_name(),handle=shape.get_handle(),collidable=bool(shape.is_collidable()),
            bounding_box=shape.get_bounding_box(),mesh_arrays=arrays))
    return rows


def hierarchy_depths(inventory,sim):
    """Validate handles, then read root sentinels without PyRep's -1 check.

    The pinned sim.simGetObjectParent wrapper raises on the valid root value
    -1. Its Object.get_parent catches RuntimeError broadly. Here every handle
    is independently checked, and nonroot parents must exist in the audited
    inventory; unrelated validation errors and malformed graphs fail closed.
    """
    types={int(handle):int(typ) for _,handle,typ in inventory}
    if len(types)!=len(inventory):raise RuntimeError('duplicate inventory handle')
    for handle,typ in types.items():
        if int(sim.simGetObjectType(handle))!=typ:raise RuntimeError('inventory handle type changed')
    parents={handle:int(sim.lib.simGetObjectParent(handle)) for handle in types}
    if any(parent!=-1 and parent not in types for parent in parents.values()):
        raise RuntimeError('parent outside audited inventory')
    depths={}
    for handle in types:
        chain=[];current=handle
        while current!=-1 and current not in depths:
            if current in chain:raise RuntimeError('cyclic scene hierarchy')
            chain.append(current);current=parents[current]
        value=-1 if current==-1 else depths[current]
        for child in reversed(chain):
            value+=1;depths[child]=value
    return depths


def configure_static_world(task,posts,source,sim):
    saved=source['world'];robot=task._robot
    inventory=legacy.full_audit(task,posts)['inventory']
    expected={(name,typ) for name,_,typ in saved['inventory']}
    if {(name,typ) for name,_,typ in inventory}!=expected:raise RuntimeError('object name/type inventory differs from recorded scene')
    robot_objects=robot.arm.get_objects_in_tree(exclude_base=False)+robot.gripper.get_objects_in_tree(exclude_base=False)
    robot_handles={o.get_handle() for o in robot_objects}
    # Restore nonrobot global transforms in parent-before-child order. Robot
    # links are reconstructed from model root plus frozen joint coordinates.
    depths=hierarchy_depths(inventory,sim)
    for name,handle,_ in sorted(inventory,key=lambda item:depths[item[1]]):
        if handle in robot_handles and handle!=robot.arm.get_handle():continue
        pose=saved['state']['_global_pose_'+name]['pose']
        current=sim.simGetObjectPosition(handle,-1)+sim.simGetObjectQuaternion(handle,-1)
        if not np.array_equal(current,pose):
            sim.simSetObjectPosition(handle,-1,pose[:3]);sim.simSetObjectQuaternion(handle,-1,pose[3:])
    from pyrep.const import ObjectType
    for obj in task._task.get_base().get_objects_in_tree(exclude_base=False):
        values=saved['state'][obj.get_name()]
        if obj.get_type()==ObjectType.SHAPE:obj.set_color(values['color'])
        if obj.get_type()==ObjectType.JOINT:obj.set_joint_position(values['joint_position'][0])
    for post in posts:
        values=saved['state']['_extra_'+post.get_name()]
        post.set_color(values['color']);post.set_collidable(bool(values['flags'][0]))
        post.set_respondable(bool(values['flags'][1]));post.set_dynamic(bool(values['flags'][2]))
        if not np.array_equal(post.get_bounding_box(),values['bounding_box']):raise RuntimeError('post bounding box differs from recorded geometry')
    robot.arm.set_joint_positions(saved['state']['_robot']['arm_joints'],disable_dynamics=False)
    robot.gripper.set_joint_positions(saved['state']['_robot']['gripper_joints'],disable_dynamics=False)
    for name,part in [('arm',robot.arm),('gripper',robot.gripper)]:
        values=saved['state']['_diagnostic_targets']
        part.set_joint_target_positions(values[name+'_joint_target_positions'])
        part.set_joint_target_velocities(values[name+'_joint_target_velocities'])


def static_restore(task,snapshot,targets):
    core=snapshot['core'];task._task.restore_state(core['task'])
    task._pyrep.set_configuration_tree(core['arm_tree']);task._pyrep.set_configuration_tree(core['gripper_tree'])
    task._robot.arm.set_joint_positions(core['arm_joints'],disable_dynamics=False)
    task._robot.gripper.set_joint_positions(core['gripper_joints'],disable_dynamics=False)
    for name,part in [('arm',task._robot.arm),('gripper',task._robot.gripper)]:
        part.set_joint_target_positions(targets[name+'_joint_target_positions'])
        part.set_joint_target_velocities(targets[name+'_joint_target_velocities'])
    for shape,color in core['shape_colors']:shape.set_color(color)
    for entry in snapshot['obstacles']:
        task._pyrep.set_configuration_tree(entry['tree']);entry['shape'].set_color(entry['color'])
    # A fixed discarded render then measured render; no physics time advances.
    task.get_observation()
    return task.get_observation()


def run(config,config_path,output):
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
    started=time.perf_counter();counts=dict(ik_attempts=0,path_attempts=0,simulation_start_attempts=0,ui_updates=0)
    env=None;rows=[];fatal=None;shutdown=None;setup=False
    pilot.write_json(output/'manifest.json',dict(config=config,config_sha256=digest(config_path),source_sha256=digest(__file__),
        role='DEV_COLLECTION',original_instant_reproduced=False,old_native_snapshot_available=False,
        original_query_labels_unchanged=True,new_ik_budget=0,new_route_budget=0,configuration_replays_budget=6,
        comparison_scope='Frozen joint configurations in a corresponding static scene; not original dynamic instant'))
    try:
        source=load_frozen(config);assets=verify_assets(config,env_module.DIR_PATH)
        arm_path=Path(Arm.solve_ik_via_sampling.__code__.co_filename)
        if digest(arm_path)!=config['original_arm_py_sha256']:raise ValueError('PyRep Arm source differs from original endpoint diagnostic')
        pilot.write_json(output/'model_assets.json',dict(assets=assets,original_query_runtime_mesh_hashes_captured=False,
            old_run_declared_same_pinned_dependency=True,panda_embedded_in_task_design_ttt=True,
            arm_py_sha256=digest(arm_path),arm_py_matches_original_runtime=True))
        with zero_motion_guard(Arm,sim,counts):
            cfg=ObservationConfig();cfg.set_all(False)
            cfg.front_camera.rgb=cfg.front_camera.depth=cfg.front_camera.mask=True
            cfg.front_camera.depth_in_meters=cfg.front_camera.masks_as_one_channel=True
            cfg.front_camera.image_size=(224,224);cfg.gripper_pose=cfg.gripper_open=True
            env=Environment(MoveArmThenGripper(JointVelocity(),Discrete()),obs_config=cfg,headless=True)
            env.launch()
            # TaskEnvironment/get_task starts simulation automatically. Load
            # the same task model directly into Scene instead, without reset,
            # episode initialization, planning, or starting the physics engine.
            task_model=ReachTarget(env._pyrep,env._robot);env._scene.load(task_model)
            task=SimpleNamespace(_task=task_model,_robot=env._robot,_scene=env._scene,_pyrep=env._pyrep,
                                 get_observation=env._scene.get_observation)
            centers,halves=pilot.geometry(source['geometry']);posts=[]
            for index,(center,half) in enumerate(zip(centers,halves)):
                post=Shape.create(PrimitiveShape.CUBOID,(2*half).tolist(),static=True,respondable=True,
                    position=center.tolist(),color=[.32,.34,.38]);post.set_name('derived_two_row_post_%d'%index)
                post.set_collidable(True);post.set_detectable(True);posts.append(post)
            configure_static_world(task,posts,source,sim)
            snapshot=legacy.full_snapshot(task,posts)
            target_values=endpoint.audited_world(task,posts)['state']['_diagnostic_targets']
            initial=static_restore(task,snapshot,target_values);reference=endpoint.audited_world(task,posts)
            pilot.save_observation_evidence(output/'new_static_common',initial,reference)
            pilot.write_json(output/'old_common_comparison.json',dict(original_instant_reproduced=False,
                old_recorded_world=world_difference(source['world'],reference),
                old_recorded_observation=observation_difference(source['observation'],initial),
                no_acceptance_tolerance_or_original_label_changed=True))
            shapes=task._pyrep.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False)
            pilot.write_json(output/'shape_mesh_provenance.json',mesh_provenance(shapes))
            arm=task._robot.arm
            robot_shapes=arm.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False)+task._robot.gripper.get_objects_in_tree(object_type=ObjectType.SHAPE,exclude_base=False)
            robot_shapes=list({o.get_handle():o for o in robot_shapes if o.is_collidable()}.values())
            robot_names={o.get_name() for o in robot_shapes};post_names={o.get_name() for o in posts}
            task_names={o.get_name() for o in task._task.get_base().get_objects_in_tree(exclude_base=False)}
            setup=True;static_time=float(sim.simGetSimulationTime())
            for original in source['queries']:
                record=dict(query_id=original['query_id'],original_query_sha256=hashlib.sha256(legacy.json_text(original).encode()).hexdigest(),
                    original_instant_reproduced=False,status='restore_pending')
                rows.append(record);clock=time.perf_counter()
                try:
                    before=static_restore(task,snapshot,target_values);current=endpoint.audited_world(task,posts)
                    record['simulation_time_before']=float(sim.simGetSimulationTime())
                    if record['simulation_time_before']!=static_time:raise RuntimeError('simulation time advanced in static replay')
                    record['before_world']=world_difference(reference,current);record['before_observation']=observation_difference(initial,before)
                    if not record['before_world']['exact'] or not all(v['exact'] for v in record['before_observation'].values()):
                        raise RuntimeError('new common static snapshot strict pre-restore failed')
                    q=np.asarray(original['returned_configs'][0]);arm.set_joint_positions(q.tolist(),disable_dynamics=False)
                    pose=np.asarray(arm.get_tip().get_pose());joint=np.asarray(arm.get_joint_positions())
                    if not np.array_equal(q,joint):raise RuntimeError('frozen joint vector was not applied exactly')
                    record.update(status='static_comparison_only',requested_joints=q,actual_joints=joint,
                        joint_readback_max_abs=float(np.abs(q-joint).max()),actual_tip_pose=pose,
                        old_tip_pose=original['endpoint_pose_readback'],
                        old_tip_position_difference_m=float(np.linalg.norm(pose[:3]-np.asarray(original['endpoint_pose_readback'][:3]))),
                        old_tip_rotation_difference_degrees=float(np.degrees(np.arccos(np.clip((np.trace(endpoint.rotation(pose[3:]).T@endpoint.rotation(original['endpoint_pose_readback'][3:]))-1)/2,-1,1)))),
                        arm_collection_collision_all=bool(arm.check_arm_collision()),
                        body_collision_matrix=collision_matrix(arm,robot_shapes,[o for o in shapes if o.is_collidable()],post_names,robot_names,task_names))
                    pilot.write_json(output/('query_%02d_world.json'%original['query_id']),endpoint.audited_world(task,posts))
                    pilot.save_observation_evidence(output/('query_%02d_static'%original['query_id']),task.get_observation())
                    after=static_restore(task,snapshot,target_values);current=endpoint.audited_world(task,posts)
                    record['simulation_time_after']=float(sim.simGetSimulationTime())
                    if record['simulation_time_after']!=static_time:raise RuntimeError('simulation time advanced in static replay')
                    record['after_world']=world_difference(reference,current);record['after_observation']=observation_difference(initial,after)
                    if not record['after_world']['exact'] or not all(v['exact'] for v in record['after_observation'].values()):
                        raise RuntimeError('new common static snapshot strict post-restore failed')
                finally:
                    record['wall_seconds']=time.perf_counter()-clock;legacy.append_json(output/'static_comparisons.jsonl',record)
    except Exception as error:
        fatal=repr(error);pilot.write_json(output/'failure.json',dict(error=fatal,traceback=traceback.format_exc()))
    finally:
        if env is not None:
            try:env.shutdown()
            except Exception as error:shutdown=repr(error)
        attempted=sum(r['status']=='static_comparison_only' for r in rows)
        pilot.write_json(output/'summary.json',dict(status='setup_failed' if not setup else ('diagnostic_stopped' if fatal or shutdown else 'diagnostic_finished'),
            requested_configurations=6,configurations_applied=attempted,configurations_unattempted=6-attempted,
            guard_counts=counts,new_routes=0,new_route_references=0,original_instant_reproduced=False,
            elapsed_seconds=time.perf_counter()-started,fatal_error=fatal,shutdown_error=shutdown))
        pilot.write_json(output/'artifact_hashes.json',{p.relative_to(output).as_posix():digest(p)
            for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(json.loads(args.config.read_text()),args.config,args.output)


if __name__=='__main__':main()
