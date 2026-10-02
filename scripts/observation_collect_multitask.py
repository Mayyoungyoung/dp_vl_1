"""Registered six-task collection with independently resumable parent workers.

This is RLBench-derived free-approach data. Route-type counts stay unknown.
Locked content is physically separate and never printed by the coordinator.
Actual cross-process simulator reconstruction must pass the persisted reference;
the filesystem self-test cannot establish that simulator property.
"""
import argparse
import datetime
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np


TASKS = ('reach_target','pick_and_lift','push_button','take_lid_off_saucepan','pick_up_cup','slide_block_to_target')
ROLES = ('TRAIN',)*16+('DEV_MODEL',)*2+('DEV_SCORE',)*2+('CALIBRATION',)*2+('TEST_LOCKED',)*2
VERSION = 'registered_six_task_parent_resume_v1'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquire_lock(path, allow_stale=False):
    if path.exists() and allow_stale:
        previous=json.loads(path.read_text())
        if previous['pid']==os.getpid():
            raise RuntimeError('Existing lock owner still exists: '+str(previous['pid']))
        if os.name!='posix':
            raise RuntimeError('Stale foreign PID validation is supported only on the Linux collector host')
        try:
            os.kill(previous['pid'],0)
        except ProcessLookupError:
            # Preserve our stale marker. A live/reused PID remains a block;
            # this code never terminates a process to claim its lock.
            path.rename(path.with_name(path.name+'.stale-'+str(time.time_ns())))
        else:
            raise RuntimeError('Existing lock owner still exists: '+str(previous['pid']))
    descriptor=os.open(str(path),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:
        json.dump(dict(pid=os.getpid(),started_utc=now()),stream);stream.flush();os.fsync(stream.fileno())


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def registration(seed):
    parents=[]
    for task_index,task in enumerate(TASKS):
        for parent_index,role in enumerate(ROLES):
            parent_seed=seed+task_index*10000+parent_index
            parents.append(dict(parent_id=task+'_'+str(parent_seed),task=task,parent_index=parent_index,
                seed=parent_seed,split=role,requested_attempts=3))
    return dict(version=VERSION,seed=seed,tasks=list(TASKS),parents=parents,parents_requested=len(parents),
        attempts_requested=3*len(parents),strict_state_atol=0.,strict_rgb=True,render_warmup=True,
        image_size=224,proposal_scope='original demo or signed-y free-prefix, original task success flag',
        variation_rule='parent_index modulo task variation_count; role variation distributions may differ, no IID claim',
        route_type_definition=None,full_motion_collision_certification=False)


def ensure_registration(root, seed):
    root.mkdir(parents=True,exist_ok=True);path=root/'partition_manifest.json';expected=registration(seed)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Registration mismatch; existing parents must not be reassigned')
    else:
        atomic_json(path,expected)
    return expected


def ensure_source(root):
    source=dict(collector_sha256=digest(__file__),restore_helper_sha256=digest(Path(__file__).with_name('observation_collect_rlbench.py')),
        expected_rlbench_revision='02720bba4c73fe02eb75df946b8791b806028a9d',
        expected_pyrep_revision='8f420be8064b1970aae18a9cfbc978dfb15747ef')
    path=root/'source_manifest.json'
    if path.exists():
        if json.loads(path.read_text())!=source:raise ValueError('Pinned collector source mismatch; resume requires original source')
    else:atomic_json(path,source)


def parent_folder(root, spec):
    return root/spec['split']/'parents'/spec['parent_id']


def completed_records(folder):
    records=[]
    for filename in sorted((folder/'attempts').glob('slot_*/record.json')):
        row=json.loads(filename.read_text());assert row['status']=='completed'
        if row['success']:
            path=folder/row['route'];assert path.is_file() and digest(path)==row['route_file_sha256']
        records.append(row)
    if len({row['attempt'] for row in records}) != len(records):
        raise ValueError('Duplicate completed attempts')
    return records


def pending_slots(folder, count=3):
    done={row['attempt'] for row in completed_records(folder)}
    if any(slot<0 or slot>=count for slot in done):
        raise ValueError('Unexpected attempt slot')
    return [slot for slot in range(count) if slot not in done]


def strict_input_difference(reference, current):
    if set(reference)!=set(current):
        raise ValueError('Observed input field inventory differs')
    result={}
    for name in reference:
        old,new=np.asarray(reference[name]),np.asarray(current[name])
        if old.shape!=new.shape or not np.isfinite(old).all() or not np.isfinite(new).all():
            raise ValueError('Invalid or mismatched input '+name)
        result[name]=float(np.max(np.abs(old.astype(np.float64)-new.astype(np.float64))))
    return result


def atomic_jsonl(path, values):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('w',encoding='utf-8') as stream:
        for value in values:stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def export_role(root, role, plan):
    """No mixed-role manifest. Closed zero-reference parents remain represented."""
    observations=[];supervision=[];closed=[]
    role_root=root/role
    for spec in plan['parents']:
        if spec['split']!=role:continue
        folder=parent_folder(root,spec);closure=folder/'closed.json'
        if not closure.exists():continue
        result=json.loads(closure.read_text());closed.append(dict(parent_id=spec['parent_id'],**result))
        pointer=folder/'reference_pointer.json'
        if not pointer.exists():continue
        reference_folder=folder/json.loads(pointer.read_text())['directory']
        reference=json.loads((reference_folder/'reference.json').read_text())
        records=completed_records(folder)
        routes=[str((folder/row['route']).relative_to(role_root)).replace('\\','/') for row in records if row['success']]
        for index,instruction in enumerate(reference['descriptions']):
            identifier=spec['parent_id']+'_lang'+str(index)
            observations.append(dict(id=identifier,parent_id=spec['parent_id'],split=role,
                image=(reference_folder/'front.png').relative_to(role_root).as_posix(),instruction=instruction))
            supervision.append(dict(id=identifier,parent_id=spec['parent_id'],split=role,task=spec['task'],
                observation=(reference_folder/'observation.npz').relative_to(role_root).as_posix(),routes=routes,
                route_types=[None]*len(routes),semantic_targets=None,original_simulator_task_success_required=True,
                successful_attempts=[row['attempt'] for row in records if row['success']],
                requested_attempts=spec['requested_attempts'],full_motion_collision_checked=False,
                empty_reference_policy='Unknown solution availability; no negative existence label'))
    atomic_jsonl(role_root/'observations.jsonl',observations);atomic_jsonl(role_root/'supervision.jsonl',supervision)
    atomic_json(role_root/'closure_index.json',dict(split=role,requested_parents=sum(s['split']==role for s in plan['parents']),closed=closed))
    return len(closed)


def mechanical_summary(root, plan):
    """Only scheduling/cost/attempt-budget accounting; no route success labels."""
    parents=[];current=datetime.datetime.now(datetime.timezone.utc)
    for spec in plan['parents']:
        folder=parent_folder(root,spec);finished=[];unfinished=[]
        for filename in sorted((folder/'sessions').glob('*.json')):
            session=json.loads(filename.read_text())
            if session.get('elapsed_seconds') is not None:
                finished.append(dict(session_id=filename.stem,elapsed_seconds=session['elapsed_seconds'],status=session['status'],worker_exit_code=session.get('worker_exit_code')))
            else:
                unfinished.append(dict(session_id=filename.stem,elapsed_seconds_upper_bound=(current-datetime.datetime.fromisoformat(session['started_utc'])).total_seconds(),actual_compute_seconds=None))
        completed=len(list((folder/'attempts').glob('slot_*/record.json')))
        parents.append(dict(parent_id=spec['parent_id'],split=spec['split'],closed=(folder/'closed.json').exists(),
            requested_attempts=spec['requested_attempts'],completed_attempt_records=completed,
            unattempted_or_uncommitted_slots=spec['requested_attempts']-completed,
            interrupted_proposal_executions=len(list((folder/'attempts').glob('slot_*/interruptions/*.json'))),
            finalized_worker_elapsed_seconds=sum(row['elapsed_seconds'] for row in finished),
            finalized_sessions=finished,unfinished_worker_elapsed_upper_bounds=unfinished))
    result=dict(timestamp_utc=now(),scope='mechanical scheduling and wall-time only; no success labels or locked sample contents',
        requested_parents=len(parents),closed_parent_markers=sum(row['closed'] for row in parents),
        finalized_worker_elapsed_seconds=sum(row['finalized_worker_elapsed_seconds'] for row in parents),parents=parents)
    atomic_json(root/'mechanical_status.json',result);return result


def reference_data(observation):
    return dict(rgb=observation.front_rgb,depth=observation.front_depth,gripper_pose=observation.gripper_pose,
        gripper_open=np.asarray(observation.gripper_open),camera_intrinsics=observation.misc['front_camera_intrinsics'],
        camera_extrinsics=observation.misc['front_camera_extrinsics'])


def restore(task,snapshot,core):
    core.restore_native_snapshot(task,snapshot);task.get_observation();core.restore_native_snapshot(task,snapshot)
    return task.get_observation()


def worker(root, plan, spec, max_new_attempts=None):
    started=time.perf_counter();session=now()
    folder=parent_folder(root,spec);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'closed.json').exists():return 0
    lock=folder/'worker.lock';acquire_lock(lock,allow_stale=True)
    env=None;session_path=folder/'sessions'/(str(time.time_ns())+'_'+str(os.getpid())+'.json')
    atomic_json(session_path,dict(status='running',pid=os.getpid(),started_utc=session,elapsed_seconds=None))
    atomic_json(folder/'worker_status.json',dict(status='initializing',pid=os.getpid(),started_utc=session,spec=spec,
        script_sha256=digest(__file__),restore_helper_sha256=digest(Path(__file__).with_name('observation_collect_rlbench.py'))))
    try:
        from PIL import Image
        from scripts import observation_collect_rlbench as core
        from rlbench.action_modes.action_mode import MoveArmThenGripper
        from rlbench.action_modes.arm_action_modes import JointVelocity
        from rlbench.action_modes.gripper_action_modes import Discrete
        from rlbench.environment import Environment
        from rlbench.observation_config import ObservationConfig
        config=ObservationConfig();config.set_all(False);config.front_camera.rgb=True;config.front_camera.depth=True
        config.front_camera.depth_in_meters=True;config.front_camera.image_size=(plan['image_size'],)*2
        config.gripper_pose=True;config.gripper_open=True;config.joint_positions=True;config.task_low_dim_state=False
        env=Environment(MoveArmThenGripper(JointVelocity(),Discrete()),obs_config=config,headless=True);env.launch();env._pyrep.stop()
        module=importlib.import_module('rlbench.tasks.'+spec['task']);cls=getattr(module,''.join(x.title() for x in spec['task'].split('_')))
        task=env.get_task(cls);task._robot.arm.set_control_loop_enabled(True)
        random.seed(spec['seed']);np.random.seed(spec['seed']);task.set_variation(spec['parent_index']%task.variation_count())
        rng_before=np.random.get_state();python_rng_before=random.getstate()
        descriptions,_=task.reset();snapshot=core.native_snapshot(task);initial=restore(task,snapshot,core)
        actual_world=core.world_audit(task);inputs=reference_data(initial)
        pointer=folder/'reference_pointer.json'
        if pointer.exists():
            pointer_data=json.loads(pointer.read_text());ref_folder=folder/pointer_data['directory']
            for name,key in [('reference.json','reference_sha256'),('observation.npz','observation_sha256'),('front.png','image_sha256')]:
                if digest(ref_folder/name)!=pointer_data[key]:raise ValueError('Persisted reference hash changed: '+name)
            reference=json.loads((ref_folder/'reference.json').read_text())
            with np.load(ref_folder/'observation.npz',allow_pickle=False) as archive:old_inputs={key:archive[key].copy() for key in archive.files}
            old_inputs['rgb']=np.asarray(Image.open(ref_folder/'front.png'))
            delta=strict_input_difference(old_inputs,inputs);world_delta=core.audit_difference(reference['world'],actual_world)
            check=dict(observation_max_differences=delta,world_difference=world_delta,language_equal=reference['descriptions']==descriptions)
            atomic_json(folder/'latest_reconstruction_check.json',check)
            if any(delta.values()) or world_delta['max_abs']!=0 or not check['language_equal']:
                raise RuntimeError('resume_blocked: reconstructed initial state differs from persisted parent reference')
        else:
            versions=folder/'reference_versions';versions.mkdir(exist_ok=True)
            version=0
            while (versions/('%04d'%version)).exists():version+=1
            ref_folder=versions/('%04d'%version);ref_folder.mkdir()
            Image.fromarray(inputs['rgb']).save(ref_folder/'front.png')
            np.savez_compressed(ref_folder/'observation.npz',**{key:value for key,value in inputs.items() if key!='rgb'})
            reference=dict(descriptions=descriptions,world=actual_world,seed=spec['seed'],variation=task._variation_number if hasattr(task,'_variation_number') else spec['parent_index']%task.variation_count())
            atomic_json(ref_folder/'reference.json',reference)
            atomic_json(pointer,dict(directory=ref_folder.relative_to(folder).as_posix(),reference_sha256=digest(ref_folder/'reference.json'),observation_sha256=digest(ref_folder/'observation.npz'),image_sha256=digest(ref_folder/'front.png')))
        accepted=[]
        for old in completed_records(folder):
            if old['success']:
                with np.load(folder/old['route'],allow_pickle=False) as archive:accepted.append(core.resample(archive['gripper_pose']))
        completed_this_session=0
        for slot in pending_slots(folder):
            slot_folder=folder/'attempts'/('slot_%02d'%slot);slot_folder.mkdir(parents=True,exist_ok=True)
            inflight=slot_folder/'inflight.json'
            if inflight.exists():
                previous=json.loads(inflight.read_text());interrupted=slot_folder/'interruptions';interrupted.mkdir(exist_ok=True)
                ordinal=len(list(interrupted.glob('*.json')))
                previous.update(status='interrupted_before_commit',recovered_utc=now(),elapsed_seconds_upper_bound=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(previous['started_utc'])).total_seconds(),exact_active_compute_seconds=None)
                atomic_json(interrupted/('%04d.json'%ordinal),previous)
            atomic_json(inflight,dict(parent_id=spec['parent_id'],attempt=slot,started_utc=now(),pid=os.getpid(),status='running'))
            tic=time.perf_counter();record=dict(parent_id=spec['parent_id'],attempt=slot,status='completed',success=False,route_type=None,guide_is_collection_only=True,started_utc=now())
            try:
                np.random.set_state(rng_before);random.setstate(python_rng_before)
                restored_descriptions,_=task.reset();observed=restore(task,snapshot,core)
                delta=strict_input_difference(inputs,reference_data(observed));world_delta=core.audit_difference(reference['world'],core.world_audit(task))
                record['restore']=dict(world_delta,observed_field_max_difference=delta,rgb_max_difference=int(delta['rgb']),language_equal=restored_descriptions==descriptions)
                if any(delta.values()) or world_delta['max_abs']!=0 or restored_descriptions!=descriptions:
                    raise RuntimeError('Strict initial reference restore failed')
                poses=[observed.gripper_pose];events=[observed.gripper_open];robot=task._robot;robot.arm.set_control_loop_enabled(True)
                if slot>0:
                    tip=np.asarray(robot.arm.get_tip().get_pose());first=task._task.get_waypoints()[0]._waypoint.get_position()
                    guide=(tip[:3]+np.asarray(first))*.5+np.array([0.,.10 if slot%2 else -.10,.08]);record['collection_guide']=guide.tolist()
                    path=robot.arm.get_path(guide.tolist(),quaternion=tip[3:].tolist(),ignore_collisions=False);done=False;steps=0
                    while not done:
                        done=path.step();task._scene.step()
                        if robot.arm.check_arm_collision():raise RuntimeError('Collision during free-space prefix')
                        observed=task.get_observation();poses.append(observed.gripper_pose);events.append(observed.gripper_open);steps+=1
                        if steps>1000:raise RuntimeError('Prefix exceeded step budget')
                    record['free_prefix_steps']=steps
                demo=task._scene.get_demo();poses.extend(obs.gripper_pose for obs in demo);events.extend(obs.gripper_open for obs in demo)
                success,_=task._task.success()
                if not success:raise RuntimeError('Original task success condition did not pass')
                route=np.asarray(poses);opened=np.asarray(events);normalized=core.resample(route)
                if not np.isfinite(route).all() or not np.isfinite(opened).all():raise ValueError('Nonfinite trajectory')
                duplicate=any(float(np.max(np.linalg.norm(normalized-old,axis=1)))<.01 for old in accepted)
                filename=slot_folder/('route_'+str(os.getpid())+'.npz');np.savez_compressed(filename,gripper_pose=route,gripper_open=opened)
                record.update(success=True,steps=len(route),route=filename.relative_to(folder).as_posix(),route_file_sha256=digest(filename),trajectory_sha256=core.array_hash(route),event_sha256=core.array_hash(opened),event_transition_indices=np.flatnonzero(np.diff(opened>.5)).tolist(),near_duplicate=duplicate,full_motion_collision_checked=False)
                accepted.append(normalized)
            except Exception as error:
                record.update(error=type(error).__name__+': '+str(error),traceback=traceback.format_exc())
            record.update(seconds=time.perf_counter()-tic,ended_utc=now());atomic_json(slot_folder/'record.json',record);inflight.unlink(missing_ok=True)
            completed_this_session+=1
            if max_new_attempts is not None and completed_this_session>=max_new_attempts and pending_slots(folder):
                atomic_json(folder/'worker_status.json',dict(status='partial_for_resume_validation',exit_code=3,pid=os.getpid(),started_utc=session,ended_utc=now(),completed_this_session=completed_this_session))
                return 3
        records=completed_records(folder)
        interruptions=list((folder/'attempts').glob('slot_*/interruptions/*.json'))
        atomic_json(folder/'closed.json',dict(status='complete',completed_attempts=len(records),successes=sum(r['success'] for r in records),failed_attempts=sum(not r['success'] for r in records),interrupted_proposal_executions=len(interruptions),total_proposal_executions=len(records)+len(interruptions),requested_attempts=spec['requested_attempts'],seconds_current_session=time.perf_counter()-started,ended_utc=now()))
        atomic_json(folder/'worker_status.json',dict(status='complete',exit_code=0,pid=os.getpid(),started_utc=session,ended_utc=now()))
        return 0
    except Exception as error:
        status='resume_blocked' if (folder/'reference_pointer.json').exists() else 'setup_failed'
        detail=dict(status=status,exit_code=0 if status=='setup_failed' else 2,error=type(error).__name__+': '+str(error),traceback=traceback.format_exc(),seconds_current_session=time.perf_counter()-started,pid=os.getpid(),started_utc=session,ended_utc=now())
        atomic_json(folder/'worker_status.json',detail)
        if status=='setup_failed':atomic_json(folder/'closed.json',dict(detail,completed_attempts=0,successes=0,failed_attempts=0,requested_attempts=spec['requested_attempts'],unattempted_slots=spec['requested_attempts'],total_proposal_executions=0))
        return 0 if status=='setup_failed' else 2
    finally:
        shutdown_error=None;active_exception=sys.exc_info()[0]
        try:
            if env is not None:env.shutdown()
        except BaseException as error:
            shutdown_error=type(error).__name__+': '+str(error)
            raise
        finally:
            state=json.loads((folder/'worker_status.json').read_text())
            atomic_json(session_path,dict(status=state['status'] if active_exception is None and shutdown_error is None else 'interrupted_or_shutdown_error',pid=os.getpid(),started_utc=session,ended_utc=now(),elapsed_seconds=time.perf_counter()-started,worker_exit_code=state.get('exit_code') if active_exception is None and shutdown_error is None else None,shutdown_error=shutdown_error,exception_in_flight=None if active_exception is None else active_exception.__name__))
            lock.unlink(missing_ok=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=281000);parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--resume',action='store_true',help='preserve a dead coordinator lock and continue unfinished parents')
    parser.add_argument('--worker-parent');parser.add_argument('--max-parents',type=int)
    parser.add_argument('--max-new-attempts',type=int,help='worker-only deterministic stop after committed attempts, for actual resume validation')
    args=parser.parse_args()
    if args.max_new_attempts is not None and (not args.worker_parent or args.max_new_attempts<1):parser.error('--max-new-attempts requires a worker parent and positive count')
    root=args.output.resolve();plan=ensure_registration(root,args.seed);ensure_source(root)
    if args.prepare_only:return
    if args.worker_parent:
        matches=[s for s in plan['parents'] if s['parent_id']==args.worker_parent]
        if len(matches)!=1:raise ValueError('Unknown registered parent')
        code=worker(root,plan,matches[0],args.max_new_attempts)
        mechanical_summary(root,plan)
        raise SystemExit(code)
    lock=root/'coordinator.lock'
    acquire_lock(lock,allow_stale=args.resume)
    try:
        # Recover a crash after a parent commit but before its role manifest
        # export, including when every parent is already closed.
        for role in dict.fromkeys(ROLES):export_role(root,role,plan)
        dispatched=0
        for spec in plan['parents']:
            folder=parent_folder(root,spec)
            if (folder/'closed.json').exists():continue
            if args.max_parents is not None and dispatched>=args.max_parents:break
            folder.mkdir(parents=True,exist_ok=True);execution=len(list(folder.glob('worker_*.log')))
            command=[sys.executable,str(Path(__file__).resolve()),'--output',str(root),'--seed',str(args.seed),'--worker-parent',spec['parent_id']]
            environment=os.environ.copy();release=Path(__file__).resolve().parents[1]
            environment['PYTHONPATH']=os.pathsep.join([str(release),str(release/'scripts'),environment.get('PYTHONPATH','')])
            tic=time.perf_counter()
            with (folder/('worker_%04d.log'%execution)).open('w') as log:
                child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=environment);code=child.wait()
            atomic_json(folder/('execution_%04d.json'%execution),dict(command=command,pid=child.pid,exit_code=code,seconds=time.perf_counter()-tic,ended_utc=now(),source_sha256=digest(__file__)))
            export_role(root,spec['split'],plan);dispatched+=1
            print(json.dumps(dict(parent_id=spec['parent_id'],split=spec['split'],worker_exit_code=code,closure_exists=(folder/'closed.json').exists())),flush=True)
            if code:raise RuntimeError('Parent worker requires attention; preserved original reference')
        mechanical_summary(root,plan)
    finally:
        mechanical_summary(root,plan)
        lock.unlink(missing_ok=True)


if __name__=='__main__':main()
