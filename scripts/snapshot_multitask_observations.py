"""Freeze a pre-registered six-task TRAIN/DEV prefix, retaining all outcomes.

Raw locked-role data is never read. Mechanical cross-role hashes are mandatory.
This exporter does not create semantic target or route-validity labels.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from routeset.multitask_fingerprints import fingerprint
from scripts.audit_multitask_layout_hashes import audit
from scripts.observation_collect_multitask import TASKS, registration

PROTOCOL='multitask_closed_prefix_snapshot_v1'
INPUT_KEYS={'id','parent_id','split','image','instruction'}
CURRENT_KEYS={'depth','gripper_pose','gripper_open','camera_intrinsics','camera_extrinsics'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def child_path(parent,relative):
    path=(parent/relative).resolve()
    try:path.relative_to(parent.resolve())
    except ValueError:raise ValueError('Reference escapes registered parent')
    return path


def select_prefix(plan,selection=None):
    expected=registration(282000,'validated-five-v1','interleaved-early-dev-v1')
    if plan!=expected:raise ValueError('Unexpected six-task seed/policy/schedule registration')
    if selection is None:
        selected=[row for row in plan['parents'] if row['parent_index'] in (0,1,16,17)]
        assert len(selected)==24 and {row['task'] for row in selected}==set(TASKS)
        return selected
    train_count={'multitask_prefix60_prospective_selection_v1':8,
                 'multitask_prefix108_prospective_selection_v1':16}.get(selection.get('protocol'))
    if train_count is None:raise ValueError('Unrecognized explicit prefix selection registration')
    if (selection.get('seed')!=282000 or selection.get('motion_camera_policy')!='validated-five-v1'
            or selection.get('schedule')!='interleaved-early-dev-v1'
            or selection.get('train_parent_indices')!=list(range(train_count))
            or selection.get('dev_model_parent_indices')!=[16,17]
            or selection.get('requested_parent_counts')!={'TRAIN':6*train_count,'DEV_MODEL':12}
            or selection.get('requested_attempts')!=3*(6*train_count+12)):
        raise ValueError('Unrecognized explicit prefix selection registration')
    selected=[row for row in plan['parents'] if (row['split']=='TRAIN' and row['parent_index'] in range(train_count))
              or (row['split']=='DEV_MODEL' and row['parent_index'] in (16,17))]
    if selection.get('selected_requested_parents')!=selected:
        raise ValueError('Requested parent list differs from fixed prefix registration')
    return selected


def gate(source,compare_sources):
    if not compare_sources:raise ValueError('At least one explicit cross-batch source is required')
    if len(set([source]+compare_sources))!=1+len(compare_sources):raise ValueError('Duplicate audit source')
    intra=audit([source]);cross=audit([source]+compare_sources)
    blocked=sorted(task for task in TASKS if any(result['task_usage_status'].get(task,'').startswith('blocked_') for result in (intra,cross)))
    return dict(intra_batch=intra,cross_batch=cross,blocked_tasks=blocked,
        current_gate_required_before_model_use=True)


def strict_restore_passes(record):
    restore=record.get('restore')
    return bool(restore and restore.get('max_abs')==0 and restore.get('same_object_inventory') is True
        and restore.get('language_equal') is True and set(restore.get('observed_field_max_difference',{}))==CURRENT_KEYS|{'rgb'}
        and not any(restore['observed_field_max_difference'].values()))


def read_parent(source,spec,source_manifest):
    folder=source/spec['split']/'parents'/spec['parent_id'];hashes={}
    def read_json(path):
        hashes[str(path)]=digest(path);return json.loads(path.read_text(encoding='utf-8'))
    closure=read_json(folder/'closed.json')
    if (folder/'worker.lock').exists():raise ValueError('Selected parent still has an active/unresolved worker lock')
    sessions=[read_json(path) for path in sorted((folder/'sessions').glob('*.json'))]
    if not sessions or any(row.get('elapsed_seconds') is None for row in sessions):
        raise ValueError('Selected parent worker sessions are not finalized')
    records=[];observations=[];supervision=[]
    if closure.get('requested_attempts')!=3:raise ValueError('Closure lost requested proposal budget')
    record_paths=sorted((folder/'attempts').glob('slot_*/record.json'))
    inventory=dict(**spec,closure=closure,has_initial_observation=False,reference_count=0,
        strict_restore_passes=0,failed_strict_restore_attempts=[],source_folder=str(folder),worker_sessions=sessions,
        finalized_worker_wall_seconds=sum(row['elapsed_seconds'] for row in sessions))
    if closure.get('status')=='setup_failed':
        if record_paths or closure.get('completed_attempts')!=0 or closure.get('unattempted_slots')!=3:
            raise ValueError('Setup failure must retain three unattempted slots')
        return observations,supervision,records,inventory,hashes
    if closure.get('status')!='complete' or closure.get('completed_attempts')!=3 or len(record_paths)!=3:
        raise ValueError('Selected parent has not committed all three proposal outcomes')
    pointer=read_json(folder/'reference_pointer.json');reference=child_path(folder,pointer['directory'])
    for name,key in [('reference.json','reference_sha256'),('observation.npz','observation_sha256'),('front.png','image_sha256')]:
        path=reference/name;hashes[str(path)]=digest(path)
        if hashes[str(path)]!=pointer[key]:raise ValueError('Initial reference file hash changed: '+name)
    meta=read_json(reference/'reference.json')
    with np.load(reference/'observation.npz',allow_pickle=False) as archive:
        if set(archive.files)!=CURRENT_KEYS:raise ValueError('Unapproved current observation fields')
        current={key:archive[key].copy() for key in archive.files}
    rgb=np.asarray(Image.open(reference/'front.png'))
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or current['depth'].shape!=rgb.shape[:2]:
        raise ValueError('Initial RGB/depth shape mismatch')
    for key,shape in [('gripper_pose',(7,)),('camera_intrinsics',(3,3)),('camera_extrinsics',(4,4))]:
        if current[key].shape!=shape:raise ValueError('Invalid current '+key)
    if current['gripper_open'].size!=1 or any(not np.isfinite(value).all() for value in current.values()):
        raise ValueError('Nonfinite or invalid current state')
    mechanical=read_json(folder/'mechanical_fingerprint.json')
    expected=fingerprint(meta['world'],dict(current,rgb=rgb),spec);expected['rgb_file_sha256']=digest(reference/'front.png')
    if mechanical!=expected:raise ValueError('Initial physical/observation fingerprint differs from recorded audit')
    descriptions=meta['descriptions']
    if not descriptions or any(not isinstance(value,str) or not value.strip() for value in descriptions):
        raise ValueError('Missing original instruction descriptions')
    routes=[];successful=[]
    for path in record_paths:
        record=read_json(path)
        if record.get('parent_id')!=spec['parent_id'] or record.get('status')!='completed':
            raise ValueError('Wrong or uncommitted proposal identity')
        if record.get('motion_camera_policy')!='validated-five-v1':raise ValueError('Recorded policy differs from registration')
        for key in ('collector_sha256','restore_helper_sha256','fingerprint_helper_sha256'):
            if record.get(key)!=source_manifest[key]:raise ValueError('Attempt source differs from registered source')
        restored=strict_restore_passes(record)
        inventory['strict_restore_passes']+=int(restored)
        if not restored:inventory['failed_strict_restore_attempts'].append(record['attempt'])
        render=record.get('demo_render_audit')
        if render is not None and (render['policy']!='validated-five-v1' or not render['flags_restored'] or
                not render['initial_and_restore_rgbd_on'] or render['rgbd_suppressed']!=(spec['task']!='push_button')):
            raise ValueError('Recorded camera execution/restoration is invalid')
        if record.get('success'):
            if not restored or render is None:raise ValueError('Successful reference lacks strict restore/render evidence')
            route=child_path(folder,record['route']);hashes[str(route)]=digest(route)
            if hashes[str(route)]!=record['route_file_sha256']:raise ValueError('Positive reference file hash changed')
            with np.load(route,allow_pickle=False) as archive:
                if set(archive.files)!={'gripper_pose','gripper_open'}:raise ValueError('Unexpected route label fields')
                poses=archive['gripper_pose'];opened=archive['gripper_open']
                if poses.ndim!=2 or poses.shape[1]!=7 or not len(poses) or opened.shape!=(len(poses),):
                    raise ValueError('Invalid pose/open reference shape')
                if not np.isfinite(poses).all() or not np.isfinite(opened).all():raise ValueError('Nonfinite reference')
                if not np.array_equal(poses[0],current['gripper_pose']) or opened[0]!=current['gripper_open'].item():
                    raise ValueError('Reference starts at a different initial state')
                if hashlib.sha256(poses.tobytes()).hexdigest()!=record['trajectory_sha256'] or hashlib.sha256(opened.tobytes()).hexdigest()!=record['event_sha256']:
                    raise ValueError('Raw trajectory/event hash changed')
                if np.flatnonzero(np.diff(opened>.5)).tolist()!=record['event_transition_indices']:
                    raise ValueError('Raw gripper transition index differs from record')
                if record['steps']!=len(poses):raise ValueError('Recorded trajectory length differs')
            routes.append(str(route));successful.append(record['attempt'])
        records.append(dict(record,source_dataset=str(source),split=spec['split'],task=spec['task']))
    if sorted(row['attempt'] for row in records)!=[0,1,2]:raise ValueError('Exactly three distinct proposal slots required')
    if closure.get('successes')!=len(routes) or closure.get('failed_attempts')!=3-len(routes):
        raise ValueError('Closure success/failure counts disagree with committed slots')
    for index,instruction in enumerate(descriptions):
        identifier=spec['parent_id']+'_lang'+str(index)
        observations.append(dict(id=identifier,parent_id=spec['parent_id'],split=spec['split'],image=str(reference/'front.png'),instruction=instruction))
        supervision.append(dict(id=identifier,parent_id=spec['parent_id'],split=spec['split'],task=spec['task'],
            observation=str(reference/'observation.npz'),routes=routes,route_types=[None]*len(routes),semantic_targets=None,
            original_simulator_task_success_required=True,successful_attempts=successful,requested_attempts=3,
            full_motion_collision_checked=False,empty_reference_policy='Unknown solution availability; no negative existence label'))
    inventory.update(has_initial_observation=True,reference_count=len(routes),instructions=len(descriptions))
    return observations,supervision,records,inventory,hashes


def build(source,compare_sources,output,selection_registration=None):
    source=Path(source).resolve();compare_sources=[Path(value).resolve() for value in compare_sources];output=Path(output).resolve()
    staging=output.with_name(output.name+'.staging');blocked=output.with_name(output.name+'.blocked.json')
    if output.exists() or staging.exists() or blocked.exists():raise FileExistsError('Preserve existing snapshot, staging or blocked audit')
    plan=json.loads((source/'partition_manifest.json').read_text())
    selection=None;selection_hashes={}
    if selection_registration is not None:
        selection_registration=Path(selection_registration).resolve()
        selection=json.loads(selection_registration.read_text(encoding='utf-8'))
        if (Path(selection['source_dataset']).resolve()!=source
                or [Path(path).resolve() for path in selection['compare_sources']]!=compare_sources
                or selection['source_partition_manifest_sha256']!=digest(source/'partition_manifest.json')):
            raise ValueError('Prefix selection source/compare/registration hash mismatch')
        if selection.get('output_dataset')!=output.name:
            raise ValueError('Prefix selection requires the registered new dataset name')
        selection_hashes[str(selection_registration)]=digest(selection_registration)
    selected=select_prefix(plan,selection)
    missing=[row['parent_id'] for row in selected if not (source/row['split']/'parents'/row['parent_id']/'closed.json').exists()]
    if missing:raise ValueError('Registered prefix is not fully closed: '+','.join(missing))
    gates=gate(source,compare_sources)
    if gates['blocked_tasks']:
        output.parent.mkdir(parents=True,exist_ok=True);write_json(blocked,dict(status='blocked_layout_duplicate',selected_requested_parents=selected,layout_gate=gates))
        raise ValueError('Layout gate blocks pre-registered snapshot tasks: '+','.join(gates['blocked_tasks']))
    source_manifest=json.loads((source/'source_manifest.json').read_text())
    observations=[];supervision=[];attempts=[];parents=[]
    hashes=dict(selection_hashes,**{str(source/'partition_manifest.json'):digest(source/'partition_manifest.json'),str(source/'source_manifest.json'):digest(source/'source_manifest.json')})
    for spec in selected:
        obs,sup,records,parent,files=read_parent(source,spec,source_manifest)
        observations.extend(obs);supervision.extend(sup);attempts.extend(records);parents.append(parent);hashes.update(files)
    if any(set(row)!=INPUT_KEYS for row in observations):raise ValueError('Observation input whitelist changed')
    if len({row['id'] for row in observations})!=len(observations):raise ValueError('Duplicate input id')
    # Parent commits must stay immutable; live unrelated collection may advance.
    for path,expected in hashes.items():
        if digest(path)!=expected:raise RuntimeError('Selected closed source changed during snapshot: '+path)
    gates=gate(source,compare_sources)
    if gates['blocked_tasks']:
        output.parent.mkdir(parents=True,exist_ok=True);write_json(blocked,dict(status='blocked_layout_duplicate',selected_requested_parents=selected,layout_gate=gates))
        raise ValueError('A new mechanical duplicate appeared during snapshot')
    manifest=dict(protocol=PROTOCOL,created_at=datetime.now(timezone.utc).isoformat(),source_dataset=str(source),
        snapshot_script_sha256=digest(__file__),
        compare_sources=[str(path) for path in compare_sources],seed=282000,selected_requested_parents=selected,
        requested_parents=len(selected),requested_attempts=3*len(selected),actual_attempt_records=len(attempts),
        setup_failed_parents=sum(row['closure']['status']=='setup_failed' for row in parents),
        unattempted_slots=sum(row['closure'].get('unattempted_slots',0) for row in parents),
        total_proposal_executions=sum(row['closure'].get('total_proposal_executions',0) for row in parents),
        interrupted_proposal_executions=sum(row['closure'].get('interrupted_proposal_executions',0) for row in parents),
        finalized_worker_wall_seconds=sum(row['finalized_worker_wall_seconds'] for row in parents),
        parents_with_observation=sum(row['has_initial_observation'] for row in parents),
        parents_with_zero_reference=sum(row['has_initial_observation'] and row['reference_count']==0 for row in parents),
        successful_reference_routes=sum(row['reference_count'] for row in parents),observations=len(observations),
        source_manifest=source_manifest,source_files_sha256=hashes,layout_gate=gates,
        evaluation_scope='Positive-reference path/event reconstruction only; semantic targets, generated success/validity/types remain unknown',
        current_gate_required_before_model_use=True,model_input_keys=sorted(INPUT_KEYS),
        excluded_raw_roles=['DEV_SCORE','CALIBRATION','TEST_LOCKED'])
    if selection is not None:
        manifest.update(selection_registration=selection,selection_registration_path=str(selection_registration),
            selection_registration_sha256=selection_hashes[str(selection_registration)],
            requested_parent_counts=dict(selection['requested_parent_counts']),
            reused_dev_policy='The same DEV parent indices16/17 were used in prefix24 model selection; no independent-test claim.')
    output.parent.mkdir(parents=True,exist_ok=True);staging.mkdir()
    for name,rows in [('observations',observations),('supervision',supervision),('attempts',attempts)]:
        (staging/(name+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
    write_json(staging/'parent_inventory.json',parents)
    manifest['snapshot_files_sha256']={name:digest(staging/name) for name in
        ('observations.jsonl','supervision.jsonl','attempts.jsonl','parent_inventory.json')}
    write_json(staging/'snapshot_manifest.json',manifest)
    staging.rename(output)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--compare-source',type=Path,action='append',required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection-registration',type=Path,help='explicit pre-registered prefix60 JSON; omitted preserves original prefix24 behavior')
    args=parser.parse_args();result=build(args.source,args.compare_source,args.output,args.selection_registration)
    print(json.dumps({key:result[key] for key in ('protocol','requested_parents','actual_attempt_records','setup_failed_parents','parents_with_zero_reference','successful_reference_routes','observations')}))
