"""Create an independently guarded DEV-only snapshot of the unchanged 6c corpus."""
import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone

import numpy as np
from PIL import Image

from routeset.legacy_multitask_transfer import (PROTOCOL,INPUT_KEYS,FILES,mechanical_gate,validate_checkpoint_plan)
from scripts.snapshot_multitask_observations import child_path,digest,write_json,strict_restore_passes,CURRENT_KEYS


def read_parent(source,spec,hashes):
    folder=child_path(source/'DEV_MODEL'/'parents',spec['parent_id'])
    def read(path):
        path=child_path(folder,path.relative_to(folder))
        hashes[str(path)]=digest(path);return json.loads(path.read_text())
    closure=read(folder/'closed.json')
    if (folder/'worker.lock').exists():raise ValueError('Active/unresolved legacy parent')
    sessions=[read(path) for path in sorted((folder/'sessions').glob('*.json'))]
    if not sessions or any(row.get('elapsed_seconds') is None or row.get('shutdown_error') for row in sessions):
        raise ValueError('Unfinalized legacy sessions')
    inventory=dict(spec,closure=closure,worker_sessions=sessions,has_initial_observation=False,reference_count=0,
        source_folder=str(folder),strict_restore_passes=0)
    record_paths=sorted((folder/'attempts').glob('slot_*/record.json'))
    if closure['status']=='setup_failed':
        if record_paths or closure.get('completed_attempts')!=0 or closure.get('unattempted_slots')!=3:
            raise ValueError('Malformed setup failure denominator')
        return [],[],[],inventory
    if closure['status']!='complete' or closure.get('completed_attempts')!=3 or len(record_paths)!=3:
        raise ValueError('Exactly three finalized attempts required')
    pointer=read(folder/'reference_pointer.json');reference=child_path(folder,pointer['directory'])
    for name,key in [('reference.json','reference_sha256'),('observation.npz','observation_sha256'),('front.png','image_sha256')]:
        path=child_path(folder,Path(pointer['directory'])/name);hashes[str(path)]=digest(path)
        if hashes[str(path)]!=pointer[key]:raise ValueError('Legacy initial reference changed')
    metadata=read(reference/'reference.json')
    with np.load(reference/'observation.npz',allow_pickle=False) as archive:
        if set(archive.files)!=CURRENT_KEYS:raise ValueError('Unapproved current fields')
        current={key:archive[key].copy() for key in archive.files}
    rgb=np.asarray(Image.open(reference/'front.png'))
    if rgb.ndim!=3 or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8 or current['depth'].shape!=rgb.shape[:2]:
        raise ValueError('Invalid RGB/depth')
    for key,shape in [('gripper_pose',(7,)),('camera_intrinsics',(3,3)),('camera_extrinsics',(4,4))]:
        if current[key].shape!=shape:raise ValueError('Invalid current '+key)
    if current['gripper_open'].size!=1 or any(not np.isfinite(value).all() for value in current.values()):
        raise ValueError('Invalid current state')
    routes=[];records=[]
    for path in record_paths:
        record=read(path)
        if record.get('parent_id')!=spec['parent_id'] or record.get('status')!='completed':raise ValueError('Wrong attempt identity')
        restored=strict_restore_passes(record);inventory['strict_restore_passes']+=int(restored)
        if record['success']:
            if not restored:raise ValueError('Positive reference lacks strict restore')
            route=child_path(folder,record['route']);hashes[str(route)]=digest(route)
            if hashes[str(route)]!=record['route_file_sha256']:raise ValueError('Route file changed')
            with np.load(route,allow_pickle=False) as archive:
                if set(archive.files)!={'gripper_pose','gripper_open'}:raise ValueError('Unapproved label fields')
                poses=archive['gripper_pose'];events=archive['gripper_open']
                if poses.ndim!=2 or poses.shape[1]!=7 or events.shape!=(len(poses),) or not len(poses):raise ValueError('Invalid route shape')
                if not np.isfinite(poses).all() or not np.isfinite(events).all():raise ValueError('Nonfinite route')
                if not np.array_equal(poses[0],current['gripper_pose']) or events[0]!=current['gripper_open'].item():raise ValueError('Initial state mismatch')
                if hashlib.sha256(poses.tobytes()).hexdigest()!=record['trajectory_sha256'] or hashlib.sha256(events.tobytes()).hexdigest()!=record['event_sha256']:raise ValueError('Raw route/event hash changed')
                if len(poses)!=record['steps'] or np.flatnonzero(np.diff(events>.5)).tolist()!=record['event_transition_indices']:raise ValueError('Event/step mismatch')
            routes.append(str(route))
        records.append(dict(record,task=spec['task'],split='DEV_MODEL',source_dataset=str(source)))
    if sorted(r['attempt'] for r in records)!=[0,1,2] or closure['successes']!=len(routes) or closure['failed_attempts']!=3-len(routes):raise ValueError('Attempt accounting changed')
    descriptions=metadata['descriptions']
    if not descriptions or any(not isinstance(x,str) or not x.strip() for x in descriptions):raise ValueError('No instruction')
    observations=[];supervision=[]
    for index,instruction in enumerate(descriptions):
        identifier=spec['parent_id']+'_lang'+str(index)
        observations.append(dict(id=identifier,parent_id=spec['parent_id'],split='DEV_MODEL',image=str(reference/'front.png'),instruction=instruction))
        supervision.append(dict(id=identifier,parent_id=spec['parent_id'],split='DEV_MODEL',task=spec['task'],
            observation=str(reference/'observation.npz'),routes=routes,route_types=[None]*len(routes),semantic_targets=None,
            requested_attempts=3,full_motion_collision_checked=False,original_simulator_task_success_required=True,
            empty_reference_policy='Unknown, not a no-solution label'))
    inventory.update(has_initial_observation=True,reference_count=len(routes),instructions=len(descriptions))
    return observations,supervision,records,inventory


def build(config,output):
    validate_checkpoint_plan(config);output=Path(output).resolve();staging=output.with_name(output.name+'.staging')
    if output.name!=config['output_dataset']:raise ValueError('Registered output name required')
    if output.exists() or staging.exists():raise FileExistsError('Preserve existing export/staging')
    before=mechanical_gate(config);hashes=dict(before['source_files_sha256'])
    source=Path(config['source_dataset']).resolve();observations=[];supervision=[];attempts=[];parents=[]
    for spec in config['selected_requested_parents']:
        obs,sup,records,parent=read_parent(source,spec,hashes)
        observations.extend(obs);supervision.extend(sup);attempts.extend(records);parents.append(parent)
    if any(set(row)!=INPUT_KEYS for row in observations):raise ValueError('Input whitelist changed')
    after=mechanical_gate(config)
    if before!=after:raise ValueError('Mechanical evidence changed during export')
    for path,expected in hashes.items():
        if digest(path)!=expected:raise ValueError('Committed source changed during export')
    output.parent.mkdir(parents=True,exist_ok=True);staging.mkdir()
    for name,rows in [('observations',observations),('supervision',supervision),('attempts',attempts)]:
        (staging/(name+'.jsonl')).write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in rows),encoding='utf-8')
    write_json(staging/'parent_inventory.json',parents)
    manifest=dict(protocol=PROTOCOL,created_at=datetime.now(timezone.utc).isoformat(),registration=config,
        requested_parents=12,requested_attempts=36,parents_with_observation=sum(p['has_initial_observation'] for p in parents),
        actual_attempt_records=len(attempts),successful_reference_routes=sum(p['reference_count'] for p in parents),
        setup_failed_parents=sum(p['closure']['status']=='setup_failed' for p in parents),
        unattempted_slots=36-len(attempts),
        parents_with_zero_reference=sum(p['has_initial_observation'] and not p['reference_count'] for p in parents),
        observations=len(observations),mechanical_gate=after,source_files_sha256=hashes,
        output_files_sha256={name:digest(staging/name) for name in FILES},exporter_sha256=digest(__file__),
        current_gate_required_before_model_use=True,training_authorized=False,original_source_unchanged=True)
    write_json(staging/'export_manifest.json',manifest);staging.rename(output);return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=build(json.loads(a.registration.read_text()),a.output)
    print(json.dumps({key:result[key] for key in ('requested_parents','actual_attempt_records','successful_reference_routes','observations')}))
