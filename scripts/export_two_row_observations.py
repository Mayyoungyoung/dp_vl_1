"""Closed, fixed TRAIN/DEV prefix of formal116; other roles stay mechanical-only."""
import argparse
import json
from pathlib import Path
from datetime import datetime,timezone

import numpy as np
from PIL import Image

from scripts import collect_two_row_formal as collector
from scripts.collect_two_row_canonical_repair import route_acceptance
from scripts.evaluate_observed_two_row import validate_labels
from scripts.snapshot_multitask_observations import child_path,digest,write_json,CURRENT_KEYS

PROTOCOL='observed_two_row_closed_prefix_export_v1'
INPUT_KEYS={'id','parent_id','split','image','instruction'}
SELECTED=list(range(16))+list(range(64,76))
PREFIX_TRAIN_COUNTS={
    'observed_two_row_prefix28_development_v1':16,
    'observed_two_row_prefix44_development_v1':32,
    'observed_two_row_prefix76_development_v1':64,
}
OUTPUT_FILES=('observations.jsonl','supervision.jsonl','attempts.jsonl','parent_inventory.json')


def selection_guard(selection):
    count=PREFIX_TRAIN_COUNTS.get(selection.get('protocol'))
    indices=selection.get('selected_indices',[])
    quality=selection.get('train_quality_audit',{})
    if (count is None or selection.get('formal_protocol')!='observed_two_row_formal116_registration_v1'
            or indices!=list(range(count))+list(range(64,76)) or any(type(i) is not int for i in indices)
            or selection.get('roles')!=['TRAIN','DEV_MODEL'] or selection.get('allowed_raw_roles')!=['TRAIN','DEV_MODEL']
            or selection.get('requested_parents')!={'TRAIN':count,'DEV_MODEL':12}
            or selection.get('requested_inputs')!=(count+12)*3
            or selection.get('requested_route_proposals')!=(count+12)*27
            or selection.get('require_all_selected_parents_closed') is not True
            or selection.get('failed_parent_replacements')!=0 or selection.get('allow_public_development_parents') is not False
            or quality.get('indices')!=list(range(count)) or quality.get('use_dev') is not False
            or quality.get('exclude_or_truncate_long_routes') is not False
            or quality.get('horizon')!=24 or quality.get('pixel_stride')!=2 or quality.get('endpoint_axis_bound_m')!=.05):
        raise ValueError('Only the prospectively fixed16 TRAIN, fixed32 TRAIN or fixed64 TRAIN +12 DEV prefixes are permitted')
    return count


def closed_prefix(source,selection):
    source=Path(source).resolve();selection_guard(selection)
    registration,manifest=collector.verify_corpus(source)
    plans=[registration['parent_plan'][i] for i in selection['selected_indices']]
    if any(p['index']!=i or p['role']!=('TRAIN' if i<64 else 'DEV_MODEL')
           for i,p in zip(selection['selected_indices'],plans)):raise ValueError('Forbidden raw role or registration identity')
    closures=[];missing=[]
    for p in plans:
        path=source/'closures'/('%03d.json'%p['index'])
        if not path.exists():missing.append(p['parent_id']);continue
        c=json.loads(path.read_text())
        if any(c[k]!=p[k] for k in ('parent_id','index','role')) or c['requested_routes']!=27:
            raise ValueError('Closure identity or route budget changed')
        closures.append(c)
    return registration,manifest,plans,closures,missing


def strict_restore(record):
    check=record.get('strict_restore',{});world=check.get('world',{})
    observed=check.get('observation',{})
    expected={'front_rgb_equal','front_depth_equal','gripper_pose_equal','gripper_open_equal',
              'front_camera_intrinsics_equal','front_camera_extrinsics_equal'}
    return (check.get('passed') is True and world.get('max_abs')==0 and world.get('rgb_max_difference')==0
        and world.get('same_object_inventory') is True and world.get('global_inventory_equal') is True
        and set(observed)==expected and all(value is True for value in observed.values()))


def read_parent(source,plan,closed,hashes):
    """Read only one selected closed parent; never recurse into another role."""
    folder=child_path(source/'parents'/plan['role'],plan['parent_id'])
    inventory=dict(parent_id=plan['parent_id'],index=plan['index'],split=plan['role'],requested_inputs=3,
        requested_routes=27,closure=closed,observed_inputs=0,positive_references=0,unobserved_requested_inputs=3)
    def checked(name,expected=None):
        path=child_path(folder,name);value=digest(path)
        if expected is not None and value!=expected:raise ValueError('Closed parent source changed: '+name)
        hashes[str(path)]=value;return path
    for name,value in closed['mechanical_files_sha256'].items():checked(name,value)
    if not (folder/'artifact_hashes.json').exists():
        if closed['initial_observation_saved'] or closed['completed_slots']:raise ValueError('Unindexed closed parent evidence')
        return [],[],[],inventory
    artifacts=json.loads(checked('artifact_hashes.json').read_text())
    def artifact(name):
        if name not in artifacts:raise ValueError('Unregistered parent artifact: '+name)
        return checked(name,artifacts[name])
    def rows(name):
        if name not in artifacts:return []
        return [json.loads(line) for line in artifact(name).read_text().splitlines() if line.strip()]
    observations=rows('observations.jsonl');original_labels=rows('supervision.jsonl');attempts=rows('attempts.jsonl')
    if len(attempts)!=closed['completed_slots']:raise ValueError('Closed attempt count differs from actual records')
    seen=set()
    for record in attempts:
        key=(record['input_id'],record['attempt'])
        if (record['parent_id']!=plan['parent_id'] or record['input_id'] not in [plan['parent_id']+'_target%d'%i for i in range(3)]
                or record['attempt'] not in range(9) or key in seen):raise ValueError('Attempt identity/budget changed')
        seen.add(key)
    if not observations:
        if any(r['success'] for r in attempts):raise ValueError('Positive route has no recorded observation')
        return [],[],attempts,inventory
    if not closed['initial_observation_saved']:raise ValueError('Inputs lack accepted initial-state receipt')
    parent=plan['parent_id'];current_path=artifact(parent+'/observation.npz')
    with np.load(current_path,allow_pickle=False) as a:
        if set(a.files)!=CURRENT_KEYS:raise ValueError('Strict current observation fields required')
        current={k:a[k].copy() for k in a.files}
    if (current['gripper_pose'].shape!=(7,) or current['gripper_open'].size!=1
            or current['camera_intrinsics'].shape!=(3,3) or current['camera_extrinsics'].shape!=(4,4)
            or any(not np.isfinite(v).all() for v in current.values())):raise ValueError('Invalid current observation')
    image=artifact(parent+'/front.png');rgb=np.asarray(Image.open(image))
    if rgb.shape!=(224,224,3) or rgb.dtype!=np.uint8 or current['depth'].shape!=(224,224):raise ValueError('Original RGB-D shape required')
    verification=artifact(parent+'/verification_only.npz')
    with np.load(verification,allow_pickle=False) as a:
        geometry={key:a[key].copy() for key in ('obstacle_centers','obstacle_halfsizes')};goals=a['target_centers'].copy()
    cfgpath=artifact('route_configs/'+parent+'.json');cfg=json.loads(cfgpath.read_text())
    if cfg!=plan['config']:raise ValueError('Parent label geometry differs from registration')
    if closed['actual_geometry_1mm_sha256']!=plan['registered_geometry_1mm_sha256']:
        raise ValueError('Actual physical geometry differs from registration')
    labels_by_id={r['id']:r for r in original_labels}
    if len(labels_by_id)!=len(original_labels):raise ValueError('Duplicate original supervision')
    inputs=[];labels=[];seen_inputs=set()
    for row in observations:
        if set(row)!=INPUT_KEYS or row['parent_id']!=parent or row['split']!=plan['role'] or row['id'] in seen_inputs:
            raise ValueError('Observation identity/strict input whitelist changed')
        expected_ids=[parent+'_target%d'%i for i in range(3)]
        if row['id'] not in expected_ids or row['image']!=parent+'/front.png':raise ValueError('Unregistered target input')
        target=expected_ids.index(row['id']);seen_inputs.add(row['id'])
        expected_instruction='Move the gripper to touch the %s sphere while avoiding the gray posts.'%plan['target_colors'][target]['name']
        if row['instruction']!=expected_instruction:raise ValueError('Recorded language differs from physical color registration')
        routes=[];types=[]
        for record in sorted((r for r in attempts if r['input_id']==row['id'] and r['success']),key=lambda r:r['attempt']):
            if not strict_restore(record):raise ValueError('Positive route lacks strict same-state evidence')
            filename=parent+'/'+record['trajectory']['file'];path=artifact(filename)
            if digest(path)!=record['trajectory']['sha256']:raise ValueError('Positive trajectory SHA changed')
            with np.load(path,allow_pickle=False) as a:
                poses=a['gripper_pose'];events=a['gripper_open']
                if (poses.ndim!=2 or poses.shape[1]!=7 or len(poses)<2 or events.shape!=(len(poses),)
                        or not np.isfinite(poses).all() or not np.isfinite(events).all()):raise ValueError('Invalid raw reference')
                if not np.array_equal(poses[0],current['gripper_pose']) or events[0]!=current['gripper_open'].item():raise ValueError('Reference initial state differs')
                if not np.all((events>.5)==(current['gripper_open'].item()>.5)):raise ValueError('Reach reference event changed')
                if collector.physical.legacy.array_hash(poses)!=record['trajectory']['pose_array_sha256'] or len(poses)!=record['trajectory']['samples']:
                    raise ValueError('Raw pose record changed')
                fields,h24,passed=route_acceptance(poses[:,:3],goals,target,cfg)
                canonical=lambda value:json.dumps(value,sort_keys=True)
                if (not passed or canonical(fields['actual_route_type'])!=canonical(record['actual_route_type'])
                        or not np.array_equal(a['xyz_24'],h24)):raise ValueError('Original raw/H24 acceptance differs')
            routes.append(str(path));types.append(record['actual_route_type'])
        specification=dict(centers=goals.tolist(),target_index=target,tolerance=.03)
        validate_labels(geometry,specification,cfg,types)
        original=labels_by_id.get(row['id'])
        if original is not None:
            if (original['parent_id']!=parent or original['split']!=plan['role']
                    or [str(child_path(folder,p)) for p in original['routes']]!=routes
                    or original['route_types']!=types or original['semantic_targets']!=specification):
                raise ValueError('Original supervision disagrees with completed positive records')
        inputs.append(dict(row,image=str(image)))
        labels.append(dict(id=row['id'],parent_id=parent,split=plan['role'],task='rlbench_derived_two_row_reach',
            observation=str(current_path),routes=routes,route_types=types,verification_only=str(verification),
            route_config=str(cfgpath),route_config_sha256=digest(cfgpath),semantic_targets=specification,
            reference_set_complete=False,requested_attempts=9,original_supervision_present=original is not None,
            missing_reference_policy='Known positives only; absence is not an invalidity or no-solution label'))
    if set(labels_by_id)-seen_inputs:raise ValueError('Supervision has no actual input')
    inventory.update(observed_inputs=len(inputs),positive_references=sum(len(r['routes']) for r in labels),
        unobserved_requested_inputs=3-len(inputs),inputs_without_reference=sum(not r['routes'] for r in labels))
    return inputs,labels,attempts,inventory


def build(source,selection,output):
    source,output=Path(source).resolve(),Path(output).resolve();staging=output.with_name(output.name+'.staging')
    if output.exists() or staging.exists():raise FileExistsError('Fresh atomic export required')
    registration,corpus,plans,closures,missing=closed_prefix(source,selection)
    if missing:raise ValueError('Registered prefix not fully closed: '+','.join(missing))
    gate=collector.live_layout_gate(source)
    if set(gate['blocked_parent_ids'])&{p['parent_id'] for p in plans}:raise ValueError('Selected layout gate blocked')
    hashes={str(source/name):digest(source/name) for name in ('registration.json','corpus_manifest.json')}
    inputs=[];labels=[];attempts=[];parents=[]
    for plan,closure in zip(plans,closures):
        path=source/'closures'/('%03d.json'%plan['index']);hashes[str(path)]=digest(path)
        x,y,z,parent=read_parent(source,plan,closure,hashes);inputs.extend(x);labels.extend(y);attempts.extend(z);parents.append(parent)
    if any(digest(path)!=value for path,value in hashes.items()):raise ValueError('Selected source changed during export')
    output.parent.mkdir(parents=True,exist_ok=True);staging.mkdir()
    for name,rows in [('observations',inputs),('supervision',labels),('attempts',attempts)]:
        (staging/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in rows),encoding='utf-8')
    write_json(staging/'parent_inventory.json',parents)
    manifest=dict(protocol=PROTOCOL,created_at=datetime.now(timezone.utc).isoformat(),selection=selection,source_dataset=str(source),
        registration_sha256=corpus['registration_sha256'],requested_parents=len(plans),
        requested_inputs=len(plans)*3,requested_routes=len(plans)*27,
        actual_inputs=len(inputs),positive_references=sum(p['positive_references'] for p in parents),
        unobserved_requested_inputs=sum(p['unobserved_requested_inputs'] for p in parents),
        actual_attempt_records=len(attempts),source_files_sha256=hashes,
        output_files_sha256={name:digest(staging/name) for name in OUTPUT_FILES},mechanical_gate_at_export=gate,
        exporter_sha256=digest(__file__),live_gate_required_before_model_use=True,locked_raw_access=False)
    write_json(staging/'export_manifest.json',manifest);staging.rename(output);return manifest


def verify_export(output):
    output=Path(output).resolve();m=json.loads((output/'export_manifest.json').read_text())
    if m.get('protocol')!=PROTOCOL:raise ValueError('Explicit two-row export protocol required')
    selection_guard(m['selection']);count=len(m['selection']['selected_indices'])
    if (m['requested_parents'],m['requested_inputs'],m['requested_routes'])!=(count,count*3,count*27):
        raise ValueError('Export requested denominators differ from fixed registration')
    if set(m['output_files_sha256'])!=set(OUTPUT_FILES):raise ValueError('Incomplete export hashes')
    for name,value in m['output_files_sha256'].items():
        if digest(output/name)!=value:raise ValueError('Export changed')
    for name,value in m['source_files_sha256'].items():
        if digest(name)!=value:raise ValueError('Closed source changed')
    _,_,plans,_,missing=closed_prefix(m['source_dataset'],m['selection'])
    gate=collector.live_layout_gate(Path(m['source_dataset']))
    if missing or set(gate['blocked_parent_ids'])&{p['parent_id'] for p in plans}:raise ValueError('Current model-use gate blocked')
    return m,gate


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--readiness-only',action='store_true');a=p.parse_args()
    config=json.loads(a.selection.read_text())
    if a.readiness_only:
        *_,missing=closed_prefix(a.source,config);print(json.dumps(dict(ready=not missing,missing=missing)))
    else:
        if a.output is None:p.error('--output required for export')
        m=build(a.source,config,a.output);print(json.dumps({k:m[k] for k in ('requested_parents','actual_inputs','positive_references')}))
