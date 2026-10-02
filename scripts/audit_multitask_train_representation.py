"""Read only the twelve registered TRAIN parents of a sealed pilot snapshot.

Measures reference H24/event preservation and endpoint support by the actual
initial observed point grid. No label correction, route filtering or DEV scale
selection is performed. Semantic or robot task validity is not inferred.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from scripts.audit_multitask_layout_hashes import audit

HORIZON=24
PIXEL_STRIDE=2
ENDPOINT_AXIS_BOUND=.05


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def phases(xyz,opened):
    states=np.asarray(opened)>.5;cuts=np.r_[0,np.flatnonzero(states[1:]!=states[:-1])+1,len(states)]
    return states[cuts[:-1]].tolist(),[(xyz[a],xyz[b-1]) for a,b in zip(cuts[:-1],cuts[1:])]


def point_to_polyline(points,path):
    start=path[:-1];delta=path[1:]-start;norm2=(delta**2).sum(axis=1)
    t=np.clip(((points[:,None]-start[None])*delta[None]).sum(axis=2)/np.maximum(norm2[None],1e-16),0,1)
    projected=start[None]+t[...,None]*delta[None]
    return np.linalg.norm(points[:,None]-projected,axis=2).min(axis=1)


def route_diagnostics(poses,opened,resampled,events,world_points):
    raw=np.asarray(poses,dtype=float)[:,:3];resampled=np.asarray(resampled,dtype=float)
    sequence,edges=phases(raw,opened);new_sequence,new_edges=phases(resampled,events)
    phase_error=None
    if sequence==new_sequence:
        phase_error=max(float(np.linalg.norm(a-b)) for pair,new_pair in zip(edges,new_edges) for a,b in zip(pair,new_pair))
    displacement=np.abs(world_points-raw[-1]);euclidean=np.linalg.norm(displacement,axis=1);axis=displacement.max(axis=1)
    raw_length=float(np.linalg.norm(np.diff(raw,axis=0),axis=1).sum())
    new_length=float(np.linalg.norm(np.diff(resampled,axis=0),axis=1).sum())
    deviation=point_to_polyline(raw,resampled)
    return dict(raw_steps=len(raw),resampled_steps=len(resampled),raw_event_sequence=sequence,resampled_event_sequence=new_sequence,
        event_sequence_preserved=sequence==new_sequence,phase_boundary_max_error_m=phase_error,
        initial_error_m=float(np.linalg.norm(raw[0]-resampled[0])),endpoint_error_m=float(np.linalg.norm(raw[-1]-resampled[-1])),
        raw_length_m=raw_length,resampled_length_m=new_length,length_ratio=new_length/raw_length if raw_length else None,
        raw_sample_to_resampled_polyline_max_m=float(deviation.max()),raw_sample_to_resampled_polyline_rms_m=float(np.sqrt(np.mean(deviation**2))),
        endpoint_nearest_observed_point_m=float(euclidean.min()),endpoint_nearest_observed_point_linf_m=float(axis.min()),
        hard_surface_anchor_with_5cm_axis_residual_has_support=bool(np.any(axis<=ENDPOINT_AXIS_BOUND)),
        note='Endpoint support is a representation capacity proxy over initial visible sampled points, not semantic grounding; raw-point/polyline error is not continuous collision validity')


def run(snapshot,output):
    # Keep pure diagnostic helpers importable without loading torch.
    from routeset.observed_geometry import backproject_rgbd
    from routeset.observed_route_head import resample_event_segments
    snapshot=Path(snapshot).resolve();output=Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve prior TRAIN representation audit')
    manifest=json.loads((snapshot/'snapshot_manifest.json').read_text())
    if manifest['protocol']!='multitask_closed_prefix_snapshot_v1':raise ValueError('Registered pilot snapshot required')
    for filename in ('observations.jsonl','supervision.jsonl','parent_inventory.json'):
        if digest(snapshot/filename)!=manifest['snapshot_files_sha256'][filename]:raise ValueError('Sealed snapshot file changed')
    requested=[row for row in manifest['selected_requested_parents'] if row['split']=='TRAIN']
    if len(requested)!=12 or any(row['parent_index'] not in (0,1) for row in requested):raise ValueError('Only the fixed twelve TRAIN parents may be audited')
    gate=audit([manifest['source_dataset']]+manifest['compare_sources'])
    tasks={row['task'] for row in requested}
    if any(gate['task_usage_status'].get(task,'').startswith('blocked_') for task in tasks):raise ValueError('Current mechanical layout gate blocks use')
    observations=[json.loads(line) for line in (snapshot/'observations.jsonl').read_text().splitlines() if line.strip()]
    labels={row['id']:row for row in (json.loads(line) for line in (snapshot/'supervision.jsonl').read_text().splitlines() if line.strip())}
    observed={}
    for row in observations:
        if row['split']=='TRAIN':observed.setdefault(row['parent_id'],[]).append(row)
    inventory={row['parent_id']:row for row in json.loads((snapshot/'parent_inventory.json').read_text()) if row['split']=='TRAIN'}
    source_hashes={str(snapshot/'snapshot_manifest.json'):digest(snapshot/'snapshot_manifest.json')}
    def verify(path):
        path=Path(path);value=digest(path)
        if value!=manifest['source_files_sha256'][str(path)]:raise ValueError('TRAIN source file changed: '+str(path))
        source_hashes[str(path)]=value;return path
    began=time.perf_counter();parents=[];routes=[]
    for spec in requested:
        parent=inventory[spec['parent_id']];inputs=observed.get(spec['parent_id'],[])
        record=dict(parent_id=spec['parent_id'],task=spec['task'],split='TRAIN',requested_attempts=3,
            setup_failed=parent['closure']['status']=='setup_failed',successful_references=parent['reference_count'],routes=[])
        if not inputs:
            record['status']='no_initial_observation';parents.append(record);continue
        first=inputs[0];label=labels[first['id']]
        if any(labels[row['id']]['routes']!=label['routes'] or labels[row['id']]['observation']!=label['observation'] or row['image']!=first['image'] for row in inputs):
            raise ValueError('TRAIN paraphrases do not share exactly one initial state/reference set')
        image=verify(first['image']);observation=verify(label['observation'])
        with Image.open(image) as handle:rgb=np.asarray(handle).copy()
        with np.load(observation,allow_pickle=False) as archive:
            cloud=backproject_rgbd(rgb,archive['depth'],archive['camera_intrinsics'],archive['camera_extrinsics'],pixel_stride=PIXEL_STRIDE)
        points=cloud['world_xyz'][cloud['valid_mask']]
        record.update(status='audited',observed_points=len(points),instruction_count=len(inputs))
        for route in label['routes']:
            path=verify(route)
            with np.load(path,allow_pickle=False) as archive:poses=archive['gripper_pose'].copy();opened=archive['gripper_open'].copy()
            row=dict(parent_id=spec['parent_id'],task=spec['task'],source_route=str(path),source_route_sha256=digest(path))
            try:
                sampled,events=resample_event_segments(poses,opened,HORIZON)
                row.update(status='audited',**route_diagnostics(poses,opened,sampled,events,points))
            except ValueError as error:
                row.update(status='representation_failed',error=str(error),raw_steps=len(poses))
            routes.append(row);record['routes'].append(row)
        parents.append(record)
    valid=[row for row in routes if row['status']=='audited']
    summary=dict(protocol='multitask_train_representation_v1',snapshot=str(snapshot),raw_roles_read=['TRAIN'],
        audit_script_sha256=digest(__file__),
        requested_train_parents=12,requested_attempts=36,raw_positive_references=len(routes),audited_references=len(valid),
        representation_failures=len(routes)-len(valid),setup_failed_parents=sum(row['setup_failed'] for row in parents),
        zero_reference_parents=sum(row['successful_references']==0 and not row['setup_failed'] for row in parents),
        event_sequences_preserved=sum(row['event_sequence_preserved'] for row in valid),
        unsupported_hard_anchor_endpoints=sum(not row['hard_surface_anchor_with_5cm_axis_residual_has_support'] for row in valid),
        horizon=HORIZON,pixel_stride=PIXEL_STRIDE,endpoint_axis_bound_m=ENDPOINT_AXIS_BOUND,
        support_scope='Hard selected observed-point anchor plus coordinatewise residual only; a soft weighted anchor can lie between points, so this is not its infeasibility certificate',
        elapsed_seconds=time.perf_counter()-began,current_layout_gate=gate,source_files_sha256=source_hashes,
        label_modifications=False,reference_filtering=False,dev_scale_selection=False,semantic_task_success=None,
        source_helpers={name:digest(Path(__file__).resolve().parents[1]/name) for name in ('routeset/observed_geometry.py','routeset/observed_route_head.py')})
    output.mkdir(parents=True)
    for name,value in [('summary',summary),('parents',parents),('routes',routes)]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({key:summary[key] for key in ('requested_train_parents','raw_positive_references','representation_failures','event_sequences_preserved','unsupported_hard_anchor_endpoints')}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.snapshot,args.output)
