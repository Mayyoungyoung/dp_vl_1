"""TRAIN-only support audit for recorded interaction locations; no model changes.

Only first open-to-close events and reference endpoints are measured. The
initial RGB-D cloud is unlabelled: proximity cannot identify the correct object,
certify visibility, contact, collision clearance, or task execution.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


RADII=(.01,.025,.05,.10)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def support(point,world_points):
    point,world_points=np.asarray(point,dtype=float),np.asarray(world_points,dtype=float)
    if point.shape!=(3,) or world_points.ndim!=2 or world_points.shape[1]!=3 or not len(world_points):
        raise ValueError('A finite point and nonempty observed XYZ cloud are required')
    if not np.isfinite(point).all() or not np.isfinite(world_points).all():raise ValueError('Nonfinite support input')
    displacement=world_points-point;distance=np.linalg.norm(displacement,axis=1)
    return dict(xyz=point.tolist(),nearest_observed_l2_m=float(distance.min()),
        nearest_observed_linf_m=float(np.abs(displacement).max(1).min()),
        observed_point_counts={str(radius):int((distance<=radius).sum()) for radius in RADII})


def event_locations(xyz,opened,points):
    xyz,opened=np.asarray(xyz,dtype=float),np.asarray(opened,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or not len(xyz) or opened.shape!=(len(xyz),):raise ValueError('Invalid route')
    if not np.isfinite(xyz).all() or not np.isfinite(opened).all():raise ValueError('Nonfinite route')
    states=opened>.5;transitions=np.flatnonzero(states[1:]!=states[:-1])+1
    closes=np.flatnonzero(states[:-1]&~states[1:])+1
    result=dict(steps=len(xyz),event_sequence=states[np.r_[0,transitions]].tolist(),
        transitions=transitions.tolist(),close_indices=closes.tolist(),endpoint=support(xyz[-1],points),
        first_close_before=None,first_close_after=None,primary_landmark=None)
    if len(closes):
        index=int(closes[0]);result.update(category='first_open_to_close',first_close_index=index,
            first_close_before=support(xyz[index-1],points),first_close_after=support(xyz[index],points),
            close_boundary_displacement_m=float(np.linalg.norm(xyz[index]-xyz[index-1])),primary_landmark='first_close_after')
    elif not len(transitions):
        result.update(category='constant_open' if states[0] else 'constant_closed',primary_landmark='endpoint')
    else:
        result.update(category='transitions_without_open_to_close',primary_landmark=None)
    return result


def selected_train(manifest):
    if manifest.get('protocol')!='multitask_closed_prefix_snapshot_v1':raise ValueError('Sealed multitask snapshot required')
    selected=[row for row in manifest['selected_requested_parents'] if row['split']=='TRAIN']
    from scripts.observation_collect_multitask import registration
    expected=[r for r in registration(282000,'validated-five-v1','interleaved-early-dev-v1')['parents']
              if r['split']=='TRAIN' and r['parent_index']<8]
    if selected!=expected:raise ValueError('Only the fixed prefix60 forty-eight TRAIN parents may be audited')
    return selected


def aggregate(rows,representation,landmark):
    values=[]
    for row in rows:
        entry=row.get(representation)
        if entry is None:continue
        key=entry['primary_landmark'] if landmark=='primary' else landmark
        if key is not None and entry.get(key) is not None:values.append((row['parent_id'],entry[key]))
    if not values:return dict(references=0,parents=0,nearest_l2_m_mean=None)
    distances=np.array([value['nearest_observed_l2_m'] for _,value in values])
    parents=sorted({parent for parent,_ in values})
    return dict(references=len(values),parents=len(parents),nearest_l2_m_mean=float(distances.mean()),
        nearest_l2_m_median=float(np.median(distances)),nearest_l2_m_p95=float(np.percentile(distances,95)),
        nearest_l2_m_max=float(distances.max()),
        parent_mean_nearest_l2_m=float(np.mean([np.mean([value['nearest_observed_l2_m'] for p,value in values if p==parent]) for parent in parents])),
        references_with_support={str(r):int((distances<=r).sum()) for r in RADII},
        parent_mean_support_fraction={str(r):float(np.mean([np.mean([value['nearest_observed_l2_m']<=r for p,value in values if p==parent]) for parent in parents])) for r in RADII})


def run(snapshot,output):
    from routeset.observed_geometry import backproject_rgbd
    from routeset.observed_route_head import resample_event_segments
    from routeset.observed_multitask import check_multitask_model_gate
    snapshot,output=Path(snapshot).resolve(),Path(output).resolve()
    if output.exists():raise FileExistsError('Preserve previous audit output')
    manifest_path=snapshot/'snapshot_manifest.json';manifest=json.loads(manifest_path.read_text())
    requested=selected_train(manifest)
    gate=check_multitask_model_gate(snapshot/'observations.jsonl',snapshot/'supervision.jsonl',manifest_path)
    if gate is None:raise ValueError('Mandatory live mechanical gate missing')
    observations=[json.loads(line) for line in (snapshot/'observations.jsonl').read_text().splitlines()]
    labels={row['id']:row for row in map(json.loads,(snapshot/'supervision.jsonl').read_text().splitlines())}
    inventory={row['parent_id']:row for row in json.loads((snapshot/'parent_inventory.json').read_text()) if row['split']=='TRAIN'}
    observed={}
    for row in observations:
        if row['split']=='TRAIN':observed.setdefault(row['parent_id'],[]).append(row)
    hashes={str(manifest_path):digest(manifest_path)};routes=[];parents=[];started=time.perf_counter()
    def verify(path,parent_id):
        path=Path(path).resolve();allowed=Path(manifest['source_dataset']).resolve()/'TRAIN/parents'/parent_id
        try:path.relative_to(allowed)
        except ValueError:raise ValueError('Attempted non-TRAIN or foreign-parent raw read')
        value=digest(path)
        if manifest['source_files_sha256'].get(str(path))!=value:raise ValueError('TRAIN source changed after snapshot')
        hashes[str(path)]=value;return path
    for spec in requested:
        identifier=spec['parent_id'];record=inventory[identifier];inputs=observed.get(identifier,[])
        parent=dict(parent_id=identifier,task=spec['task'],requested_attempts=3,reference_count=record['reference_count'],
            setup_failed=record['closure']['status']=='setup_failed',observed=bool(inputs));parents.append(parent)
        if not inputs:continue
        first=inputs[0];label=labels[first['id']]
        if any(labels[row['id']]['routes']!=label['routes'] or labels[row['id']]['observation']!=label['observation'] or row['image']!=first['image'] for row in inputs):
            raise ValueError('Paraphrases changed initial state or positive reference set')
        image=verify(first['image'],identifier);observation=verify(label['observation'],identifier)
        with Image.open(image) as im:rgb=np.asarray(im).copy()
        with np.load(observation,allow_pickle=False) as archive:
            cloud=backproject_rgbd(rgb,archive['depth'],archive['camera_intrinsics'],archive['camera_extrinsics'],pixel_stride=2)
        points=cloud['world_xyz'][cloud['valid_mask']];parent['observed_points']=len(points)
        for route_path in label['routes']:
            route=verify(route_path,identifier)
            with np.load(route,allow_pickle=False) as archive:pose=archive['gripper_pose'].copy();opened=archive['gripper_open'].copy()
            row=dict(parent_id=identifier,task=spec['task'],route=str(route),route_sha256=hashes[str(route)])
            row['raw']=event_locations(pose[:,:3],opened,points)
            try:
                xyz,event=resample_event_segments(pose,opened,24);row['H24']=event_locations(xyz,event,points)
                row['event_sequence_preserved']=row['raw']['event_sequence']==row['H24']['event_sequence']
                row['first_close_position_difference_m']=(float(np.linalg.norm(np.array(row['raw']['first_close_after']['xyz'])-row['H24']['first_close_after']['xyz']))
                    if row['raw']['first_close_after'] is not None and row['H24']['first_close_after'] is not None else None)
                row['resampling_status']='passed'
            except ValueError as exc:row.update(resampling_status='failed',error=str(exc),H24=None)
            routes.append(row)
    metrics=lambda subset:{rep:{key:aggregate(subset,rep,key) for key in ('primary','first_close_before','first_close_after','endpoint')} for rep in ('raw','H24')}
    summary=dict(protocol='multitask_train_event_surface_support_v1',snapshot=str(snapshot),source_script_sha256=digest(__file__),
        requested_train_parents=len(requested),requested_attempts=3*len(requested),raw_reference_count=len(routes),
        setup_failed_parents=sum(p['setup_failed'] for p in parents),zero_reference_parents=sum(p['reference_count']==0 and not p['setup_failed'] for p in parents),
        resampling_failures=sum(r['resampling_status']=='failed' for r in routes),event_sequences_preserved=sum(r.get('event_sequence_preserved',False) for r in routes),
        horizon=24,pixel_stride=2,radii_m=RADII,raw_roles_read=['TRAIN'],
        raw_categories={category:sum(r['raw']['category']==category for r in routes) for category in ('first_open_to_close','constant_open','constant_closed','transitions_without_open_to_close')},
        overall=metrics(routes),per_task={task:metrics([r for r in routes if r['task']==task]) for task in sorted({p['task'] for p in requested})},
        elapsed_seconds=time.perf_counter()-started,current_layout_gate=gate,source_files_sha256=hashes,
        source_helpers={name:digest(Path(__file__).resolve().parents[1]/name) for name in ('routeset/observed_geometry.py','routeset/observed_route_head.py','routeset/observed_multitask.py')},
        scope='Nearest initial unlabelled observed point, including robot/background; proximity alone does not identify the correct object or prove visibility/contact. No mask/true geometry read. No model, label mutation, DEV threshold selection or task-ID-conditioned input.')
    output.mkdir(parents=True)
    for name,value in [('summary',summary),('parents',parents),('routes',routes)]:
        (output/(name+'.json')).write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({key:summary[key] for key in ('requested_train_parents','raw_reference_count','resampling_failures','event_sequences_preserved','raw_categories')}))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.snapshot,args.output)
