"""TRAIN-only reference capacity/length audit, never a route filtering rule."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

from scripts import export_two_row_observations as exporter


def trajectory_statistics(xyz):
    xyz=np.asarray(xyz,dtype=float)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<2 or not np.isfinite(xyz).all():raise ValueError('Finite reference polyline required')
    return dict(points=len(xyz),length_m=float(np.linalg.norm(np.diff(xyz,axis=0),axis=1).sum()),
                minimum_world_z_m=float(xyz[:,2].min()),maximum_world_z_m=float(xyz[:,2].max()))


def endpoint_support(endpoint,world_xyz,valid_mask,bound=.05):
    points=np.asarray(world_xyz)[np.asarray(valid_mask,dtype=bool)]
    if not len(points):return dict(valid_points=0,nearest_L2_m=None,nearest_Linf_m=None,axis_bound_representable=False)
    difference=points-np.asarray(endpoint)
    linf=float(np.max(np.abs(difference),axis=1).min())
    return dict(valid_points=len(points),nearest_L2_m=float(np.linalg.norm(difference,axis=1).min()),
        nearest_Linf_m=linf,axis_bound_representable=linf<=bound)


def audit(data_root,output):
    # Torch is imported only for the existing CPU backprojection/resampler.
    from routeset.observed_geometry import backproject_rgbd
    from routeset.observed_route_head import resample_event_segments
    from scripts.evaluate_observed_two_row import scene_metrics
    data_root,output=Path(data_root),Path(output)
    if output.exists():raise FileExistsError('Fresh quality audit output required')
    started=time.perf_counter();m=json.loads((data_root/'export_manifest.json').read_text())
    if m['protocol']!=exporter.PROTOCOL:raise ValueError('Frozen formal prefix required')
    exporter.selection_guard(m['selection'])
    for name,value in m['output_files_sha256'].items():
        if exporter.digest(data_root/name)!=value:raise ValueError('Export metadata changed')
    gate=exporter.collector.live_layout_gate(Path(m['source_dataset']))
    selected={f'two_row_reach_{283200+i}' for i in range(16)}
    if selected&set(gate['blocked_parent_ids']):raise ValueError('TRAIN layout gate blocked')
    # Parse manifest rows to select roles, then open only selected TRAIN raw files.
    inputs={r['id']:r for r in [json.loads(x) for x in (data_root/'observations.jsonl').read_text().splitlines()]
            if r['split']=='TRAIN'}
    labels=[r for r in [json.loads(x) for x in (data_root/'supervision.jsonl').read_text().splitlines()] if r['split']=='TRAIN']
    if any(r['parent_id'] not in selected for r in labels) or set(inputs)!={r['id'] for r in labels}:raise ValueError('Fixed TRAIN identities required')
    hashes={};points={};references=[];conditions=[]
    def checked(path):
        path=Path(path);actual=exporter.digest(path)
        if m['source_files_sha256'].get(str(path))!=actual:raise ValueError('TRAIN source changed: '+str(path))
        hashes[str(path)]=actual;return path
    for label in labels:
        row=inputs[label['id']];parent=row['parent_id']
        if parent not in points:
            image=np.asarray(Image.open(checked(row['image'])).convert('RGB'))
            with np.load(checked(label['observation']),allow_pickle=False) as archive:
                current={key:archive[key].copy() for key in archive.files}
            cloud=backproject_rgbd(image,current['depth'],current['camera_intrinsics'],current['camera_extrinsics'],pixel_stride=2)
            points[parent]=(cloud,current)
        cloud,current=points[parent]
        with np.load(checked(label['verification_only']),allow_pickle=False) as archive:
            geometry={key:archive[key] for key in ('obstacle_centers','obstacle_halfsizes')}
        cfg=json.loads(checked(label['route_config']).read_text())
        conditions.append(dict(id=label['id'],parent_id=parent,positive_references=len(label['routes']),unknown_types=sum(t is None for t in label['route_types'])))
        for filename in label['routes']:
            with np.load(checked(filename),allow_pickle=False) as archive:
                pose,events=archive['gripper_pose'],archive['gripper_open']
                h24,h24_events=resample_event_segments(pose,events,24)
            metric,candidates=scene_metrics(h24[None],h24_events[None],current,geometry,label['semantic_targets'],label['route_types'],cfg)
            references.append(dict(id=label['id'],parent_id=parent,route=filename,raw=trajectory_statistics(pose[:,:3]),
                model_H24=trajectory_statistics(h24),raw_event_transitions=int(np.count_nonzero(np.diff(events>.5))),
                h24_event_transitions=int(np.count_nonzero(np.diff(h24_events>.5))),
                endpoint_support=endpoint_support(pose[-1,:3],cloud['world_xyz'],cloud['valid_mask']),
                model_H24_tip_valid=metric['TipValidAtK']==1,model_H24_type=candidates[0]['declared_passage_type']))
    if any(exporter.digest(path)!=value for path,value in hashes.items()):raise ValueError('TRAIN sources changed during audit')
    errors=[r for r in references if not r['endpoint_support']['axis_bound_representable']]
    passed=bool(references) and not errors
    result=dict(protocol='two_row_train_endpoint_capacity_and_reference_quality_v1',registered_train_parents=16,
        source_export_manifest_sha256=exporter.digest(data_root/'export_manifest.json'),
        train_input_ids=sorted(inputs),train_reference_counts={r['id']:len(r['routes']) for r in labels},
        observed_train_parents=len(points),observed_train_inputs=len(labels),positive_references=len(references),
        conditions=conditions,references=references,axis_bound_m=.05,pixel_stride=2,horizon=24,
        capacity_gate_passed=passed,capacity_failures=[dict(id=r['id'],route=r['route'],support=r['endpoint_support']) for r in errors],
        model_h24_tip_valid_references=sum(r['model_H24_tip_valid'] for r in references),
        source_files_sha256=hashes,audit_source_sha256=exporter.digest(__file__),elapsed_seconds=time.perf_counter()-started,
        data_filtering=False,dev_raw_arrays_opened=False,locked_raw_opened=False,
        decision='Proceed with predeclared surface-peak capacity only if gate passes; failures require explicit TRAIN-only representation review, never deleting references.',
        limitation='Observed-point support establishes endpoint representation capacity only, not learnability, target identity, free space or robot execution.')
    output.parent.mkdir(parents=True,exist_ok=True);exporter.write_json(output,result)
    print(json.dumps({k:result[k] for k in ('positive_references','capacity_gate_passed','model_h24_tip_valid_references','elapsed_seconds')}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();audit(a.data_root,a.output)
