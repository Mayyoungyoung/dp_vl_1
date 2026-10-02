"""After freezing all eight predictions, inspect format and target endpoints only."""
import argparse
import json
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json
from routeset.vlm_sft_evaluation import safe_parse_paths
from routeset.vlm_sft_train_probe import PARENTS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--supervision-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh post-generation diagnostic output required')
    generated=json.loads((args.generation/'summary.json').read_text())
    if (generated['status']!='completed' or generated['evaluation_scope']!='train8_preflight'
            or generated['split']!='TRAIN' or generated['examples']!=8 or generated['repeats']!=1 or generated['seed']!=0):
        raise ValueError('Complete exact TRAIN8 probe required before reading evaluation labels')
    if generated['checkpoint_sha256']!='675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f':
        raise ValueError('Original actual SFT best checkpoint required')
    for name, receipt in generated['artifacts'].items():
        path=(args.generation/name).resolve()
        if args.generation.resolve() not in path.parents or sha256(path)!=receipt['sha256']:
            raise ValueError('Frozen prediction pool artifact changed')
    requests=[json.loads(line) for line in (args.generation/'requests.jsonl').read_text().splitlines()]
    if len(requests)!=8 or [r['parent_id'] for r in requests]!=PARENTS:
        raise ValueError('All eight exact requests must finish before label access')
    paths,events=[],[]
    for request in requests:
        if (request['scene_id']!=request['parent_id']+'_target0' or request['split']!='TRAIN'
                or request['k']!=1 or request['horizon']!=24 or request['max_new_tokens']!=512):
            raise ValueError('Predeclared request identity/count changed')
        xyz,opened,receipt=safe_parse_paths(request['text'],1,24)
        if receipt!=request['parse']: raise ValueError('Original strict parser receipt changed')
        paths.append(xyz);events.append(opened)
    paths,events=np.stack(paths),np.stack(events)
    with np.load(args.generation/'predictions.npz',allow_pickle=False) as saved:
        if (saved['scene_ids'].tolist()!=[r['scene_id'] for r in requests]
                or not np.array_equal(saved['paths'],paths,equal_nan=True)
                or not np.array_equal(saved['gripper_open'],events,equal_nan=True)):
            raise ValueError('Predictions differ from original unmodified raw text')
    # This is the first access to any evaluation-only target metadata.
    if sha256(args.supervision_metadata)!='37da38855af0357732ac48f125f140e8ed5d1a60008f1424fd5080c7a9421237':
        raise ValueError('Original source supervision metadata required')
    wanted={r['scene_id'] for r in requests}
    labels={row['id']:row for row in map(json.loads,args.supervision_metadata.read_text().splitlines()) if row['id'] in wanted}
    if set(labels)!=wanted: raise ValueError('Missing exact TRAIN evaluation target identity')
    records=[]
    for i,request in enumerate(requests):
        label=labels[request['scene_id']]
        if label['split']!='TRAIN' or label['parent_id']!=request['parent_id']:
            raise ValueError('Exact TRAIN target join mismatch')
        specification=label['semantic_targets']
        if specification['tolerance']!=.03: raise ValueError('Unchanged 3cm endpoint rule required')
        finite=bool(np.isfinite(paths[i]).all() and np.isfinite(events[i]).all())
        endpoint=error=nearest=None; within=correct=False
        if finite:
            endpoint=paths[i,0,-1].astype(np.float64)
            distances=np.linalg.norm(np.asarray(specification['centers'],dtype=np.float64)-endpoint,axis=-1)
            error=float(distances[specification['target_index']]);nearest=int(distances.argmin())
            within=error<=.03;correct=within and nearest==specification['target_index']
        records.append(dict(scene_id=request['scene_id'],parent_id=request['parent_id'],split='TRAIN',
            strict_h24_format_finite=finite,endpoint_m=None if endpoint is None else endpoint.tolist(),
            requested_target_distance_m=error,nearest_target_index=nearest,within_3cm=within,
            requested_target_nearest_and_within_3cm=correct,
            raw_parse=request['parse'],tokens=request['tokens'],failure=request['failure']))
    errors=[r['requested_target_distance_m'] for r in records if r['requested_target_distance_m'] is not None]
    summary=dict(protocol='vlm_constrained_train8_endpoint_probe_v1',split='TRAIN',status='completed',
        generation_source_commit=generated['code_commit'],generation_summary_sha256=sha256(args.generation/'summary.json'),
        requests_sha256=sha256(args.generation/'requests.jsonl'),supervision_metadata_sha256=sha256(args.supervision_metadata),
        analyzer_sha256=sha256(__file__),checkpoint_sha256=generated['checkpoint_sha256'],
        requested_calls=8,requested_slots=8,strict_h24_format_finite=sum(r['strict_h24_format_finite'] for r in records),
        within_3cm_count=sum(r['within_3cm'] for r in records),
        correct_requested_target_count=sum(r['requested_target_nearest_and_within_3cm'] for r in records),
        endpoint_error_m=dict(count=len(errors),min=float(min(errors)),median=float(np.median(errors)),mean=float(np.mean(errors)),max=float(max(errors))) if errors else dict(count=0),
        generation_elapsed_seconds=generated['elapsed_seconds'],gpu_hours_reserved=generated['gpu_hours_reserved'],
        peak_cuda_allocated_bytes=generated['peak_cuda_allocated_bytes'],
        additional_model_calls=0,raw_reference_paths_opened=0,verification_geometry_opened=0,repaired_outputs=0,
        scope='Training-capacity/endpoint diagnosis only. Unchanged H24 parser; no collision/route validity or DEV/generalization result.',records=records)
    args.output.mkdir(parents=True);write_json(args.output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('strict_h24_format_finite','within_3cm_count','correct_requested_target_count','endpoint_error_m','generation_elapsed_seconds')},indent=2))


if __name__=='__main__':main()
