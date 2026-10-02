"""Evaluate the already-fitted old64 prototype on all reserved fresh DEV inputs.

This entry never fits prototypes. It emits at most one observed endpoint per
instruction; no route, learned Qwen representation or privileged test geometry.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np
from scripts.observation_prototype_grounding import (CONFIG,INPUT_KEYS,digest,read_rows,
    load_observation,predict,evaluate_endpoint)


OLD64_REPORT_SHA='9c7d0eb48c449ec64e1d5165bcf852c44ad38d26f3305455bd0a8bf19de7b7f9'
FRESH_INPUT_SHA='aa17ecef147f73ada2e902ac30f7de39be0acc47a25ced5959607619bf945495'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-report',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--reservation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('fresh output required')
    started_total=time.perf_counter()
    model_source_hash=digest(args.model_report)
    if model_source_hash!=OLD64_REPORT_SHA:raise ValueError('must use the unchanged old64 fitted prototype report')
    source=json.loads(args.model_report.read_text());model=source['model']
    if model['config']!=CONFIG or source['config']!=CONFIG:raise ValueError('prototype hyperparameters differ from frozen source')
    implementation=Path(__file__).with_name('observation_prototype_grounding.py')
    if digest(implementation)!=source['script_sha256']:raise ValueError('prototype implementation differs from original fitted baseline')
    train_parents={row['id'].rsplit('_target',1)[0] for row in model['training_rows']}
    if train_parents!={'derived_reach_'+str(seed) for seed in range(262000,262064)}:raise ValueError('prototype was not fit on exactly old64 TRAIN')
    if model['training_examples']!=192 or model['training_parents']!=64:raise ValueError('unexpected source TRAIN exposure')
    manifest=json.loads((args.data/'export_manifest.json').read_text())
    reservation=json.loads(args.reservation.read_text())
    entry=next(row for row in reservation['collections'] if row['source']=='data/observation_reach_fast256_20261002')
    first,last=entry['roles_inclusive']['DEV_MODEL']
    dev_parents={entry['parent_prefix']+('%06d'%seed) for seed in range(first,last+1)}
    if manifest['reservation_sha256']!=digest(args.reservation):raise ValueError('reservation changed')
    if set(manifest['requested_roles'])!={'TRAIN','DEV_MODEL'}:raise ValueError('development-only export required')
    if digest(args.data/'observations.jsonl')!=FRESH_INPUT_SHA:raise ValueError('fixed fresh observation manifest required')
    if digest(args.data/'supervision.jsonl')!=manifest['supervision_manifest_sha256']:raise ValueError('fresh supervision changed')
    observations=read_rows(args.data/'observations.jsonl')
    if any(set(row)!=INPUT_KEYS for row in observations):raise ValueError('strict observation whitelist required')
    labels={row['id']:row for row in read_rows(args.data/'supervision.jsonl')}
    dev=[row for row in observations if row['split']=='DEV_MODEL']
    if len(dev)!=48 or {row['parent_id'] for row in dev}!=dev_parents or train_parents&dev_parents:
        raise ValueError('exact fresh 16 parents/48 instructions, disjoint from old64 TRAIN required')
    predictions,rows,hashes={},[],{}
    for row in dev:
        label=labels[row['id']]
        if label['split']!=row['split'] or label['parent_id']!=row['parent_id']:raise ValueError('supervision join mismatch')
        started=time.perf_counter()
        (rgb,xyz,valid),paths=load_observation(args.data,row,label['observation'])
        endpoint,details=predict(rgb,xyz,valid,row['instruction'],model)
        latency=time.perf_counter()-started
        predictions[row['id']]=endpoint
        hashes.update({str(path):digest(path) for path in paths})
        # Test labels enter only after the complete prediction and timing.
        if label['semantic_targets']['tolerance']!=.03:raise ValueError('original 3cm criterion required')
        metrics=evaluate_endpoint(endpoint,label['semantic_targets'])
        references=[]
        for filename in label.get('routes',[]):
            path=args.data/filename
            with np.load(path,allow_pickle=False) as archive:references.append(archive['gripper_pose'][-1,:3])
            hashes[str(path)]=digest(path)
        rows.append(dict(id=row['id'],parent_id=row['parent_id'],instruction=row['instruction'],
            prediction_endpoint=None if endpoint is None else endpoint.tolist(),latency_seconds=latency,
            reference_count=len(references),reference_endpoint_error_m=float(np.linalg.norm(np.asarray(references)-endpoint,axis=-1).min()) if references and endpoint is not None else None,
            **details,**metrics))
    groups=defaultdict(list)
    for row in dev:groups[row['parent_id']].append(row)
    swaps=[]
    for parent,group in groups.items():
        if len({digest(args.data/row['image']) for row in group})!=1:raise ValueError('language swap images differ')
        if len({labels[row['id']]['observation'] for row in group})!=1:raise ValueError('language swap current observations differ')
        for before in group:
            for after in group:
                if before['id']==after['id']:continue
                first_endpoint,second_endpoint=predictions[before['id']],predictions[after['id']]
                swaps.append(dict(parent_id=parent,original_id=before['id'],changed_instruction_id=after['id'],
                    endpoint_response_m=float(np.linalg.norm(first_endpoint-second_endpoint)) if first_endpoint is not None and second_endpoint is not None else None,
                    changed_instruction_semantic_accuracy=evaluate_endpoint(second_endpoint,labels[after['id']]['semantic_targets'])['semantic_goal_accuracy'],
                    stale_instruction_semantic_accuracy=evaluate_endpoint(first_endpoint,labels[after['id']]['semantic_targets'])['semantic_goal_accuracy']))
    if digest(args.model_report)!=model_source_hash:raise RuntimeError('fitted model source changed')
    errors=[row['goal_error_m'] for row in rows if row['goal_error_m'] is not None]
    reference_errors=[row['reference_endpoint_error_m'] for row in rows if row['reference_endpoint_error_m'] is not None]
    latency=np.asarray([row['latency_seconds'] for row in rows])
    report=dict(evaluation_protocol='observation_eval_v2',checkpoint_selection_protocol='fixed_old64_prototype_fresh_dev_transfer',
        baseline='TRAIN endpoint-supervised color prototype RGB-D localization',scope='closed exact instruction endpoint only; not a route baseline or open vocabulary',
        source_model_report=str(args.model_report.resolve()),source_model_report_sha256=model_source_hash,source_model_unchanged=True,
        source_training_parents=64,source_training_examples=192,source_training_manifests=source['manifest_sha256'],source_training_seconds=model['training_seconds'],
        additional_training_examples=0,config=CONFIG,source_commit=os.environ.get('CODE_COMMIT'),script_sha256=digest(__file__),
        prototype_implementation_sha256=digest(implementation),
        data=str(args.data.resolve()),reservation_sha256=digest(args.reservation),export_manifest_sha256=digest(args.data/'export_manifest.json'),
        manifest_sha256={name:digest(args.data/name) for name in ('observations.jsonl','supervision.jsonl')},
        examples=48,parents=16,semantic_evaluation_examples=48,reference_evaluation_examples=sum(row['reference_count']>0 for row in rows),
        candidate_budget=1,candidate_budget_scope='one endpoint, including abstentions; no route generated',
        internal_components_are='observed color regions; all counts and processing included, not route candidates',
        semantic_goal_accuracy=float(np.mean([row['semantic_goal_accuracy'] for row in rows])),
        unknown_instruction_count=sum(row['status']=='unsupported_exact_instruction' for row in rows),
        abstention_count=sum(row['candidates_emitted']==0 for row in rows),
        goal_error_m_conditional_on_prediction=float(np.mean(errors)) if errors else None,
        reference_endpoint_error_m_conditional_on_prediction=float(np.mean(reference_errors)) if reference_errors else None,
        cpu_latency_ms=dict(median=float(np.median(latency)*1000),p95=float(np.quantile(latency,.95)*1000),first=float(latency[0]*1000),
            includes='RGB/depth load, backprojection, all color components, component scoring and endpoint; no Qwen/route/execution'),
        total_wall_seconds=time.perf_counter()-started_total,
        language_swap=dict(directed_pairs=len(swaps),changed_instruction_semantic_accuracy=float(np.mean([row['changed_instruction_semantic_accuracy'] for row in swaps])),
            stale_instruction_semantic_accuracy=float(np.mean([row['stale_instruction_semantic_accuracy'] for row in swaps])),per_pair=swaps),
        evaluation_source_sha256=hashes,per_scene=rows,locked_test_access=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('examples','semantic_goal_accuracy','unknown_instruction_count','abstention_count',
        'goal_error_m_conditional_on_prediction','reference_endpoint_error_m_conditional_on_prediction','cpu_latency_ms','total_wall_seconds')}),flush=True)


if __name__=='__main__':main()
