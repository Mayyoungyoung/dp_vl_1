"""CPU-only fixed-protocol checks after all VLM candidates have been emitted.

All 24 DEV instructions remain, including zero-reference, format failures and
over-budget scenes. Repeats are evaluated separately, then averaged; never
combine their candidates. No refit, reranking, replacement or route repair.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

from routeset.common import sha256,write_json
from routeset.vlm_sft_data import validate_reserved_observation_manifest
from routeset.vlm_sft_evaluation import EVALUATION_PROTOCOL,METHODS,request_seed,safe_parse_paths
# Historical geometry helper uses sibling-script imports. Resolve them here
# without modifying its already-run implementation or evaluation semantics.
sys.path.insert(0,str(Path(__file__).resolve().parent))
from scripts.evaluate_observed_obstacles import PROTOCOL,scene_metrics


FIELDS=('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','KnownReferenceTypeCoverageAtK',
        'DuplicateClassifiedTipValidCount','UnknownTypeTipValidCount','semantic_goal_accuracy',
        'AnySemanticGoalAtK','TipClearAtK','StartCorrectAtK','EventSequenceCorrectAtK','endpoint_error_m',
        'mean_path_length_m','SelectedTipValidAtK','ValidAtK','AnyValidAtK','UniqueValidAtK','SelectedValidAtK')


def average(rows,fields=FIELDS):
    return {key:float(np.mean([row[key] for row in rows if row[key] is not None]))
            if any(row[key] is not None for row in rows) else None for key in fields}


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def validate_source_index(source_hashes,expected):
    if hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()!=expected:
        raise ValueError('Original training data source index changed')


def verify_exported_file(path,parent,exported):
    record=exported['source_parents'][parent]
    digest=sha256(path)
    if record['reserved_role']!='DEV_MODEL' or record['source_files_sha256'].get(str(path))!=digest:
        raise ValueError('DEV evaluation artifact changed since its original reserved export')
    return digest


def load_group(folder,generated,method,repeat,ids):
    records=[json.loads(line) for line in (folder/'requests.jsonl').read_text().splitlines()]
    rows=read(folder/'per_scene.json')
    if [r['scene_id'] for r in rows]!=ids:raise ValueError('Generation scene order/identity changed')
    with np.load(folder/'predictions.npz',allow_pickle=False) as saved:
        if list(map(str,saved['scene_ids']))!=ids:raise ValueError('Prediction identity mismatch')
        paths,events=saved['paths'],saved['gripper_open']
        parents=list(map(str,saved['parent_ids']))
        charged=saved['charged_candidate_slots'].copy()
    if paths.shape!=(len(ids),4,24,3) or events.shape!=(len(ids),4,24):raise ValueError('K4/H24 shape mismatch')
    expected_calls=4 if method=='independent4' else 1
    if len(records)!=len(ids)*expected_calls:raise ValueError('Missing/extra generation requests')
    for index,identifier in enumerate(ids):
        calls=records[index*expected_calls:(index+1)*expected_calls];pp=[];ee=[];budget=0
        for number,call in enumerate(calls):
            k=1 if method=='independent4' else 4
            if (call['scene_id']!=identifier or call['parent_id']!=parents[index] or call['method']!=method or
                    call['repeat']!=repeat or call['request']!=number or call['k']!=k or call['max_new_tokens']!=512*k or
                    call['decoding_seed']!=request_seed(generated['seed'],repeat,identifier,method,number)):
                raise ValueError('Declared finite request budget/seed changed')
            path,event,receipt=safe_parse_paths(call['text'],k,24)
            if receipt!=call['parse']:raise ValueError('Saved parse differs from strict raw-text parse')
            pp.append(path);ee.append(event);budget+=receipt['charged_candidate_slots']
        p,e=np.concatenate(pp),np.concatenate(ee)
        if budget>4:p[:]=np.nan;e[:]=np.nan
        if budget!=charged[index] or budget!=rows[index]['charged_candidate_slots']:
            raise ValueError('Charged candidate accounting changed')
        if not np.array_equal(p,paths[index],equal_nan=True) or not np.array_equal(e,events[index],equal_nan=True):
            raise ValueError('Saved predictions differ from raw strict parse; no repair/selection allowed')
    return paths,events,parents,rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();started=time.perf_counter()
    if args.output.exists():raise FileExistsError('Fresh post-generation analysis output required')
    generated=read(args.generation/'summary.json')
    if generated['protocol']!=EVALUATION_PROTOCOL or generated['status']!='completed':
        raise ValueError('Complete fixed-protocol generation required before labels open')
    for name,record in generated['artifacts'].items():
        path=(args.generation/name).resolve()
        try:
            path.relative_to(args.generation.resolve())
        except ValueError:
            raise ValueError('Generation artifact escaped its output')
        if sha256(path)!=record['sha256']:
            raise ValueError('Generation artifact changed or path escaped its output')
    config=generated['config'];data=Path(config['observations']).parent
    observations=validate_reserved_observation_manifest(data/'observations.jsonl')
    selected=sorted([r for r in observations if r['split']=='DEV_MODEL'],key=lambda r:r['id'])
    ids=[r['id'] for r in selected]
    if len(ids)!=24 or len({r['parent_id'] for r in selected})!=8:raise ValueError('Full registered DEV required')
    source_hashes=read(Path(generated['checkpoint']).parent/'data_source_hashes.json')
    validate_source_index(source_hashes,config['data_fingerprint'])
    for path in (data/'observations.jsonl',data/'supervision.jsonl'):
        if source_hashes.get(str(path.resolve()))!=sha256(path):raise ValueError('Training/evaluation manifest changed')
    # Validate every fixed candidate pool before opening verification labels.
    groups={}
    for repeat in range(generated['repeats']):
        for method in METHODS:
            groups[(repeat,method)]=load_group(args.generation/f'repeat{repeat}'/method,generated,method,repeat,ids)
    labels={row['id']:row for row in map(json.loads,(data/'supervision.jsonl').read_text().splitlines()) if row['split']=='DEV_MODEL'}
    exported=read(data/'export_manifest.json')
    if sha256(data/'export_manifest.json')!=generated['export_manifest_sha256']:
        raise ValueError('Export manifest changed after generation')
    collection=read(Path(exported['source_dataset'])/'manifest.json')
    if collection['acceptance']['tip_polyline_clearance_m']!=.02:raise ValueError('Original 2cm clearance required')
    if set(labels)!=set(ids):raise ValueError('Complete DEV label identity join required')
    args.output.mkdir(parents=True)
    all_results=[];case_map={};input_hashes={str(path):sha256(path) for path in
        (data/'export_manifest.json',data/'observations.jsonl',data/'supervision.jsonl',Path(exported['source_dataset'])/'manifest.json')}
    for (repeat,method),(paths,events,parents,timing) in groups.items():
        per_scene=[];per_candidate=[]
        for index,row in enumerate(selected):
            tic=time.perf_counter();label=labels[row['id']]
            if parents[index]!=row['parent_id'] or label['parent_id']!=row['parent_id'] or label['semantic_targets']['tolerance']!=.03:
                raise ValueError('Fixed parent/semantic protocol mismatch')
            observed=(data/label['observation']).resolve();verification=(data/label['verification_only']).resolve()
            for path in (observed,verification):
                if path.parent.name!=row['parent_id']:raise ValueError('Evaluation-only artifact outside its DEV parent')
                input_hashes[str(path)]=verify_exported_file(path,row['parent_id'],exported)
            if source_hashes.get(str(observed))!=input_hashes[str(observed)]:raise ValueError('Current state changed since training')
            with np.load(observed,allow_pickle=False) as saved:
                current={key:saved[key].astype(np.float32) for key in ('gripper_pose','gripper_open')}
            with np.load(verification,allow_pickle=False) as saved:
                geometry={key:saved[key] for key in ('obstacle_centers','obstacle_halfsizes')}
            result,candidates=scene_metrics(paths[index],events[index],current,geometry,
                label['semantic_targets'],label.get('route_types',[]),clearance=.02)
            check_seconds=time.perf_counter()-tic
            result.update(scene_id=row['id'],parent_id=row['parent_id'],reference_count=len(label['routes']),
                charged_candidate_slots=timing[index]['charged_candidate_slots'],budget_exceeded=timing[index]['budget_exceeded'],
                generation_seconds=timing[index]['observed_input_to_routes_seconds'],checker_seconds=check_seconds,
                generation_plus_checker_seconds=timing[index]['observed_input_to_routes_seconds']+check_seconds)
            per_scene.append(result)
            per_candidate.extend(dict(scene_id=row['id'],parent_id=row['parent_id'],**c) for c in candidates)
        metrics=average(per_scene)
        metrics.update(method=method,repeat=repeat,examples=24,parents=8,requested_candidate_slots=96,
            charged_candidate_slots=sum(r['charged_candidate_slots'] for r in per_scene),
            budget_exceeded_examples=sum(r['budget_exceeded'] for r in per_scene),
            examples_without_reference=sum(not r['reference_count'] for r in per_scene),
            format_or_budget_failure_slots=sum(r['format_failure_count'] for r in per_scene),
            total_tip_valid_candidates=sum(c['TipValid'] for c in per_candidate),
            reference_coverage_evaluable_examples=sum(bool(r['known_reference_types']) for r in per_scene),
            generation_ms_median=float(np.median([r['generation_seconds'] for r in per_scene])*1000),
            generation_plus_checker_ms_median=float(np.median([r['generation_plus_checker_seconds'] for r in per_scene])*1000),
            generation_plus_checker_ms_p95=float(np.percentile([r['generation_plus_checker_seconds'] for r in per_scene],95)*1000))
        destination=args.output/f'repeat{repeat}'/method;destination.mkdir(parents=True)
        write_json(destination/'metrics.json',metrics);write_json(destination/'per_scene.json',per_scene)
        write_json(destination/'per_candidate.json',per_candidate)
        all_results.append(metrics);case_map[(repeat,method)]=per_scene
    paired=[]
    for repeat in range(generated['repeats']):
        for parent in sorted({r['parent_id'] for r in selected}):
            values={method:average([r for r in case_map[(repeat,method)] if r['parent_id']==parent]) for method in METHODS}
            paired.append(dict(repeat=repeat,parent_id=parent,
                whole_minus_independent={key:values['whole4'][key]-values['independent4'][key]
                    if values['whole4'][key] is not None and values['independent4'][key] is not None else None for key in FIELDS}))
    summary=dict(protocol=EVALUATION_PROTOCOL,tip_evaluation_protocol=PROTOCOL,split='DEV_MODEL',
        checkpoint_sha256=generated['checkpoint_sha256'],checkpoint_step=generated['checkpoint_step'],
        generation_summary_sha256=sha256(args.generation/'summary.json'),
        repeats=generated['repeats'],results_by_repeat=all_results,
        methods={method:average([r for r in all_results if r['method']==method]) for method in METHODS},
        candidate_pooling_between_repeats=False,source_sha256={__file__:sha256(__file__),
            'scripts/evaluate_observed_obstacles.py':sha256(Path(__file__).with_name('evaluate_observed_obstacles.py'))},
        evaluation_input_sha256=input_hashes,cpu_wall_seconds=time.perf_counter()-started,
        scope='Same trained checkpoint and all24 DEV; sampling repeats are evaluated separately before metric averaging. '
              'Over-budget scenes conservatively fail all four nominal slots, with actual charged pool separately reported. '
              'No scorer/SelectedValid estimate; boxes-only tip validity excludes arm, IK, table and execution. '
              'Generation+checker is measured component sum, not a jointly timed deployed call. No refinement or reference candidate replacement.')
    write_json(args.output/'parent_paired_differences.json',paired);write_json(args.output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('checkpoint_step','repeats','methods','cpu_wall_seconds')}),flush=True)


if __name__=='__main__':main()
