"""Transfer six already-selected natural64 heads to reserved fresh DEV_MODEL.

No training, model selection, optimizer step or resume occurs. Original training
configs/fingerprints/checkpoints remain untouched. All fresh DEV inputs enter
the existing observation_eval_v2 metric, including any without references.
"""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from routeset.common import sha256,write_json
from routeset.observed_geometry import ObservedGeometryRouteHead
from routeset.observed_route_head import load_observed_dataset
from scripts.reevaluate_anchor_fixed_step import source_run
from scripts.train_observed_geometry import load_geometry,evaluate


SELECTION_PROTOCOL='original_dev_best_fresh_dev_transfer'
EXPECTED_MANIFEST_SHA='aa17ecef147f73ada2e902ac30f7de39be0acc47a25ced5959607619bf945495'


def verify_export(data_dir,reservation):
    manifest=json.loads((data_dir/'export_manifest.json').read_text())
    registry=json.loads(reservation.read_text())
    collection=next(item for item in registry['collections'] if item['source']=='data/observation_reach_fast256_20261002')
    roles=collection['roles_inclusive']
    parents={role:{collection['parent_prefix']+('%06d'%number) for number in range(bounds[0],bounds[1]+1)}
             for role,bounds in roles.items() if role in ('TRAIN','DEV_MODEL')}
    rows=[json.loads(line) for line in (data_dir/'observations.jsonl').read_text().splitlines() if line.strip()]
    if set(manifest['requested_roles'])!={'TRAIN','DEV_MODEL'}:raise ValueError('development-only export required')
    if manifest['reservation_sha256']!=sha256(reservation):raise ValueError('reservation changed after export')
    if sha256(data_dir/'observations.jsonl')!=EXPECTED_MANIFEST_SHA:raise ValueError('fresh evaluation manifest changed')
    if sha256(data_dir/'supervision.jsonl')!=manifest['supervision_manifest_sha256']:raise ValueError('fresh supervision changed')
    if any(set(row)!={'id','parent_id','split','image','instruction'} for row in rows):raise ValueError('invalid observation contract')
    if any(row['split'] not in parents or row['parent_id'] not in parents[row['split']] for row in rows):
        raise ValueError('input parent is outside its reserved development role')
    if {row['parent_id'] for row in rows if row['split']=='DEV_MODEL'}!=parents['DEV_MODEL']:
        raise ValueError('all reserved fresh development parents required; do not select successful parents')
    return manifest,parents['DEV_MODEL']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--reservation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError('fresh output required; never overwrite prior development evaluations')
    manifest,fresh_parents=verify_export(args.data,args.reservation)
    sources=[(method,seed,source_run(args.root,'natural64',method,seed)) for seed in (0,1,2) for method in ('soft','peak')]
    configs=[]
    for method,seed,run in sources:
        if json.loads((run/'status.json').read_text())!=dict(status='completed',step=1000,exit_code=0):
            raise ValueError('all six source trainings must be complete')
        config=json.loads((run/'config.json').read_text())
        old_rows=[json.loads(line) for line in Path(config['observations']).read_text().splitlines() if line.strip()]
        if {row['parent_id'] for row in old_rows}&fresh_parents:
            raise ValueError('fresh development parent already appeared in source training/selection inputs')
        configs.append(config)
    architecture=('feature_dim','horizon','candidates','width','depth','point_width','endpoint_residual_bound','pixel_stride','pooling')
    if any(any(config[key]!=configs[0][key] for key in architecture) for config in configs):
        raise ValueError('six source architecture/input settings differ')
    tic=time.perf_counter()
    config=configs[0]
    data=load_observed_dataset(args.data/'observations.jsonl',args.data/'supervision.jsonl',args.data/'qwen_cache',
                               config['horizon'],config['pooling'])
    geometry=load_geometry(data,args.data/'observations.jsonl',args.data/'supervision.jsonl',config['pixel_stride'])
    ids=np.flatnonzero(data['splits']=='DEV_MODEL')
    if set(data['parent_ids'][ids])!=fresh_parents or len(ids)!=48:raise ValueError('fresh 16-parent/48-instruction denominator changed')
    preprocessing_seconds=time.perf_counter()-tic
    index=[]
    for (method,seed,run),source_config in zip(sources,configs):
        checkpoint_path=run/'best.pt'
        before_hash=sha256(checkpoint_path)
        checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
        config=checkpoint['config']
        mode='soft' if method=='soft' else 'straight_through_peak'
        if config.get('anchor_mode','soft')!=mode or config['seed']!=seed:raise ValueError('checkpoint arm does not match requested pair')
        if config['dataset_fingerprint']!=source_config['dataset_fingerprint']:raise ValueError('source checkpoint/config fingerprint mismatch')
        if config['dataset_fingerprint']==geometry['fingerprint']:raise ValueError('expected distinct transfer evaluation data')
        model=ObservedGeometryRouteHead(config['feature_dim'],config['horizon'],config['candidates'],config['width'],
            config['depth'],config['point_width'],config['endpoint_residual_bound'],anchor_mode=mode).eval()
        model.load_state_dict(checkpoint['model'],strict=True)
        out=args.output/(method+'_seed'+str(seed))
        started=time.perf_counter()
        metrics=evaluate(model,data,geometry,ids,'cpu',out)
        elapsed=time.perf_counter()-started
        metrics.update(checkpoint_selection_protocol=SELECTION_PROTOCOL,checkpoint_step=checkpoint['step'],
                       evaluation_device='cpu',selection_note='original old-DEV-selected best transferred unchanged; no selection or training on this fresh DEV',
                       development_only=True)
        write_json(out/'metrics.json',metrics)
        if sha256(checkpoint_path)!=before_hash:raise RuntimeError('source checkpoint changed during evaluation')
        provenance=dict(method=method,seed=seed,source_training_run=str(run),source_checkpoint=str(checkpoint_path),
            checkpoint_sha256=before_hash,source_config_sha256=sha256(run/'config.json'),source_training_commit=config['code_commit'],
            source_training_dataset_fingerprint=config['dataset_fingerprint'],fresh_evaluation_dataset_fingerprint=geometry['fingerprint'],
            source_training_candidate_exposures=checkpoint['trajectory_exposures'],additional_training_exposures=0,
            evaluation_source_commit=os.environ.get('CODE_COMMIT'),evaluation_script_sha256=sha256(__file__),
            reservation_sha256=sha256(args.reservation),export_manifest_sha256=sha256(args.data/'export_manifest.json'),
            observations_sha256=sha256(args.data/'observations.jsonl'),supervision_sha256=sha256(args.data/'supervision.jsonl'),
            cache_config_sha256=sha256(args.data/'qwen_cache/cache_config.json'),prediction_sha256=sha256(out/'predictions.npz'),
            fresh_development_parent_ids=sorted(fresh_parents),checkpoint_selection_protocol=SELECTION_PROTOCOL,
            semantic_protocol='observation_eval_v2',source_checkpoint_unchanged=True,torch_version=torch.__version__,
            cpu_evaluation_seconds=elapsed,shared_data_preprocessing_seconds=preprocessing_seconds,
            timing_scope='CPU evaluation with genuine frozen-Qwen cache; not full-request Qwen latency',
            all_requested_arms=[str(item[2]) for item in sources],locked_test_access=False)
        write_json(out/'provenance.json',provenance)
        index.append(dict(method=method,seed=seed,metrics=metrics,provenance=provenance))
        print(json.dumps(dict(method=method,seed=seed,semantic=metrics['semantic_goal_accuracy'],ADE=metrics['candidate_matched_ADE_m'])),flush=True)
    aggregate={}
    for method in ('soft','peak'):
        aggregate[method]={}
        for key in ('semantic_goal_accuracy','AnySemanticGoalAtK','candidate_matched_ADE_m','candidate_endpoint_error_m'):
            values=[row['metrics'][key] for row in index if row['method']==method]
            aggregate[method][key]=dict(values=values,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)))
    write_json(args.output/'index.json',dict(checkpoint_selection_protocol=SELECTION_PROTOCOL,runs=index,aggregate=aggregate,
        setting='natural64 models on fresh reserved natural16 DEV; same 16 parents for all six arms',
        requested_parent_counts=manifest['requested_parent_counts'],new_training_performed=False))


if __name__=='__main__':main()
