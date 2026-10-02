"""Uniform exploratory fixed-step evaluation of every paired anchor baseline.

The original DEV-ADE-selected outputs remain immutable. All requested soft and
peak runs must finish first; every arm uses its step1000 last checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch
from routeset.common import sha256,write_json
from routeset.observed_route_head import load_observed_dataset
from routeset.observed_geometry import ObservedGeometryRouteHead
from scripts.train_observed_geometry import load_geometry,evaluate


def source_run(root,setting,method,seed):
    if seed:
        return root/'runs/observed_anchor_peak_replication_v1'/setting/(method+'_seed'+str(seed))
    if method=='peak':return root/'runs/observed_anchor_peak_v1'/(setting+'_seed0')
    family='observed_learning_curve_new64_v1' if setting=='natural64' else 'observed_obstacle_new32_v1'
    return root/'runs'/family/'aux_seed0'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    args=parser.parse_args()
    torch.set_num_threads(1)
    if args.output.exists():raise FileExistsError('preserve existing fixed-step output')
    sources=[(setting,method,seed,source_run(args.root,setting,method,seed))
             for setting in ('natural64','obstacle32') for seed in args.seeds for method in ('soft','peak')]
    for setting,method,seed,run in sources:
        status=json.loads((run/'status.json').read_text())
        if status!=dict(status='completed',step=1000,exit_code=0):raise ValueError('all arms must finish first: '+str(run))
    cached_data={}
    index=[]
    for setting,method,seed,run in sources:
        checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
        config=checkpoint['config']
        expected_mode='soft' if method=='soft' else 'straight_through_peak'
        if checkpoint['step']!=1000 or config['seed']!=seed or config.get('anchor_mode','soft')!=expected_mode:
            raise ValueError('fixed checkpoint/method/seed mismatch')
        if setting not in cached_data:
            data=load_observed_dataset(config['observations'],config['supervision'],config['cache_dir'],config['horizon'],config['pooling'])
            geometry=load_geometry(data,config['observations'],config['supervision'],config['pixel_stride'])
            cached_data[setting]=(data,geometry)
        data,geometry=cached_data[setting]
        if geometry['fingerprint']!=config['dataset_fingerprint']:raise ValueError('paired input fingerprints differ')
        model=ObservedGeometryRouteHead(config['feature_dim'],config['horizon'],config['candidates'],config['width'],
            config['depth'],config['point_width'],config['endpoint_residual_bound'],anchor_mode=expected_mode).eval()
        model.load_state_dict(checkpoint['model'],strict=True)
        ids=np.flatnonzero(data['splits']=='DEV_MODEL')
        out=args.output/setting/(method+'_seed'+str(seed))
        metrics=evaluate(model,data,geometry,ids,'cpu',out)
        metrics.update(checkpoint_selection_protocol='fixed_step_1000',checkpoint_step=1000,
                       evaluation_device='cpu',
                       selection_note='uniform fixed step for BOTH methods and ALL seeds; exploratory DEV sensitivity, not original best selection')
        write_json(out/'metrics.json',metrics)
        provenance=dict(setting=setting,method=method,seed=seed,training_run=str(run),
            checkpoint=str(run/'last.pt'),checkpoint_sha256=sha256(run/'last.pt'),
            original_best_checkpoint_sha256=sha256(run/'best.pt'),original_config_sha256=sha256(run/'config.json'),
            training_source_commit=config['code_commit'],evaluation_source_commit=os.environ.get('CODE_COMMIT'),
            evaluation_script_sha256=sha256(__file__),checkpoint_selection_protocol='fixed_step_1000',
            evaluation_device='cpu',torch_version=torch.__version__,
            semantic_protocol='observation_eval_v2',all_requested_arms=[str(item[3]) for item in sources],
            prediction_sha256=sha256(out/'predictions.npz'),candidate_exposures=checkpoint['trajectory_exposures'],
            evaluation_scope='cached genuine frozen Qwen and current RGB-D; no online LoRA or full latency claim')
        write_json(out/'provenance.json',provenance)
        index.append(dict(setting=setting,method=method,seed=seed,metrics=metrics,provenance=provenance))
        print(json.dumps(dict(setting=setting,method=method,seed=seed,semantic=metrics['semantic_goal_accuracy'],
                             ADE=metrics['candidate_matched_ADE_m'])),flush=True)
    write_json(args.output/'index.json',dict(checkpoint_selection_protocol='fixed_step_1000',runs=index))


if __name__=='__main__':main()
