"""Paired seed analysis of the conventional soft/peak observation baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,default=Path('reports'))
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if len(set(args.seeds))!=len(args.seeds):raise ValueError('duplicate seeds')
    fields=('semantic_goal_accuracy','AnySemanticGoalAtK','candidate_matched_ADE_m','candidate_endpoint_error_m')
    shared=('dataset_fingerprint','feature_dim','horizon','candidates','width','depth','point_width','pixel_stride',
            'endpoint_residual_bound','grounding_weight','grounding_sigma','event_scale','pooling','lr','steps','batch_size','seed','eval_every')
    settings={}
    for setting,soft_family in [('natural64','observed_learning_curve_new64_v1'),('obstacle32','observed_obstacle_new32_v1')]:
        pairs=[]
        for seed in args.seeds:
            pair=dict(seed=seed,arms={})
            configs={}
            for method in ('soft','peak'):
                if seed==0:
                    run=args.reports/soft_family/'aux_seed0' if method=='soft' else args.reports/'observed_anchor_peak_v1'/(setting+'_seed0')
                else:
                    run=args.reports/'observed_anchor_peak_replication_v1'/setting/(method+'_seed'+str(seed))
                summary=json.loads((run/'summary.json').read_text())
                config=json.loads((run/'config.json').read_text())
                history=json.loads((run/'history.json').read_text())
                assert config['seed']==seed and config.get('anchor_mode','soft')==('soft' if method=='soft' else 'straight_through_peak')
                assert summary['parameters']==1231965 and summary['trajectory_exposures']==128000
                assert history[-1]['step']==1000
                configs[method]=config
                pair['arms'][method]=dict(report=str(run),best_step=summary['best_step'],
                    best={key:summary['metrics'][key] for key in fields},last={key:history[-1]['dev_model'][key] for key in fields},
                    train_semantic_at_best=summary['train_metrics']['semantic_goal_accuracy'],
                    examples=summary['metrics']['examples'],reference_examples=summary['metrics']['reference_evaluation_examples'],
                    elapsed_s=summary['elapsed_s'],gpu_hours_reserved=summary['gpu_hours_reserved'],
                    parameters=summary['parameters'],candidate_exposures=summary['trajectory_exposures'],
                    summary_sha256=digest(run/'summary.json'),history_sha256=digest(run/'history.json'),config_sha256=digest(run/'config.json'))
            assert all(configs['soft'][key]==configs['peak'][key] for key in shared),'paired settings differ'
            pair['paired_gain']={stage:{key:pair['arms']['peak'][stage][key]-pair['arms']['soft'][stage][key] for key in fields} for stage in ('best','last')}
            pairs.append(pair)
        aggregate={}
        for method in ('soft','peak'):
            aggregate[method]={}
            for stage in ('best','last'):
                aggregate[method][stage]={}
                for key in fields:
                    values=[pair['arms'][method][stage][key] for pair in pairs]
                    aggregate[method][stage][key]=dict(mean=float(np.mean(values)),std=float(np.std(values,ddof=1)) if len(values)>1 else None,values=values)
        settings[setting]=dict(seeds=args.seeds,pairs=pairs,aggregate=aggregate,
                              caveat='conventional grounding baseline repair; no robot execution or route-type coverage claim')
    result=dict(evaluation_protocol='observation_eval_v2',shared_condition='frozen real Qwen plus current RGB-D/state',
                selection='original DEV reference ADE, no endpoint-selected checkpoint replacement',settings=settings,
                script_sha256=digest(__file__))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({name:{method:value['aggregate'][method]['best']['semantic_goal_accuracy'] for method in ('soft','peak')} for name,value in settings.items()}))


if __name__=='__main__':main()
