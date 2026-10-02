"""Verify saved ordinary/auxiliary artifacts, then compare unchanged DEV metrics."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(ordinary, auxiliary, historical_preflight=None):
    paths={'ordinary':Path(ordinary),'auxiliary':Path(auxiliary)}
    summaries={name:read(path/'summary.json') for name,path in paths.items()}
    configs={name:read(path/'config.json') for name,path in paths.items()}
    a,b=configs['ordinary'],configs['auxiliary']
    keys=('observations','supervision','cache_dir','steps','batch_size','candidates','horizon','width','depth','pooling',
        'geometry_pooling','anchor_mode','endpoint_mode','sampling_mode','metric_aggregation','refinement_mode',
        'checkpoint_selection','point_width','pixel_stride','endpoint_residual_bound','grounding_sigma','event_scale','lr',
        'seed','eval_every','feature_dim','dataset_fingerprint','train_examples','dev_examples','train_parents','dev_parents',
        'checkpoint_selection_protocol','evaluation_protocol')
    for key in keys:
        if a[key]!=b[key]:raise ValueError('Paired setting differs: '+key)
    if (a['grounding_weight'],a.get('grounding_target','endpoint'),b['grounding_weight'],b['grounding_target'])!=(0.,'endpoint',.02,'event_supported'):
        raise ValueError('Expected registered ordinary/event-supported auxiliary pair')
    source_indices={name:read(path/'source_hashes.json') for name,path in paths.items()}
    if source_indices['ordinary']!=source_indices['auxiliary']:raise ValueError('Actual input/reference/cache source hashes differ')
    artifacts={}
    for name,path in paths.items():
        summary=summaries[name]
        expected={'best.pt':'best_checkpoint_sha256','last.pt':'last_checkpoint_sha256',
            'dev_model/predictions.npz':'prediction_sha256','last_dev_model/predictions.npz':'last_prediction_sha256'}
        artifacts[name]={filename:sha(path/filename) for filename in expected}
        if any(artifacts[name][filename]!=summary[field] for filename,field in expected.items()):
            raise ValueError('Actual checkpoint/prediction hashes differ from summary')
        if summary['trajectory_exposures']!=a['steps']*a['batch_size']*a['candidates']:
            raise ValueError('Training exposure mismatch')
        if summary['parameters']!=1231965:raise ValueError('Unexpected architecture size')
    audits=[summaries[name].get('sample_stream_audit') for name in paths]
    if all(audit is not None for audit in audits):
        if audits[0]!=audits[1]:raise ValueError('Actual initial state or sampled stream differs')
        if audits[0]['batches']!=a['steps'] or audits[0]['observation_draws']!=a['steps']*a['batch_size']:
            raise ValueError('Incomplete actual sampled-index audit')
        initialization=dict(evidence='actual per-run initial state and actual draw hash chain',audit=audits[0])
    elif any(audit is not None for audit in audits):raise ValueError('One-sided actual sample audit')
    else:
        if historical_preflight is None:raise ValueError('Historical seed0 requires explicit actual-source replay receipt')
        receipt=read(historical_preflight)
        if a['seed']!=0 or receipt['status']!='passed' or not receipt['initial_forward_bit_equal'] or not receipt['sampler_final_state_equals_completed_control']:
            raise ValueError('Missing historical initial-forward/sampling evidence')
        if receipt['ordinary_checkpoint_sha256']!=summaries['ordinary']['last_checkpoint_sha256'] or receipt['ordinary_config_sha256']!=sha(paths['ordinary']/'config.json'):
            raise ValueError('Historical preflight is for another control')
        if receipt['grounding_target_fingerprint']!=summaries['auxiliary']['grounding_target_fingerprint']:
            raise ValueError('Historical target fingerprint mismatch')
        initialization=dict(evidence='seed0 actual-source/seed reconstruction and full sampler replay only; no historical actual draw digest or saved initial checkpoint',receipt_sha256=sha(historical_preflight),receipt=receipt)
    metrics={};task_deltas=[];parent_deltas=[]
    fields=('candidate_matched_ADE_m','reference_matched_ADE_m','candidate_endpoint_error_m','best_endpoint_error_m','event_state_accuracy','event_sequence_accuracy')
    for label,key in [('best','metrics'),('last','last_metrics'),('train_best','train_metrics')]:
        x,y=[summaries[name][key] for name in paths]
        for count in ('examples','parents','reference_evaluation_examples','semantic_evaluation_examples','examples_without_reference'):
            if x[count]!=y[count]:raise ValueError('Evaluation denominator differs: '+count)
        for value in (x,y):
            if value['semantic_goal_accuracy'] is not None or value['ValidAtK'] is not None:
                raise ValueError('Generic multitask run has unsupported success metrics')
        metrics[label]={name:{field:summaries[name][key][field] for field in fields} for name in paths}
        metrics[label]['denominators']={field:x[field] for field in ('examples','parents','reference_evaluation_examples','semantic_evaluation_examples','examples_without_reference')}
        for category,id_key,destination in [('per_task','task',task_deltas),('per_parent','parent_id',parent_deltas)]:
            left={row[id_key]:row for row in x['task_parent_reference_metrics'][category]}
            right={row[id_key]:row for row in y['task_parent_reference_metrics'][category]}
            if left.keys()!=right.keys():raise ValueError('Missing paired task/parent')
            for ident in sorted(left):
                l,r=left[ident],right[ident]
                row=dict(checkpoint=label,task=l['task'],
                    ordinary={field:l[field] for field in fields},auxiliary={field:r[field] for field in fields},
                    delta_auxiliary_minus_ordinary={field:(r[field]-l[field] if r[field] is not None and l[field] is not None else None) for field in fields})
                row[id_key]=ident;destination.append(row)
    return dict(protocol='multitask_landmark_paired_saved_artifacts_v1',status='passed',seed=a['seed'],
        analysis_source_sha256=sha(__file__),runs={name:str(path) for name,path in paths.items()},
        matched_config={key:a[key] for key in keys},initialization_and_stream=initialization,artifacts=artifacts,
        actual_source_index_sha256={name:sha(path/'source_hashes.json') for name,path in paths.items()},
        metrics=metrics,per_task=task_deltas,per_parent=parent_deltas,
        cost={name:{field:summary[field] for field in ('elapsed_s','data_load_preprocess_s','gpu_hours_reserved','parameters','trajectory_exposures','peak_cuda_memory_mb','best_step','last_step')} for name,summary in summaries.items()},
        interpretation='Same reused 12 DEV parents/48 language inputs, one paired training seed. Reference/event reconstruction only, no semantic/contact/collision/execution certification. Positive ordinary-baseline evidence is not a novel route-set mechanism.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('ordinary','auxiliary','output'):parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--historical-preflight',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh analysis output required')
    result=analyze(args.ordinary,args.auxiliary,args.historical_preflight)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='passed',seed=result['seed'],metrics=result['metrics'])))


if __name__=='__main__':main()
