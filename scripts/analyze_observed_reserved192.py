"""Audit the equal-budget new192 soft/peak pair, keeping best and last separate."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--observations',type=Path)
    parser.add_argument('--supervision',type=Path)
    args=parser.parse_args()
    fields=('semantic_goal_accuracy','AnySemanticGoalAtK','candidate_matched_ADE_m','candidate_endpoint_error_m')
    shared=('dataset_fingerprint','feature_dim','horizon','candidates','width','depth','point_width','pixel_stride',
            'endpoint_residual_bound','grounding_weight','grounding_sigma','event_scale','pooling','lr','steps','batch_size','seed','eval_every')
    arms,configs={},{}
    for method in ('soft','peak'):
        folder=args.runs/(method+'_seed0')
        config=json.loads((folder/'config.json').read_text());summary=json.loads((folder/'summary.json').read_text())
        history=json.loads((folder/'history.json').read_text())
        assert json.loads((folder/'status.json').read_text())==dict(status='completed',step=3000,exit_code=0)
        assert config['seed']==0 and config.get('anchor_mode','soft')==('soft' if method=='soft' else 'straight_through_peak')
        assert config['steps']==3000 and config['batch_size']==32 and config['candidates']==4
        assert summary['trajectory_exposures']==384000 and summary['parameters']==1231965 and history[-1]['step']==3000
        assert summary['metrics']['examples']==summary['metrics']['semantic_evaluation_examples']==summary['metrics']['reference_evaluation_examples']==48
        assert summary['train_metrics']['examples']==576
        files={str(path.relative_to(args.runs)):dict(sha256=digest(path),bytes=path.stat().st_size)
               for path in sorted(folder.rglob('*')) if path.suffix in ('.pt','.npz')}
        assert files[str((folder/'best.pt').relative_to(args.runs))]['sha256']==summary['best_checkpoint_sha256']
        configs[method]=config
        arms[method]=dict(best_step=summary['best_step'],best={key:summary['metrics'][key] for key in fields},
            last_step=3000,last={key:history[-1]['dev_model'][key] for key in fields},last_metric_source='actual step3000 history, not substituted for original best',
            train_semantic_at_best=summary['train_metrics']['semantic_goal_accuracy'],parameters=summary['parameters'],
            candidate_exposures=summary['trajectory_exposures'],elapsed_s=summary['elapsed_s'],gpu_hours_reserved=summary['gpu_hours_reserved'],
            peak_cuda_memory_mb=summary['peak_cuda_memory_mb'],latency=summary['latency'],source_commit=config['code_commit'],
            summary_sha256=digest(folder/'summary.json'),config_sha256=digest(folder/'config.json'),history_sha256=digest(folder/'history.json'),
            binary_files=files)
    assert all(configs['soft'][key]==configs['peak'][key] for key in shared),'paired training/input settings differ'
    result=dict(evaluation_protocol='observation_eval_v2',setting='natural reserved192 TRAIN /16 DEV_MODEL',
        seeds=[0],paired_settings_identical_except_anchor_mode=True,common_training_candidate_slots=384000,
        comparison_scope='within new192 equal budget only; old64 had one-third data and total training exposures',
        exposure_per_supervised_instruction=96000/576,selection='original DEV reference ADE; best and last3000 both preserved',
        arms=arms,paired_gain={stage:{key:arms['peak'][stage][key]-arms['soft'][stage][key] for key in fields} for stage in ('best','last')},
        caveat='one training seed; baseline grounding repair, no route-type or full robot validity evidence',script_sha256=digest(__file__))
    if args.observations or args.supervision:
        if not (args.observations and args.supervision):raise ValueError('both metadata manifests required for post-hoc diagnostics')
        observations={row['id']:row for row in map(json.loads,args.observations.read_text().splitlines())}
        labels={row['id']:row for row in map(json.loads,args.supervision.read_text().splitlines())}
        diagnostic={}
        for method in ('soft','peak'):
            folder=args.runs/(method+'_seed0')/'dev_model'
            metrics={row['scene_id']:row for row in json.loads((folder/'per_scene.json').read_text())}
            counts=Counter();rows=[]
            with np.load(folder/'predictions.npz',allow_pickle=False) as archive:
                for identifier,xyz in zip(map(str,archive['scene_ids']),archive['paths'][:,:,-1]):
                    label=labels[identifier]['semantic_targets'];target=label['target_index']
                    assert label['tolerance']==.03
                    distances=np.linalg.norm(xyz[:,None]-np.asarray(label['centers'])[None],axis=-1)
                    nearest=distances.argmin(axis=1);success=(nearest==target)&(distances[:,target]<=.03)
                    categories=['semantic_success' if ok else ('nearest_identity_wrong' if identity!=target else 'nearest_identity_correct_beyond_3cm') for ok,identity in zip(success,nearest)]
                    assert float(success.mean())==metrics[identifier]['semantic_goal_accuracy']
                    counts.update(categories)
                    rows.append(dict(id=identifier,instruction=observations[identifier]['instruction'],semantic_success_candidates=int(success.sum()),
                        expected_target_index=target,nearest_target_indices=nearest.tolist(),candidate_categories=categories,target_center_error_m=distances[:,target].tolist()))
            assert len(rows)==48
            diagnostic[method]=dict(candidate_category_counts=dict(counts),all_48_scenes=rows,all_failure_ids=[row['id'] for row in rows if row['semantic_success_candidates']<4])
        result['posthoc_best_endpoint_diagnostic']=dict(arms=diagnostic,supervision_sha256=digest(args.supervision),observations_sha256=digest(args.observations),
            scope='labels used only after saved predictions; no forward or model change; all 48 DEV included')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({method:{key:arm[key] for key in ('best_step','best','last','elapsed_s','gpu_hours_reserved')} for method,arm in arms.items()},indent=2))


if __name__=='__main__':main()
