"""Build mixed-scope evidence tables without filling unmeasured metrics."""
import csv
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]; reports=root/'reports'; rows=[]
    historical=json.loads((reports/'multiseed_summary.json').read_text())
    for method,splits in historical['summary'].items():
        for split,metrics in splits.items():
            if split not in ['test','ood']: continue
            for index,seed in enumerate(historical['training_seeds']):
                rows.append(dict(tier='historical_controlled_CLIP_oracle',task='fixed_three_modes',protocol='historical_public',
                    method=method,seed=seed,split=split.upper(),K=3,
                    run_id='historical_'+method+'_seed'+str(seed),
                    ValidAtK=metrics['valid_rate']['values'][index],AnyValidAtK=metrics['success']['values'][index],
                    UniqueValidAtK=metrics['unique_valid']['values'][index],ReferenceCoverageAtK=metrics['coverage']['values'][index],
                    source='reports/multiseed_summary.json'))
    source=reports/'v2_controlled'/'results.json'
    if source.exists():
        for row in json.loads(source.read_text()):
            rows.append(dict(tier=row['tier'],task='variable_opening_pairs',protocol='development_selection',
                method=row['objective']+'_set_regression',seed=row['seed'],split=row['split'],K=row['K'],
                run_id=row['run_id'],training_exposures=row['gradient_target_exposures'],training_threads=4,
                cost_scope='entire training run including development evaluation; no VLM; deduplicate by run_id',
                ValidAtK=row['ValidAtK'],AnyValidAtK=row['AnyValidAtK'],UniqueValidAtK=row['UniqueValidAtK'],
                ReferenceCoverageAtK=row['ReferenceCoverageAtK'],head_batch1_ms=row['head_batch1_ms'],
                elapsed_s=row['elapsed_s'],gpu_hours=row['gpu_hours'],code_commit=row['code_commit'],source=str(source.relative_to(root))))
    source=reports/'v2_planner'/'results.json'
    if source.exists():
        for row in json.loads(source.read_text()):
            rows.append(dict(tier=row['tier'],task='variable_opening_pairs',protocol='analytical_special_case',method=row['method'],
                seed=None,split=row['split'],K=row['K'],ValidAtK=row['valid_rate'],AnyValidAtK=row['success'],
                UniqueValidAtK=row['unique_valid'],ReferenceCoverageAtK=row['reference_coverage'],SelectedValidAtK=row['selected_valid'],
                source=str(source.relative_to(root))))
    completion_sources = sorted((reports/'v2_completion').glob('*/summary.json'))
    completion_sources += sorted((reports/'v2_completion/replication').glob('seed*/*/summary.json'))
    completion_sources += sorted((reports/'v2_completion_selfdraft').glob('seed*/*/summary.json'))
    for source in completion_sources:
        result=json.loads(source.read_text())
        config_path=source.parent/'config.json'
        config=json.loads(config_path.read_text()) if config_path.exists() else {'seed':0}
        for context,metrics in result['metrics'].items():
            if not isinstance(metrics,dict): continue
            rows.append(dict(tier='controlled_completion',task=context,protocol=metrics['budget_description'],
                method=source.parent.name+('_selfdraft' if config.get('self_draft_prob',0)>0 else ''),
                seed=config['seed'],split='DEV_MODEL',K=metrics['total_candidates'],
                run_id=str(source.parent.relative_to(reports)),training_exposures=result['trajectory_exposures'],
                training_threads=config.get('threads',4),extra_training_draft_forwards=result.get('extra_training_draft_forwards',0),
                cost_scope='entire training run repeated across contexts; deduplicate by run_id; no VLM',
                ValidAtK=metrics['union_valid_rate'],AnyValidAtK=metrics['union_success'],UniqueValidAtK=metrics['union_unique_valid'],
                ReferenceCoverageAtK=metrics['union_reference_coverage'],additional_unique_valid=metrics['additional_unique_valid'],
                new_valid_rate=metrics['new_valid_rate'],new_candidates=metrics['new_candidates'],
                supplied_candidates=metrics['supplied_candidates'],end_to_end_controlled_ms=metrics['batch1_gate_head_check_ms_median'],
                elapsed_s=result['elapsed_s'],gpu_hours=result['gpu_hours_reserved'],
                code_commit=config.get('code_commit'),source=str(source.relative_to(root))))
    for folder in ['observed_frozen_v1','observed_online_v1','observed_online_warm_v2','observed_geometry_v1',
                   'observed_geometry_grounding_v2','observed_geometry_seeds_v2']:
        for source in sorted((reports/folder).glob('*/summary.json')):
            result=json.loads(source.read_text()); metrics=result['metrics']
            config=json.loads((source.parent/'config.json').read_text())
            rows.append(dict(tier='observed_RGBD_language_current' if 'geometry' in folder else 'observed_RGB_language_current',
                task='RLBench_derived_reach_three_targets',
                protocol=metrics.get('evaluation_protocol','observation_eval_v1_reference_subset'),
                method=folder+'_'+config.get('adapter_mode','frozen_cache'),
                seed=config['seed'],split='DEV_MODEL',K=config['candidates'],
                run_id=str(source.parent.relative_to(reports)),training_exposures=result['trajectory_exposures'],
                selected_step=result['best_step'],grounding_weight=config.get('grounding_weight'),
                training_threads=config['threads'],reference_evaluation_examples=metrics.get('reference_evaluation_examples',metrics['examples']),
                semantic_evaluation_examples=metrics['semantic_evaluation_examples'],
                cost_scope=('online Qwen/head train and evaluation; gpu_hours includes setup; elapsed_s excludes setup; common head pretraining separate'
                    if 'online' in folder else 'head training/evaluation only; separate Qwen encoding is excluded'),
                candidate_ADE_m=metrics['candidate_matched_ADE_m'],endpoint_error_m=metrics['candidate_endpoint_error_m'],
                semantic_goal_accuracy=metrics['semantic_goal_accuracy'],AnySemanticGoalAtK=metrics['AnySemanticGoalAtK'],
                elapsed_s=result['elapsed_s'],gpu_hours=result['gpu_hours_reserved'],code_commit=config['code_commit'],
                source=str(source.relative_to(root))))
    for source in sorted((reports/'observation_eval_v2').glob('*/metrics.json')):
        metrics=json.loads(source.read_text()); provenance=json.loads((source.parent/'provenance.json').read_text())
        config=provenance.get('original_training_config',{})
        run_path=provenance.get('original_run',provenance.get('run',''))
        if not config:
            stored=reports/'observed_frozen_v1/seed0/config.json'
            config=json.loads(stored.read_text())
        rows.append(dict(tier='observed_RGB_language_current',task='RLBench_derived_reach_three_targets',
            protocol=metrics['evaluation_protocol'],method=source.parent.name,
            run_id=run_path,seed=config['seed'],split='DEV_MODEL',K=config['candidates'],
            selected_step=provenance.get('checkpoint_step',provenance.get('original_checkpoint_step')),
            candidate_ADE_m=metrics['candidate_matched_ADE_m'],endpoint_error_m=metrics['candidate_endpoint_error_m'],
            semantic_goal_accuracy=metrics['semantic_goal_accuracy'],AnySemanticGoalAtK=metrics['AnySemanticGoalAtK'],
            reference_evaluation_examples=metrics['reference_evaluation_examples'],
            semantic_evaluation_examples=metrics['semantic_evaluation_examples'],
            cost_scope='reevaluation of same saved checkpoint; original training cost is listed once in the corresponding training run',
            code_commit=provenance['original_training_source_commit'],source=str(source.relative_to(root))))
    for folder in ['v2_completion/rollout_seed0','v2_completion_selfdraft/rollout_seed0']:
        source=reports/folder/'results.json'
        if not source.exists(): continue
        for result in json.loads(source.read_text()):
            rows.append(dict(tier='controlled_selfdraft_rollout',task='variable_opening_pairs',protocol='strict_total4_no_discard',
                method=result['mechanism']+'_'+result['mode']+('_mixed_training' if 'selfdraft' in folder else '_reference_training'),
                seed=result['seed'],split=result['split'],K=result['K'],ValidAtK=result['ValidAtK'],
                AnyValidAtK=result['AnyValidAtK'],UniqueValidAtK=result['UniqueValidAtK'],
                ReferenceCoverageAtK=result['ReferenceCoverageAtK'],SelectedValidAtK=result['SelectedValidAtK'],
                end_to_end_controlled_ms=result['median_ms'],forward_passes=result['forward_passes'],
                cost_scope=result['latency_scope'],source=str(source.relative_to(root))))
    keys=['tier','task','protocol','method','seed','split','K','run_id','ValidAtK','AnyValidAtK','UniqueValidAtK','ReferenceCoverageAtK',
          'SelectedValidAtK','additional_unique_valid','new_valid_rate','new_candidates','supplied_candidates','head_batch1_ms',
          'end_to_end_controlled_ms','candidate_ADE_m','endpoint_error_m','semantic_goal_accuracy','AnySemanticGoalAtK',
          'reference_evaluation_examples','semantic_evaluation_examples','training_exposures','training_threads','selected_step','grounding_weight',
          'extra_training_draft_forwards','forward_passes','cost_scope','elapsed_s','gpu_hours','code_commit','source']
    rows=[{key:row.get(key) for key in keys} for row in rows]
    (reports/'MAIN_RESULTS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (reports/'MAIN_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print('Wrote %d measured result rows; null means unmeasured.'%len(rows))


if __name__=='__main__':main()
