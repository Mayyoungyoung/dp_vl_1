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
    for source in sorted(reports.glob('multigate_diffusion_*/training/seed*/*/summary.json')):
        result=json.loads(source.read_text()); config=json.loads((source.parent/'config.json').read_text())
        for budget, entry in result['metrics'].items():
            metrics=entry['mean']; timing=entry['repeats'][0]
            rows.append(dict(tier='controlled_oracle_diffusion',task='variable_opening_pairs',
                protocol='DEV_selected_separate_sampling_repeats',method=result['arm']+'_'+config.get('parameterization','epsilon'),
                seed=config['seed'],split='DEV_MODEL',K=int(budget[1:]),
                run_id=str(source.parent.relative_to(reports)),selected_step=result['best_step'],
                ValidAtK=metrics['valid_rate'],AnyValidAtK=metrics['any_valid'],UniqueValidAtK=metrics['unique_valid'],
                ReferenceCoverageAtK=metrics['reference_coverage'],training_exposures=result['trajectory_exposures'],
                training_threads=config['threads'],sampling_repeats=len(entry['repeats']),budget_status=entry['budget_status'],
                forward_passes=config['sampling_steps'],end_to_end_controlled_ms=timing.get('request_generate_transfer_check_ms_p50'),
                elapsed_s=result['elapsed_s'],gpu_hours=result['gpu_hours_reserved'],code_commit=config['code_commit'],
                cumulative_elapsed_s=result.get('cumulative_elapsed_s',result['elapsed_s']),
                cumulative_gpu_hours=result.get('cumulative_gpu_hours_reserved',result['gpu_hours_reserved']),
                incremental_training_exposures=result.get('incremental_trajectory_exposures',result['trajectory_exposures']),
                cost_scope='elapsed/gpu_hours are this output tree; cumulative includes prior continuation once; exposures cumulative; repeated across K, deduplicate by run_id; no VLM/scorer',
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
    for source in sorted(reports.glob('constraint_update_v*/training/seed*/*/summary.json')):
        result=json.loads(source.read_text()); config=json.loads((source.parent/'config.json').read_text())
        for budget,metrics in result['metrics'].items():
            if not isinstance(metrics,dict) or 'total_candidates' not in metrics:
                continue
            rows.append(dict(tier='controlled_constraint_change',task='one_physical_opening_closed',
                protocol='equal_parent_cumulative_old4_plus_new_budget',method=config['arm'],seed=config['seed'],
                split='DEV_MODEL',K=metrics['total_candidates'],run_id=str(source.parent.relative_to(reports)),
                ValidAtK=metrics['union_valid_rate'],AnyValidAtK=metrics['union_any_valid'],
                UniqueValidAtK=metrics['union_unique_valid'],ReferenceCoverageAtK=metrics['union_reference_coverage'],
                additional_unique_valid=metrics['additional_unique_valid'],new_valid_rate=metrics['new_valid_rate'],
                new_candidates=metrics['new_candidates'],supplied_candidates=4,
                training_exposures=result['trajectory_exposures'],training_threads=config['threads'],
                incremental_training_exposures=result.get('incremental_trajectory_exposures',result['trajectory_exposures']),
                elapsed_s=result['elapsed_s'],gpu_hours=result['gpu_hours_reserved'],
                cumulative_elapsed_s=result.get('cumulative_elapsed_s',result['elapsed_s']),
                cumulative_gpu_hours=result.get('cumulative_gpu_hours_reserved',result['gpu_hours_reserved']),
                cost_scope='elapsed/gpu_hours are incremental output-tree cost; cumulative includes prior continuation; shared old4 producer cost separate; deduplicate budgets by run_id',
                code_commit=config['code_commit'],source=str(source.relative_to(root))))
    prototype_sources = sorted((reports/'observation_prototype_v1').glob('*.json'))
    prototype_sources += sorted(reports.glob('observation_prototype_obstacle_*/report.json'))
    prototype_sources += sorted((reports/'observation_prototype_fresh_dev_v1').glob('fixed_old64.json'))
    for source in prototype_sources:
        result=json.loads(source.read_text())
        obstacle = 'obstacle_' in source.parent.name
        fresh = 'fresh_dev' in source.parent.name
        run_id = str(source.parent.relative_to(reports))+'/'+source.stem
        rows.append(dict(tier='observed_RGBD_task_specific_endpoint',
            task='RLBench_derived_obstacle_reach_three_targets' if obstacle else ('RLBench_derived_reach_three_targets_fresh16DEV' if fresh else 'RLBench_derived_reach_three_targets'),
            protocol=result['evaluation_protocol'],method='TRAIN_color_prototype_'+(source.parent.name if obstacle else source.stem),split='DEV_MODEL',K=1,
            run_id=run_id,semantic_goal_accuracy=result['semantic_goal_accuracy'],
            endpoint_error_m=result['reference_endpoint_error_m_conditional_on_prediction'],
            center_endpoint_error_m=result['goal_error_m_conditional_on_prediction'],
            reference_evaluation_examples=result['reference_evaluation_examples'],semantic_evaluation_examples=result['semantic_evaluation_examples'],
            cpu_endpoint_ms=result['cpu_latency_ms']['median'],cost_scope='one endpoint only; no path, Qwen, scorer or execution; reference and center error denominators differ',
            source=str(source.relative_to(root))))
    for source in sorted(reports.glob('observation_astar_obstacle_new32_v*/dev_model/report.json')):
        result=json.loads(source.read_text())
        metrics=json.loads((source.parent/'tip_evaluation/metrics.json').read_text())
        aggregate=source.parent.parent/'aggregate_analysis.json'
        analysis=json.loads(aggregate.read_text()) if aggregate.exists() else {}
        rows.append(dict(tier='observed_RGBD_closed_instruction_planner',task='RLBench_derived_obstacle_reach_three_targets',
            protocol=metrics['evaluation_protocol'],method=result['baseline'],seed=None,split='DEV_MODEL',K=metrics['candidates'],
            run_id=str(source.parent.relative_to(reports)),TipValidAtK=metrics['TipValidAtK'],
            AnyTipValidAtK=metrics['AnyTipValidAtK'],UniqueClassifiedTipValidAtK=metrics['UniqueClassifiedTipValidAtK'],
            KnownReferenceTypeCoverageAtK=metrics['KnownReferenceTypeCoverageAtK'],TipClearAtK=metrics['TipClearAtK'],
            semantic_goal_accuracy=metrics['semantic_goal_accuracy'],AnySemanticGoalAtK=metrics['AnySemanticGoalAtK'],
            center_endpoint_error_m=metrics['endpoint_error_m'],semantic_evaluation_examples=metrics['examples'],
            observed_RGBD_plan_proxycheck_ms=analysis.get('request_seconds_quantiles',{}).get('median',0)*1000 if analysis else None,
            code_commit=analysis.get('source_release'),
            cost_scope='current RGBD + closed TRAIN instruction prototype + exactly K searches + observed proxy checks; failed slots retained; no Qwen/scorer/full robot execution',
            source=str(source.relative_to(root))))
    tip_sources = sorted(reports.glob('obstacle_new*_evaluation/tip_evaluation_v1/*/dev_model/metrics.json'))
    tip_sources += sorted(reports.glob('obstacle_new*_evaluation/hard_tip_evaluation_v1/dev_model/metrics.json'))
    tip_sources += sorted(reports.glob('obstacle_fixed_step1000_evaluation/tip_evaluation_v1/*/metrics.json'))
    tip_sources += sorted(reports.glob('obstacle_original_best_evaluation/original_best_tip_v1/*/metrics.json'))
    for source in tip_sources:
        metrics=json.loads(source.read_text())
        fixed_step = 'obstacle_fixed_step1000_evaluation' in source.parts
        original_all_seeds = 'obstacle_original_best_evaluation' in source.parts
        method = source.parent.name if fixed_step or original_all_seeds else source.parent.parent.name
        seed = int(method.rsplit('_seed',1)[1]) if fixed_step or original_all_seeds else 0
        rows.append(dict(tier='observed_RGBD_box_tip_check',task='RLBench_derived_obstacle_reach_three_targets',
            protocol=metrics['evaluation_protocol'],method=method,seed=seed,split='DEV_MODEL',K=metrics['candidates'],
            checkpoint_selection_protocol='fixed_step_1000' if fixed_step else 'original_DEV_ADE_best',
            run_id=str(source.parent.relative_to(reports)),TipValidAtK=metrics['TipValidAtK'],
            AnyTipValidAtK=metrics['AnyTipValidAtK'],UniqueClassifiedTipValidAtK=metrics['UniqueClassifiedTipValidAtK'],
            KnownReferenceTypeCoverageAtK=metrics['KnownReferenceTypeCoverageAtK'],TipClearAtK=metrics['TipClearAtK'],
            semantic_goal_accuracy=metrics['semantic_goal_accuracy'],AnySemanticGoalAtK=metrics['AnySemanticGoalAtK'],
            center_endpoint_error_m=metrics['endpoint_error_m'],semantic_evaluation_examples=metrics['examples'],
            cost_scope='saved-prediction evaluation; training cost in corresponding neural run; box-only tip checks exclude arm/table/execution',
            source=str(source.relative_to(root))))
    observation_folders = ['observed_frozen_v1','observed_online_v1','observed_online_warm_v2','observed_geometry_v1',
                           'observed_geometry_grounding_v2','observed_geometry_seeds_v2']
    observation_folders += [p.name for p in reports.glob('observed_learning_curve_*') if p.is_dir()]
    observation_folders += [p.name for p in reports.glob('observed_obstacle_*') if p.is_dir()]
    observation_folders += [p.name for p in reports.glob('observed_online_geometry_*') if p.is_dir()]
    observation_folders += [p.name for p in reports.glob('observed_anchor_*') if p.is_dir()]
    observation_folders += [p.name for p in reports.glob('observed_natural_reserved*') if p.is_dir()]
    for folder in observation_folders:
        for source in sorted((reports/folder).rglob('summary.json')):
            result=json.loads(source.read_text()); metrics=result['metrics']
            config=json.loads((source.parent/'config.json').read_text())
            rows.append(dict(tier='observed_RGBD_language_current' if 'geometry_role' in config or 'geometry_parameters' in config else 'observed_RGB_language_current',
                task='RLBench_derived_obstacle_reach_three_targets' if 'obstacle' in str(source.parent.relative_to(reports)) else 'RLBench_derived_reach_three_targets',
                protocol=metrics.get('evaluation_protocol','observation_eval_v1_reference_subset'),
                checkpoint_selection_protocol='original_DEV_ADE_best',
                method=folder+'_'+'_'.join(source.parent.relative_to(reports/folder).parts)+'_'+config.get('adapter_mode','frozen_cache'),
                seed=config['seed'],split='DEV_MODEL',K=config['candidates'],
                run_id=str(source.parent.relative_to(reports)),training_exposures=result['trajectory_exposures'],
                common_pretraining_exposures=result.get('common_pretraining',{}).get('total_trajectory_exposures'),
                common_pretraining_gpu_hours=result.get('common_pretraining',{}).get('gpu_hours_reserved'),
                selected_step=result['best_step'],grounding_weight=config.get('grounding_weight'),
                anchor_mode=config.get('anchor_mode','soft') if 'geometry_parameters' in config or 'geometry_role' in config else None,
                training_threads=config['threads'],reference_evaluation_examples=metrics.get('reference_evaluation_examples',metrics['examples']),
                semantic_evaluation_examples=metrics['semantic_evaluation_examples'],
                cost_scope=('online Qwen/head train and evaluation; gpu_hours includes setup; elapsed_s excludes setup; common head pretraining separate'
                    if 'online' in folder else 'head training/evaluation only; separate Qwen encoding is excluded'),
                candidate_ADE_m=metrics['candidate_matched_ADE_m'],endpoint_error_m=metrics['candidate_endpoint_error_m'],
                semantic_goal_accuracy=metrics['semantic_goal_accuracy'],AnySemanticGoalAtK=metrics['AnySemanticGoalAtK'],
                RGBD_Qwen_generation_ms=metrics.get('online_request_ms_median'),
                elapsed_s=result['elapsed_s'],gpu_hours=result['gpu_hours_reserved'],code_commit=config['code_commit'],
                source=str(source.relative_to(root))))
    for source in sorted((reports/'observed_natural_fresh_dev_v1').glob('*/metrics.json')):
        metrics=json.loads(source.read_text()); provenance=json.loads((source.parent/'provenance.json').read_text())
        rows.append(dict(tier='observed_RGBD_language_current',task='RLBench_derived_reach_three_targets_fresh16DEV',
            protocol=metrics['evaluation_protocol'],checkpoint_selection_protocol=provenance['checkpoint_selection_protocol'],
            method=provenance['method'],seed=provenance['seed'],split='DEV_MODEL',K=metrics['candidates'],
            run_id=provenance['source_training_run'],selected_step=metrics['checkpoint_step'],
            candidate_ADE_m=metrics['candidate_matched_ADE_m'],endpoint_error_m=metrics['candidate_endpoint_error_m'],
            semantic_goal_accuracy=metrics['semantic_goal_accuracy'],AnySemanticGoalAtK=metrics['AnySemanticGoalAtK'],
            reference_evaluation_examples=metrics['reference_evaluation_examples'],semantic_evaluation_examples=metrics['semantic_evaluation_examples'],
            cost_scope='unchanged old64 best checkpoint transfer to fresh16 DEV; no additional training or new selection; not locked TEST',
            code_commit=provenance['source_training_commit'],source=str(source.relative_to(root))))
    for source in sorted((reports/'observed_anchor_fixed_step1000_v1').glob('*/*/metrics.json')):
        metrics=json.loads(source.read_text()); provenance=json.loads((source.parent/'provenance.json').read_text())
        rows.append(dict(tier='observed_RGBD_language_current',
            task='RLBench_derived_obstacle_reach_three_targets' if provenance['setting']=='obstacle32' else 'RLBench_derived_reach_three_targets',
            protocol=metrics['evaluation_protocol'],checkpoint_selection_protocol='fixed_step_1000',
            method=provenance['method'],seed=provenance['seed'],split='DEV_MODEL',K=metrics['candidates'],
            run_id=provenance['training_run'],selected_step=1000,
            candidate_ADE_m=metrics['candidate_matched_ADE_m'],endpoint_error_m=metrics['candidate_endpoint_error_m'],
            semantic_goal_accuracy=metrics['semantic_goal_accuracy'],AnySemanticGoalAtK=metrics['AnySemanticGoalAtK'],
            reference_evaluation_examples=metrics['reference_evaluation_examples'],semantic_evaluation_examples=metrics['semantic_evaluation_examples'],
            cost_scope='uniform last-checkpoint exploratory sensitivity; same runs as primary best rows, no additional training',
            code_commit=provenance['training_source_commit'],source=str(source.relative_to(root))))
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
          'extra_training_draft_forwards','forward_passes','cost_scope','elapsed_s','gpu_hours',
          'cumulative_elapsed_s','cumulative_gpu_hours','incremental_training_exposures','center_endpoint_error_m','cpu_endpoint_ms',
          'common_pretraining_exposures','common_pretraining_gpu_hours','anchor_mode','RGBD_Qwen_generation_ms','checkpoint_selection_protocol',
          'sampling_repeats','budget_status','observed_RGBD_plan_proxycheck_ms',
          'TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','KnownReferenceTypeCoverageAtK','TipClearAtK','code_commit','source']
    rows=[{key:row.get(key) for key in keys} for row in rows]
    (reports/'MAIN_RESULTS.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    with (reports/'MAIN_RESULTS.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
    print('Wrote %d measured result rows; null means unmeasured.'%len(rows))


if __name__=='__main__':main()
