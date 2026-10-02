"""Audit matched local/global refinement, retaining draft and final states."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

from scripts.analyze_observed_obstacle_pair import (PAIR_FIELDS,SOURCE_FILES,read,write,require_close,state_digest,load_metadata,prepare_dev,evaluate_saved,audit_planner)
from scripts.analyze_observed_obstacle_pair import paired_config as ordinary_paired_config
from scripts.export_observation_roles import digest
from scripts.evaluate_observed_obstacles import scene_metrics,PROTOCOL
from scripts.diagnose_observed_train_geometry import first_collision,summarize

def paired_config(config,refinement=True):
    value=ordinary_paired_config(config)
    if refinement:value.pop('refinement_mode',None)
    return value


def validate_reference_peak(run,config,summary):
    if config.get('refinement_mode','none')!='none' or config.get('anchor_mode')!='straight_through_peak':
        raise ValueError('reference must be original one-pass peak, not soft or refined')
    for stage,folder,prefix in (('best','dev_model',''),('last','last_dev_model','last_')):
        checkpoint_key='best_checkpoint_sha256' if stage=='best' else 'last_checkpoint_sha256'
        if digest(run/(stage+'.pt'))!=summary[checkpoint_key] or digest(run/folder/'predictions.npz')!=summary[prefix+'prediction_sha256']:
            raise ValueError('original peak checkpoint/prediction SHA differs from summary')


def evaluate_drafts(prediction_file,rows,dataset,current,geometry,labels,output):
    """Evaluate already saved complete drafts, with unchanged event predictions."""
    from scripts.train_observed_routes import observation_metrics
    with np.load(prediction_file,allow_pickle=False) as archive:
        ids=list(map(str,archive['scene_ids']))
        order=[ids.index(str(identifier)) for identifier in dataset['scene_ids']]
        drafts,events=archive['draft_paths'][order],archive['gripper_open'][order]
    metric,scenes=observation_metrics(drafts,events,dataset,np.arange(len(rows)))
    candidates=[];tip_rows=[]
    for index,row in enumerate(rows):
        identifier=row['id'];label=labels[identifier]
        tip,items=scene_metrics(drafts[index],events[index],current[identifier],geometry[identifier],
            label['semantic_targets'],label.get('route_types',[]),clearance=.02)
        scenes[index]['tip_evaluation']=tip;tip_rows.append(tip)
        for candidate,item in enumerate(items):
            candidates.append(dict(scene_id=identifier,parent_id=row['parent_id'],has_reference=bool(label['routes']),
                first_collision=first_collision(drafts[index,candidate],geometry[identifier]),**item))
    for key in PAIR_FIELDS:
        if key in tip_rows[0]:
            values=[row[key] for row in tip_rows if row[key] is not None]
            metric[key]=float(np.mean(values)) if values else None
    metric.update(tip_evaluation_protocol=PROTOCOL,tip_evaluation_examples=len(rows),
        total_submitted_candidate_budget=len(rows)*drafts.shape[1],
        selection_scope='diagnostic saved pre-update drafts; never used to replace original final-path checkpoint selection',
        saved_prediction_sha256=digest(prediction_file),saved_array_field='draft_paths')
    failure=summarize(candidates)
    output.mkdir(parents=True,exist_ok=False)
    for name,value in (('metrics',metric),('per_scene',scenes),('per_candidate',candidates),('failure_breakdown',failure)):
        write(output/(name+'.json'),value)
    return dict(metrics=metric,failure_breakdown=failure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--training-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--expected-dev-examples', type=int, default=24)
    parser.add_argument('--expected-dev-parents', type=int, default=8)
    parser.add_argument('--planner', type=Path, help='optional completed same-data observed A* report directory')
    parser.add_argument('--comparison',choices=('refinement',),default='refinement')
    parser.add_argument('--reference-peak',type=Path,help='completed lower-cost original one-pass peak run; never a strict matched-generation-count control')
    args = parser.parse_args()
    import torch
    from routeset.common import seed_all
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from routeset.observed_path_refinement import refinement_config
    torch.set_num_threads(1)
    started = time.perf_counter()
    if args.output.exists():
        raise ValueError('fresh analysis output required')
    source_hashes = {}
    sources=SOURCE_FILES+(('routeset/observed_path_refinement.py',) if args.comparison=='refinement' else ())
    methods=('global','local') if args.comparison=='refinement' else ('soft','peak')
    baseline,alternative=methods
    for relative in sources:
        source, local = args.training_source/relative, Path(__file__).resolve().parents[1]/relative
        if digest(source) != digest(local):
            raise ValueError('initialization replay source differs: '+relative)
        source_hashes[relative] = digest(source)
    observations, labels, metadata = load_metadata(args.data)
    train_ids = np.asarray([i for i, row in enumerate(observations) if row['split']=='TRAIN' and labels[row['id']]['routes']])
    configs, checkpoints, summaries, histories, init_hashes, samplers, training_hashes = {}, {}, {}, {}, {}, {}, {}
    for method in methods:
        mode='soft' if method=='soft' else 'straight_through_peak'
        folder = args.runs/(method+'_seed'+str(args.seed)); config = read(folder/'config.json')
        summary, history, status = read(folder/'summary.json'), read(folder/'history.json'), read(folder/'status.json')
        if status != dict(status='completed', step=config['steps'], exit_code=0):
            raise ValueError('training did not complete')
        if config['anchor_mode'] != mode or config['seed'] != args.seed or config['checkpoint_selection'] != 'tip_unique_valid':
            raise ValueError('unexpected arm/seed/selection protocol')
        expected_refinement=method if args.comparison=='refinement' else 'none'
        if config.get('refinement_mode','none')!=expected_refinement:
            raise ValueError('unexpected draft refinement mode')
        if config['code_commit'] != args.training_source.name or config['source_script_sha256'] != digest(args.training_source/'scripts/train_observed_geometry.py'):
            raise ValueError('recorded immutable training source changed')
        if config['train_examples'] != len(train_ids) or config['observations'] != str(args.data/'observations.jsonl'):
            raise ValueError('training data source or supervised denominator mismatch')
        if summary['trajectory_exposures'] != config['steps']*config['batch_size']*config['candidates']:
            raise ValueError('actual candidate exposure mismatch')
        seed_all(config['seed'])
        model = ObservedGeometryRouteHead(config['feature_dim'], config['horizon'], config['candidates'], config['width'],
            config['depth'], config['point_width'], config['endpoint_residual_bound'], anchor_mode=mode,**refinement_config(config))
        init_hashes[method] = state_digest(model.state_dict())
        if model.active_parameter_count() != summary['parameters']:
            raise ValueError('model size differs from recorded training')
        sampler = np.random.default_rng(config['seed']+100000); stream = hashlib.sha256()
        for _ in range(config['steps']):
            stream.update(sampler.choice(train_ids, config['batch_size'], replace=True).astype('<i8').tobytes())
        samplers[method] = dict(reconstructed_index_stream_sha256=stream.hexdigest(), final_state=sampler.bit_generator.state)
        best, last = [torch.load(folder/(stage+'.pt'), map_location='cpu', weights_only=False) for stage in ('best', 'last')]
        if last['step'] != config['steps'] or last['sampler_state'] != sampler.bit_generator.state:
            raise ValueError('actual final sampler/step differs from exact replay')
        expected_best = max(history, key=lambda item: item['dev_model']['selection_score'])
        if best['step'] != expected_best['step'] or best['step'] != summary['best_step']:
            raise ValueError('best differs from original first-max DEV selection')
        for checkpoint in (best, last):
            if checkpoint['config'] != config or checkpoint['trajectory_exposures'] != checkpoint['step']*config['batch_size']*config['candidates']:
                raise ValueError('actual checkpoint provenance/exposure mismatch')
        configs[method], summaries[method], histories[method] = config, summary, history
        training_hashes[method] = read(folder/'source_hashes.json')
        checkpoints[method] = dict(best=best['step'], last=last['step'])
    if paired_config(configs[baseline],args.comparison=='refinement') != paired_config(configs[alternative],args.comparison=='refinement'):
        raise ValueError('paired config differs beyond anchor/output/measured preprocessing time')
    if init_hashes[baseline] != init_hashes[alternative] or samplers[baseline] != samplers[alternative]:
        raise ValueError('reconstructed initialization or actual sampler audit differs')
    if training_hashes[baseline] != training_hashes[alternative]:
        raise ValueError('actual recorded training source files differ')
    rows, dataset, current, geometry, evaluation_hashes = prepare_dev(args.data, observations, labels, configs[baseline]['horizon'])
    if len(rows) != args.expected_dev_examples or len(set(dataset['parent_ids'])) != args.expected_dev_parents:
        raise ValueError('expected full DEV size/parents differs')
    recorded_sources = {str(Path(name).resolve()): sha for name,sha in training_hashes[baseline].items()}
    for method in methods:
        for stage in ('metrics','last_metrics'):
            recorded_sources.update({str(Path(name).resolve()):sha for name,sha in summaries[method][stage]['tip_geometry_label_source_sha256'].items()})
    for path, sha in evaluation_hashes.items():
        if recorded_sources.get(path) != sha:
            raise ValueError('post-hoc DEV source changed since training/evaluation: '+path)
    args.output.mkdir(parents=True)
    results = {}; all_rows = {}; artifact_index = {}
    for method in methods:
        run = args.runs/(method+'_seed'+str(args.seed)); summary = summaries[method]
        artifacts = {str(p): dict(sha256=digest(p), bytes=p.stat().st_size) for p in sorted(run.rglob('*')) if p.suffix in ('.pt', '.npz')}
        for stage, subfolder, prefix in (('best', 'dev_model', ''), ('last', 'last_dev_model', 'last_')):
            metric, scenes, failure = evaluate_saved(run/subfolder, rows, dataset, current, geometry, labels,
                configs[method]['candidates'], args.output/(method+'_'+stage))
            for key in PAIR_FIELDS:
                require_close(metric[key], summary['metrics' if stage=='best' else 'last_metrics'][key], method+' '+stage+' summary '+key)
            checkpoint_key = 'best_checkpoint_sha256' if stage=='best' else 'last_checkpoint_sha256'
            if artifacts[str(run/(stage+'.pt'))]['sha256'] != summary[checkpoint_key] or artifacts[str(run/subfolder/'predictions.npz')]['sha256'] != summary[prefix+'prediction_sha256']:
                raise ValueError('actual checkpoint/prediction SHA differs from recorded summary')
            key = method+'_'+stage
            results[key] = dict(step=checkpoints[method][stage], metrics=metric, failure_breakdown=failure,
                actual_checkpoint=str(run/(stage+'.pt')), actual_prediction=str(run/subfolder/'predictions.npz'))
            if args.comparison=='refinement':
                budget=summary['metrics' if stage=='best' else 'last_metrics']['generation_budget']
                k=configs[method]['candidates']
                if (budget['final_candidates']!=k or budget['draft_complete_paths']!=k or budget['updates']!=1 or
                        budget['total_complete_path_states']!=2*k or summary['complete_path_state_exposures']!=2*summary['trajectory_exposures']):
                    raise ValueError('draft/final actual generation budget differs')
                with np.load(run/subfolder/'predictions.npz',allow_pickle=False) as saved:
                    if saved['draft_paths'].shape!=saved['paths'].shape or not np.array_equal(saved['draft_paths'][:,:,[0,-1]],saved['paths'][:,:,[0,-1]]):
                        raise ValueError('complete drafts missing or update altered start/endpoint')
                results[key]['generation_budget']=budget
                results[key]['draft']=evaluate_drafts(run/subfolder/'predictions.npz',rows,dataset,current,geometry,labels,
                    args.output/(method+'_'+stage+'_draft'))
                draft_metric=results[key]['draft']['metrics']
                results[key]['final_minus_draft']={field:metric[field]-draft_metric[field] if metric[field] is not None else None for field in PAIR_FIELDS}
            all_rows[key] = {r['scene_id']: r for r in scenes}
        artifact_index[method] = dict(binaries=artifacts, config_sha256=digest(run/'config.json'),
            summary_sha256=digest(run/'summary.json'), history_sha256=digest(run/'history.json'),
            training_sources_sha256=digest(run/'source_hashes.json'))
    paired = {}
    for stage in ('best', 'last'):
        differences = []
        for row in rows:
            a, b = [all_rows[method+'_'+stage][row['id']] for method in methods]
            delta = {}
            for key in PAIR_FIELDS:
                x = a['tip_evaluation'].get(key, a.get(key)); y = b['tip_evaluation'].get(key, b.get(key))
                delta[key] = None if x is None or y is None else y-x
            differences.append(dict(scene_id=row['id'], parent_id=row['parent_id'], reference_count=a['reference_count'], **{alternative+'_minus_'+baseline:delta}))
        parent_differences = []
        for parent in sorted(set(dataset['parent_ids'])):
            values = [r for r in differences if r['parent_id']==parent]
            delta_key=alternative+'_minus_'+baseline
            delta = {key: float(np.mean([r[delta_key][key] for r in values if r[delta_key][key] is not None]))
                     if any(r[delta_key][key] is not None for r in values) else None for key in PAIR_FIELDS}
            parent_differences.append(dict(parent_id=str(parent), instructions=len(values), **{delta_key:delta}))
        paired[stage] = dict(per_instruction=differences, per_parent=parent_differences,
            **{'aggregate_'+alternative+'_minus_'+baseline:{key: results[alternative+'_'+stage]['metrics'][key]-results[baseline+'_'+stage]['metrics'][key]
                if results[alternative+'_'+stage]['metrics'][key] is not None else None for key in PAIR_FIELDS}})
    result = dict(protocol='observed_refinement_pair_audit_v1', evaluation_protocol='observation_eval_v2', tip_evaluation_protocol=PROTOCOL,
        seed=args.seed, selection_protocol='dev_tip_unique_valid_v1', criterion='UniqueClassifiedTipValidAtK + 0.05 * TipValidAtK',
        comparison=args.comparison,common_candidate_slots=summaries[baseline]['trajectory_exposures'],
        common_complete_path_state_exposures=summaries[baseline].get('complete_path_state_exposures',summaries[baseline]['trajectory_exposures']),
        metadata=metadata, results=results, paired=paired,
        actual_artifact_index=artifact_index, training_source_hashes=source_hashes, evaluation_file_hashes=evaluation_hashes,
        initialization=dict(reconstructed_state_sha256=init_hashes, evidence='Exact immutable-source/seed reconstruction; initial weights were not saved during training.'),
        sampler_audit=samplers, cost={method:{key:summaries[method][key] for key in ('elapsed_s','gpu_hours_reserved','peak_cuda_memory_mb','data_load_preprocess_s','latency')} for method in summaries},
        limitations='One training seed; all parents are DEV_MODEL. Tip checks certify only added boxes at original2cm; no arm/table/IK/execution or core novelty claim.',
        script_sha256=digest(__file__), analysis_cpu_wall_s=time.perf_counter()-started)
    if args.reference_peak is not None:
        reference_config=read(args.reference_peak/'config.json');reference_summary=read(args.reference_peak/'summary.json')
        for key in ('dataset_fingerprint','candidates','horizon','train_examples','dev_examples','steps','batch_size','seed','checkpoint_selection'):
            if reference_config[key]!=configs[baseline][key]:raise ValueError('lower-cost reference source differs: '+key)
        validate_reference_peak(args.reference_peak,reference_config,reference_summary)
        reference={}
        for stage,folder in (('best','dev_model'),('last','last_dev_model')):
            metrics,scenes,failure=evaluate_saved(args.reference_peak/folder,rows,dataset,current,geometry,labels,
                configs[baseline]['candidates'],args.output/('lower_cost_peak_'+stage))
            summary_metrics=reference_summary['metrics' if stage=='best' else 'last_metrics']
            for field in PAIR_FIELDS:require_close(metrics[field],summary_metrics[field],'lower-cost '+field)
            reference[stage]=dict(metrics=metrics,failure_breakdown=failure,
                checkpoint_sha256=digest(args.reference_peak/(stage+'.pt')),
                prediction_sha256=digest(args.reference_peak/folder/'predictions.npz'))
        result['lower_cost_original_peak']=dict(stages=reference,parameters=reference_summary['parameters'],
            complete_path_states_per_request=configs[baseline]['candidates'],
            training_complete_path_state_exposures=reference_summary['trajectory_exposures'],
            comparison='Same data and final K only. Original peak has no draft update, fewer parameters and half as many complete path states; not a strict fair-generation-count control.')
    if args.planner is not None:
        result['traditional_planner'] = audit_planner(args.planner,rows,current,geometry,labels,
            {row['parent_id'] for row in observations if row['split']=='TRAIN' and labels[row['id']]['routes']},
            metadata,configs[baseline]['candidates'])
        if args.comparison=='refinement':
            result['traditional_planner']['generation_budget_caveat']='Planner emits4 complete paths; updated neural model emits4 drafts+4 finals. This is a lower-state-budget reference, not matched complete-generation count.'
    write(args.output/'analysis.json', result)
    write(args.output/'actual_predictions_index.json', artifact_index)
    print(json.dumps({key:dict(step=value['step'],metrics={field:value['metrics'][field] for field in PAIR_FIELDS}) for key,value in results.items()},indent=2))


if __name__ == '__main__':
    main()
