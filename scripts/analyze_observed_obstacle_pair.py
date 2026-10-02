"""Audit a completed same-budget soft/peak obstacle pair from saved predictions.

No model inference, fitting, candidate filtering or geometry repair is performed.
DEV_MODEL inputs with zero references remain in every semantic/tip denominator.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_observation_roles import selected_rows, digest
from scripts.evaluate_observed_obstacles import scene_metrics, PROTOCOL
from scripts.diagnose_observed_train_geometry import first_collision, summarize


PAIR_FIELDS = ('TipValidAtK', 'AnyTipValidAtK', 'UniqueClassifiedTipValidAtK',
               'DuplicateClassifiedTipValidCount', 'UnknownTypeTipValidCount',
               'KnownReferenceTypeCoverageAtK', 'TipClearAtK', 'semantic_goal_accuracy',
               'AnySemanticGoalAtK', 'candidate_matched_ADE_m', 'candidate_endpoint_error_m')
SOURCE_FILES = ('routeset/observed_geometry.py', 'routeset/observed_route_head.py',
                'routeset/models.py', 'routeset/common.py', 'routeset/train_v2.py')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def paired_config(config):
    value = json.loads(json.dumps(config))
    for key in ('output', 'anchor_mode'):
        value.pop(key, None)
    geometry = value.get('geometry_preprocessing', {})
    for key in ('load_preprocess_total_s', 'image_npz_load_backproject_ms_median',
                'image_npz_load_backproject_ms_p95'):
        geometry.pop(key, None)
    return value


def require_close(actual, expected, name):
    if actual is None or expected is None:
        if actual is not expected:
            raise ValueError('null mismatch: '+name)
    elif not np.isclose(actual, expected, rtol=0, atol=1e-7):
        raise ValueError('recomputed metric mismatch: '+name)


def state_digest(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        array = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(array.dtype).encode())
        h.update(str(tuple(array.shape)).encode())
        h.update(array.reshape(-1).view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()


def load_metadata(data):
    exported = read(data/'export_manifest.json')
    if set(exported['requested_roles']) != {'TRAIN', 'DEV_MODEL'}:
        raise ValueError('only explicit development export may be analyzed')
    for filename, key in (('observations.jsonl', 'input_manifest_sha256'),
                          ('supervision.jsonl', 'supervision_manifest_sha256')):
        if digest(data/filename) != exported[key]:
            raise ValueError('export manifest changed: '+filename)
    parents = set(exported['requested_parent_ids'])
    observations = selected_rows(data/'observations.jsonl', parents)
    labels = {r['id']: r for r in selected_rows(data/'supervision.jsonl', parents)}
    if len(labels) != len(observations) or len({r['id'] for r in observations}) != len(observations):
        raise ValueError('duplicate or missing supervision')
    roles = {}
    for row in observations:
        parent, role = row['parent_id'], row['split']
        if set(row) != {'id', 'parent_id', 'split', 'image', 'instruction'} or role not in {'TRAIN', 'DEV_MODEL'}:
            raise ValueError('input contract or development role violated')
        if roles.setdefault(parent, role) != role:
            raise ValueError('parent leakage')
        if labels[row['id']]['parent_id'] != parent or labels[row['id']]['split'] != role:
            raise ValueError('label identity/role mismatch')
    metadata_path = Path(exported['source_dataset'])/'manifest.json'
    if read(metadata_path)['acceptance']['tip_polyline_clearance_m'] != .02:
        raise ValueError('original 2cm clearance required')
    return observations, labels, dict(export_sha256=digest(data/'export_manifest.json'),
        observations_sha256=digest(data/'observations.jsonl'), supervision_sha256=digest(data/'supervision.jsonl'),
        collection_settings_sha256=digest(metadata_path), reservation_sha256=exported['reservation_sha256'])


def prepare_dev(data, observations, labels, horizon):
    from routeset.observed_route_head import resample_event_segments
    rows = [row for row in observations if row['split'] == 'DEV_MODEL']
    width = max(1, max(len(labels[row['id']]['routes']) for row in rows))
    refs = np.zeros((len(rows), width, horizon, 3), dtype=np.float32)
    events = np.zeros((len(rows), width, horizon), dtype=np.float32)
    mask = np.zeros((len(rows), width), dtype=bool)
    current, geometry, hashes = {}, {}, {}
    for index, row in enumerate(rows):
        identifier, parent = row['id'], row['parent_id']; label = labels[identifier]
        if label['semantic_targets']['tolerance'] != .03:
            raise ValueError('original strict 3cm semantic criterion required')
        for field in ('observation', 'verification_only'):
            path = (data/label[field]).resolve()
            if path.parent.name != parent:
                raise ValueError('DEV artifact resolves outside its selected parent')
            hashes[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                if field == 'observation':
                    # Match the current-state precision used by the original trainer.
                    current[identifier] = {key: archive[key].astype(np.float32) for key in ('gripper_pose', 'gripper_open')}
                else:
                    geometry[identifier] = {key: archive[key] for key in ('obstacle_centers', 'obstacle_halfsizes')}
        for slot, filename in enumerate(label['routes']):
            path = (data/filename).resolve()
            if path.parent.name != parent:
                raise ValueError('reference resolves outside selected DEV parent')
            hashes[str(path)] = digest(path)
            with np.load(path, allow_pickle=False) as archive:
                refs[index, slot], events[index, slot] = resample_event_segments(archive['gripper_pose'], archive['gripper_open'], horizon)
            mask[index, slot] = True
    dataset = dict(scene_ids=np.asarray([r['id'] for r in rows]), parent_ids=np.asarray([r['parent_id'] for r in rows]),
                   paths=refs, events=events, path_mask=mask, semantic_targets=[labels[r['id']]['semantic_targets'] for r in rows])
    return rows, dataset, current, geometry, hashes


def evaluate_saved(folder, rows, dataset, current, geometry, labels, expected_k, output):
    from scripts.train_observed_routes import observation_metrics
    with np.load(folder/'predictions.npz', allow_pickle=False) as archive:
        ids = list(map(str, archive['scene_ids'])); parent_ids = list(map(str, archive['parent_ids']))
        paths, events = archive['paths'], archive['gripper_open']
    expected_ids = list(map(str, dataset['scene_ids']))
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError('all DEV inputs including zero-reference rows required exactly once')
    order = [ids.index(identifier) for identifier in expected_ids]
    paths, events = paths[order], events[order]
    if [parent_ids[i] for i in order] != list(map(str, dataset['parent_ids'])):
        raise ValueError('prediction parent identity mismatch')
    if paths.shape != (len(rows), expected_k, dataset['paths'].shape[2], 3) or events.shape != paths.shape[:-1]:
        raise ValueError('candidate budget/horizon changed')
    metrics, per_scene = observation_metrics(paths, events, dataset, np.arange(len(rows)))
    candidate_rows = []; tip_rows = []
    for index, row in enumerate(rows):
        identifier = row['id']; label = labels[identifier]
        tip, candidates = scene_metrics(paths[index], events[index], current[identifier], geometry[identifier],
            label['semantic_targets'], label.get('route_types', []), clearance=.02)
        require_close(tip['semantic_goal_accuracy'], per_scene[index]['semantic_goal_accuracy'], 'semantic agreement')
        per_scene[index].update(tip_evaluation=tip, instruction=row['instruction'])
        tip_rows.append(tip)
        for candidate, result in enumerate(candidates):
            result.update(scene_id=identifier, parent_id=row['parent_id'], has_reference=bool(label['routes']),
                          first_collision=first_collision(paths[index, candidate], geometry[identifier]))
            candidate_rows.append(result)
    for key in PAIR_FIELDS:
        if key in tip_rows[0]:
            values = [r[key] for r in tip_rows if r[key] is not None]
            metrics[key] = float(np.mean(values)) if values else None
    metrics.update(tip_evaluation_protocol=PROTOCOL, tip_evaluation_examples=len(rows),
                   total_submitted_candidate_budget=len(rows)*expected_k,
                   selection_score=metrics['UniqueClassifiedTipValidAtK']+.05*metrics['TipValidAtK'],
                   reference_type_coverage_examples=sum(r['known_reference_types'] > 0 for r in tip_rows))
    recorded = read(folder/'metrics.json')
    for key in PAIR_FIELDS+('examples', 'parents', 'reference_evaluation_examples', 'semantic_evaluation_examples', 'selection_score'):
        require_close(metrics[key], recorded[key], key)
    collisions = [r for r in candidate_rows if r['first_collision'] is not None]
    valid_rows = [r for r in candidate_rows if r['TipValid']]
    categories = Counter(('semantic_correct' if r['semantic_goal_correct'] else 'semantic_wrong',
                          'tip_clear' if r['tip_segments_clear'] else 'tip_collision_or_nonfinite') for r in candidate_rows)
    failure = summarize(candidate_rows)
    failure.update(semantic_geometry_cross_counts={'/'.join(key): value for key, value in categories.items()},
        semantic_correct_collisions=sum(r['semantic_goal_correct'] for r in collisions),
        semantic_correct_first_quarter_collisions=sum(r['semantic_goal_correct'] and r['first_collision']['normalized_arc_before_segment'] < .25 for r in collisions),
        classified_valid_route_types=dict(Counter('|'.join(r['declared_passage_type']) for r in valid_rows if r['declared_passage_type'] is not None)),
        valid_unknown_types=sum(r['declared_passage_type'] is None for r in valid_rows),
        classified_valid_duplicates=int(sum(r['DuplicateClassifiedTipValidCount'] for r in tip_rows)),
        no_reference=summarize([r for r in candidate_rows if not r['has_reference']]))
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (('metrics', metrics), ('per_scene', per_scene), ('per_candidate', candidate_rows), ('failure_breakdown', failure)):
        write(output/(name+'.json'), value)
    return metrics, per_scene, failure


def audit_planner(folder, rows, current, geometry, labels, training_parents, metadata, candidates):
    report = read(folder/'report.json'); recorded = report['fixed_tip_evaluation']
    if (report['split'] != 'DEV_MODEL' or report['subset_training_cost_diagnostic'] or
            report['privileged_inference_input'] or report['selection_or_repair'] or
            not report['scored_only_after_all_candidates_emitted'] or
            recorded['supervision_sha256'] != metadata['supervision_sha256']):
        raise ValueError('planner information/evaluation protocol differs')
    expected = {row['id']: row for row in rows}
    with np.load(folder/'predictions.npz', allow_pickle=False) as archive:
        ids, parent_ids = list(map(str, archive['scene_ids'])), list(map(str, archive['parent_ids']))
        paths, events = archive['paths'], archive['gripper_open']
    if len(ids) != len(set(ids)) or set(ids) != set(expected) or paths.shape[:2] != (len(rows), candidates):
        raise ValueError('planner must retain all DEV and exactly the same K slots')
    if digest(folder/'predictions.npz') != recorded['prediction_sha256']:
        raise ValueError('planner actual prediction hash mismatch')
    summaries, per_candidate = [], []
    for index, identifier in enumerate(ids):
        if parent_ids[index] != expected[identifier]['parent_id']:
            raise ValueError('planner parent identity mismatch')
        label = labels[identifier]
        result, items = scene_metrics(paths[index], events[index], current[identifier], geometry[identifier],
            label['semantic_targets'], label.get('route_types', []), clearance=.02)
        summaries.append(result)
        per_candidate.extend(dict(scene_id=identifier, parent_id=parent_ids[index], **row) for row in items)
    for key in PAIR_FIELDS:
        if key not in summaries[0]:
            continue
        values = [row[key] for row in summaries if row[key] is not None]
        require_close(float(np.mean(values)) if values else None, recorded[key], 'planner '+key)
    # The fitted source hashes must contain only the same exported TRAIN parents.
    fitted_paths = report['train_workspace_prior']['source_sha256']
    fitted_parents = {Path(name).parent.name for name in fitted_paths}
    if fitted_parents != training_parents:
        raise ValueError('planner fitted workspace does not use the same TRAIN parents')
    timings = [row['total_observation_to_routes_seconds'] for row in report['records']]
    return dict(baseline=report['baseline'], metrics=recorded, per_candidate=per_candidate,
        report_sha256=digest(folder/'report.json'), prediction_sha256=digest(folder/'predictions.npz'),
        binary_index={str(p):dict(sha256=digest(p),bytes=p.stat().st_size) for p in sorted(folder.glob('*.npz'))},
        training_parents=len(fitted_parents), training_seconds=report['training_seconds'],
        observation_to_routes_ms_median=float(np.median(timings)*1000),
        observation_to_routes_ms_p95=float(np.percentile(timings,95)*1000),
        total_generation_seconds=sum(timings), evaluation_seconds=report['evaluation_seconds'],
        language_scope='Exact TRAIN instruction/color prototypes only; unseen instructions abstain. No open-vocabulary claim.',
        comparison_scope='Same exported TRAIN and DEV, RGB-D/current/language and K4; different inference runtime, no equal-time claim.',
        limitation=report['limitation'])


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
    args = parser.parse_args()
    import torch
    from routeset.common import seed_all
    from routeset.observed_geometry import ObservedGeometryRouteHead
    torch.set_num_threads(1)
    started = time.perf_counter()
    if args.output.exists():
        raise ValueError('fresh analysis output required')
    source_hashes = {}
    for relative in SOURCE_FILES:
        source, local = args.training_source/relative, Path(__file__).resolve().parents[1]/relative
        if digest(source) != digest(local):
            raise ValueError('initialization replay source differs: '+relative)
        source_hashes[relative] = digest(source)
    observations, labels, metadata = load_metadata(args.data)
    train_ids = np.asarray([i for i, row in enumerate(observations) if row['split']=='TRAIN' and labels[row['id']]['routes']])
    configs, checkpoints, summaries, histories, init_hashes, samplers, training_hashes = {}, {}, {}, {}, {}, {}, {}
    for method, mode in (('soft', 'soft'), ('peak', 'straight_through_peak')):
        folder = args.runs/(method+'_seed'+str(args.seed)); config = read(folder/'config.json')
        summary, history, status = read(folder/'summary.json'), read(folder/'history.json'), read(folder/'status.json')
        if status != dict(status='completed', step=config['steps'], exit_code=0):
            raise ValueError('training did not complete')
        if config['anchor_mode'] != mode or config['seed'] != args.seed or config['checkpoint_selection'] != 'tip_unique_valid':
            raise ValueError('unexpected arm/seed/selection protocol')
        if config['code_commit'] != args.training_source.name or config['source_script_sha256'] != digest(args.training_source/'scripts/train_observed_geometry.py'):
            raise ValueError('recorded immutable training source changed')
        if config['train_examples'] != len(train_ids) or config['observations'] != str(args.data/'observations.jsonl'):
            raise ValueError('training data source or supervised denominator mismatch')
        if summary['trajectory_exposures'] != config['steps']*config['batch_size']*config['candidates']:
            raise ValueError('actual candidate exposure mismatch')
        seed_all(config['seed'])
        model = ObservedGeometryRouteHead(config['feature_dim'], config['horizon'], config['candidates'], config['width'],
            config['depth'], config['point_width'], config['endpoint_residual_bound'], anchor_mode=mode)
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
    if paired_config(configs['soft']) != paired_config(configs['peak']):
        raise ValueError('paired config differs beyond anchor/output/measured preprocessing time')
    if init_hashes['soft'] != init_hashes['peak'] or samplers['soft'] != samplers['peak']:
        raise ValueError('reconstructed initialization or actual sampler audit differs')
    if training_hashes['soft'] != training_hashes['peak']:
        raise ValueError('actual recorded training source files differ')
    rows, dataset, current, geometry, evaluation_hashes = prepare_dev(args.data, observations, labels, configs['soft']['horizon'])
    if len(rows) != args.expected_dev_examples or len(set(dataset['parent_ids'])) != args.expected_dev_parents:
        raise ValueError('expected full DEV size/parents differs')
    recorded_sources = {str(Path(name).resolve()): sha for name,sha in training_hashes['soft'].items()}
    for method in ('soft','peak'):
        for stage in ('metrics','last_metrics'):
            recorded_sources.update({str(Path(name).resolve()):sha for name,sha in summaries[method][stage]['tip_geometry_label_source_sha256'].items()})
    for path, sha in evaluation_hashes.items():
        if recorded_sources.get(path) != sha:
            raise ValueError('post-hoc DEV source changed since training/evaluation: '+path)
    args.output.mkdir(parents=True)
    results = {}; all_rows = {}; artifact_index = {}
    for method in ('soft', 'peak'):
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
            all_rows[key] = {r['scene_id']: r for r in scenes}
        artifact_index[method] = dict(binaries=artifacts, config_sha256=digest(run/'config.json'),
            summary_sha256=digest(run/'summary.json'), history_sha256=digest(run/'history.json'),
            training_sources_sha256=digest(run/'source_hashes.json'))
    paired = {}
    for stage in ('best', 'last'):
        differences = []
        for row in rows:
            a, b = [all_rows[method+'_'+stage][row['id']] for method in ('soft', 'peak')]
            delta = {}
            for key in PAIR_FIELDS:
                x = a['tip_evaluation'].get(key, a.get(key)); y = b['tip_evaluation'].get(key, b.get(key))
                delta[key] = None if x is None or y is None else y-x
            differences.append(dict(scene_id=row['id'], parent_id=row['parent_id'], reference_count=a['reference_count'], peak_minus_soft=delta))
        parent_differences = []
        for parent in sorted(set(dataset['parent_ids'])):
            values = [r for r in differences if r['parent_id']==parent]
            delta = {key: float(np.mean([r['peak_minus_soft'][key] for r in values if r['peak_minus_soft'][key] is not None]))
                     if any(r['peak_minus_soft'][key] is not None for r in values) else None for key in PAIR_FIELDS}
            parent_differences.append(dict(parent_id=str(parent), instructions=len(values), peak_minus_soft=delta))
        paired[stage] = dict(per_instruction=differences, per_parent=parent_differences,
            aggregate_peak_minus_soft={key: results['peak_'+stage]['metrics'][key]-results['soft_'+stage]['metrics'][key]
                if results['peak_'+stage]['metrics'][key] is not None else None for key in PAIR_FIELDS})
    result = dict(protocol='observed_obstacle_pair_audit_v1', evaluation_protocol='observation_eval_v2', tip_evaluation_protocol=PROTOCOL,
        seed=args.seed, selection_protocol='dev_tip_unique_valid_v1', criterion='UniqueClassifiedTipValidAtK + 0.05 * TipValidAtK',
        common_candidate_slots=summaries['soft']['trajectory_exposures'], metadata=metadata, results=results, paired=paired,
        actual_artifact_index=artifact_index, training_source_hashes=source_hashes, evaluation_file_hashes=evaluation_hashes,
        initialization=dict(reconstructed_state_sha256=init_hashes, evidence='Exact immutable-source/seed reconstruction; initial weights were not saved during training.'),
        sampler_audit=samplers, cost={method:{key:summaries[method][key] for key in ('elapsed_s','gpu_hours_reserved','peak_cuda_memory_mb','data_load_preprocess_s','latency')} for method in summaries},
        limitations='One training seed; all parents are DEV_MODEL. Tip checks certify only added boxes at original2cm; no arm/table/IK/execution or core novelty claim.',
        script_sha256=digest(__file__), analysis_cpu_wall_s=time.perf_counter()-started)
    if args.planner is not None:
        result['traditional_planner'] = audit_planner(args.planner,rows,current,geometry,labels,
            {row['parent_id'] for row in observations if row['split']=='TRAIN' and labels[row['id']]['routes']},
            metadata,configs['soft']['candidates'])
    write(args.output/'analysis.json', result)
    write(args.output/'actual_predictions_index.json', artifact_index)
    print(json.dumps({key:dict(step=value['step'],metrics={field:value['metrics'][field] for field in PAIR_FIELDS}) for key,value in results.items()},indent=2))


if __name__ == '__main__':
    main()
