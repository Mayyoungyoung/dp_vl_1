"""Fixed-best, serial real-Qwen/RGB-D K4 evaluation of the registered DEV36.

Each request's predictions are written and hashed BEFORE its labels are read.
Labels never reach FrozenGenerator. No repair, selection, retry or extra model
call occurs. Request walltime includes prediction sealing and label checking.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import time

import numpy as np

PROTOCOL = 'two_row_best_online_dev36_v1'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
INPUT_KEYS = {'id', 'parent_id', 'split', 'image', 'instruction'}
CURRENT_KEYS = {'depth', 'camera_intrinsics', 'camera_extrinsics', 'gripper_pose', 'gripper_open'}
DEV_PARENTS = ['two_row_reach_%d' % seed for seed in range(283264, 283276)]
DEV_IDS = [parent + '_target%d' % target for parent in DEV_PARENTS for target in range(3)]
ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ('routeset/observed_geometry.py', 'routeset/observed_route_head.py',
    'routeset/observed_path_refinement.py', 'routeset/models.py', 'routeset/common.py',
    'scripts/observation_cache_qwen.py', 'scripts/train_observed_geometry.py',
    'scripts/train_observed_two_row.py', 'scripts/evaluate_observed_two_row.py',
    'scripts/evaluate_observed_obstacles.py', 'scripts/collect_observed_two_row_pilot.py',
    'scripts/collect_obstacle_reach.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def append_json(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, allow_nan=False) + '\n')


def observation_contract(row):
    if set(row) != INPUT_KEYS or row['split'] != 'DEV_MODEL' or row['id'] not in DEV_IDS:
        raise ValueError('Only exact registered DEV observation fields accepted')
    if row['parent_id'] != row['id'].rsplit('_target', 1)[0]:
        raise ValueError('Observation parent identity mismatch')
    if not isinstance(row['instruction'], str) or not row['instruction'].strip():
        raise ValueError('Nonempty observed language required')
    return dict(row)


def select_observations(rows, train_parents=16):
    if train_parents not in (16,32,64):raise ValueError('Only registered TRAIN prefix sizes accepted')
    allowed_train={'two_row_reach_%d'%s for s in range(283200,283200+train_parents)}
    selected = {}; seen = set()
    for row in rows:
        if set(row) != INPUT_KEYS or row['id'] in seen:
            raise ValueError('Observation manifest whitelist/identity violation')
        seen.add(row['id'])
        if row['split'] == 'DEV_MODEL':
            selected[row['id']] = observation_contract(row)
        elif (row['split'] != 'TRAIN' or row['parent_id'] not in allowed_train
                or row['id'] not in [row['parent_id']+'_target%d'%t for t in range(3)]):
            raise ValueError('Unexpected non-prefix role or parent')
    return selected


def checked_file(path, expected):
    path = Path(path).resolve()
    if expected.get(str(path)) != digest(path): raise ValueError('Source SHA mismatch: ' + str(path))
    return path


def input_paths(row, expected):
    row = observation_contract(row)
    image = Path(row['image']).resolve()
    if image.name != 'front.png' or image.parent.name != row['parent_id']:
        raise ValueError('Current observation must share the recorded front-image directory')
    current = image.with_name('observation.npz')
    return checked_file(image, expected), checked_file(current, expected)


def head_options(config):
    expected = dict(feature_dim=4096, horizon=24, candidates=4, width=128, depth=2,
        point_width=64, pooling='both', pixel_stride=2, endpoint_residual_bound=.05,
        anchor_mode='straight_through_peak', endpoint_mode='surface_anchor', refinement_mode='none',
        geometry_pooling='spatial', objective='saturation', checkpoint_selection='tip_unique_valid')
    for name, value in expected.items():
        if config.get(name) != value: raise ValueError('Fixed ordinary architecture mismatch: ' + name)
    if any(config.get(name) is not None for name in ('refinement_sigma', 'refinement_prefix_fraction', 'refinement_bound')):
        raise ValueError('No draft/refinement budget is authorized')
    return dict(feature_dim=4096, horizon=24, max_candidates=4, width=128, depth=2,
        point_width=64, endpoint_residual_bound=.05, geometry_pooling='spatial',
        anchor_mode='straight_through_peak', endpoint_mode='surface_anchor', refinement_mode='none')


def failed_prediction():
    return dict(paths=np.full((4, 24, 3), np.nan, np.float32),
        gripper_open=np.full((4, 24), np.nan, np.float32), current=np.full(8, np.nan, np.float32),
        mean_hidden=np.empty(0, np.float32), last_hidden=np.empty(0, np.float32), input_tokens=np.array(0))


def validate_prediction(value):
    if set(value) != {'paths', 'gripper_open', 'current', 'mean_hidden', 'last_hidden', 'input_tokens'}:
        raise ValueError('Unexpected generated artifact fields')
    if value['paths'].shape != (4, 24, 3) or value['gripper_open'].shape != (4, 24) or value['current'].shape != (8,):
        raise ValueError('Fixed K4/H24 output required; no post-hoc truncation')


def seal_prediction(output, identifier, value):
    validate_prediction(value)
    path = output / (identifier + '.npz')
    with path.open('xb') as stream: np.savez_compressed(stream, **value)
    sha = digest(path)
    write_json(path.with_suffix('.seal.json'), dict(id=identifier, prediction_sha256=sha,
        candidates=4, horizon=24, sealed_before_label_access=True))
    return path, sha


def process_request(identifier, row, generator, evaluator, output, clock=time.perf_counter, blocked=None):
    """Testable ordering boundary. Evaluator receives bytes sealed by generator.

    Neither evaluator nor a label object is passed into the generator. Every
    call consumes one registered request and four slots, including failures.
    """
    start = clock(); error = None; stages = {}; attempted = row is not None and blocked is None
    value = failed_prediction()
    if attempted:
        try:
            value, stages = generator(observation_contract(row))
            validate_prediction(value)
        except Exception as exc:
            error = type(exc).__name__ + ': ' + str(exc)
            value = failed_prediction()
    else: error = blocked or 'registered_observation_unavailable'
    generated = clock()
    path, sha = seal_prediction(output, identifier, value)
    sealed = clock()
    # Only here may evaluation inspect this request's labels. Read back the
    # exact sealed output, so metrics cannot accidentally use altered arrays.
    metrics, candidates, check_timing = evaluator(identifier, row, path, sha)
    finish = clock()
    if digest(path) != sha: raise ValueError('Evaluation modified the sealed prediction')
    record = dict(id=identifier, parent_id=identifier.rsplit('_target', 1)[0],
        requested_candidates=4, generation_attempted=attempted, generation_error=error,
        generation_ms=(generated-start)*1000, prediction_seal_ms=(sealed-generated)*1000,
        label_and_check_ms=(finish-sealed)*1000, continuous_request_wall_ms=(finish-start)*1000,
        stages=stages, check_timing=check_timing, prediction_file=path.name, prediction_sha256=sha,
        finite_path_candidates=int(np.isfinite(value['paths']).all((1, 2)).sum()),
        retries=0, repairs=0, filtered_candidates=0, additional_complete_paths=0,
        metrics=metrics, candidates=candidates)
    return record, value


class FrozenGenerator:
    """Owns frozen models, raw observation hashes and architecture only, no labels."""
    def __init__(self, backbone, processor, head, pixel_stride, expected_inputs):
        self.backbone, self.processor, self.head = backbone, processor, head
        self.pixel_stride = pixel_stride
        self.expected_inputs = dict(expected_inputs)

    def __call__(self, row):
        import torch
        from PIL import Image
        from routeset.observed_geometry import backproject_rgbd
        if self.backbone.training or self.head.training or any(p.requires_grad for m in (self.backbone, self.head) for p in m.parameters()):
            raise ValueError('Only frozen evaluation models permitted')
        def clock():
            torch.cuda.synchronize(); return time.perf_counter()
        start = clock()
        image_path, current_path = input_paths(row, self.expected_inputs)
        with Image.open(image_path) as image_file: image = image_file.convert('RGB')
        with np.load(current_path, allow_pickle=False) as archive:
            if set(archive.files) != CURRENT_KEYS: raise ValueError('Current NPZ violates strict observation whitelist')
            observed = {key: archive[key].copy() for key in CURRENT_KEYS}
        current = np.r_[observed['gripper_pose'].reshape(7), observed['gripper_open'].reshape(1)].astype(np.float32)
        if not np.isfinite(current).all(): raise ValueError('Nonfinite current state')
        read_end = clock()
        points = backproject_rgbd(np.asarray(image), observed['depth'], observed['camera_intrinsics'],
            observed['camera_extrinsics'], pixel_stride=self.pixel_stride)
        batch = {key: torch.as_tensor(points[key][None], device='cuda')
            for key in ('world_xyz', 'rgb', 'uv', 'depth', 'valid_mask')}
        batch['current'] = torch.as_tensor(current[None], device='cuda')
        geometry_end = clock()
        messages = [{'role':'user', 'content':[{'type':'image'}, {'type':'text', 'text':row['instruction']}]}]
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=[image], return_tensors='pt').to('cuda')
        processor_end = clock()
        with torch.inference_mode():
            outputs = self.backbone.model(**inputs, use_cache=False, return_dict=True)
            hidden = outputs.last_hidden_state[0][inputs.attention_mask[0].bool()].float()
            mean, last = hidden.mean(0), hidden[-1]
            batch['features'] = torch.cat((mean, last))[None]
            qwen_end = clock()
            geometry_times = []
            before_hook = self.head.geometry.register_forward_pre_hook(lambda *unused: geometry_times.append(clock()))
            after_hook = self.head.geometry.register_forward_hook(lambda *unused: geometry_times.append(clock()))
            try:
                paths, opened, _ = self.head(**batch)
            finally:
                before_hook.remove(); after_hook.remove()
            if len(geometry_times) != 2: raise ValueError('Exactly one learned geometry forward required')
            value = dict(paths=paths[0].cpu().numpy(), gripper_open=opened[0].cpu().numpy(), current=current,
                mean_hidden=mean.cpu().numpy(), last_hidden=last.cpu().numpy(), input_tokens=np.array(int(inputs.attention_mask.sum())))
        finish = clock()
        return value, dict(raw_input_read_hash_ms=(read_end-start)*1000,
            rgbd_backprojection_transfer_ms=(geometry_end-read_end)*1000,
            qwen_processor_transfer_ms=(processor_end-geometry_end)*1000,
            qwen_real_forward_pool_ms=(qwen_end-processor_end)*1000,
            learned_geometry_head_output_transfer_ms=(finish-qwen_end)*1000,
            learned_geometry_encoder_ms=(geometry_times[1]-geometry_times[0])*1000,
            route_head_output_transfer_and_instrumentation_ms=((finish-qwen_end)-(geometry_times[1]-geometry_times[0]))*1000,
            input_tokens=int(value['input_tokens']), sampled_points=len(points['depth']))


def request_label(supervision, identifier):
    """Deserialize only this label; other JSONL records are scanned for ID only."""
    selected = []
    with Path(supervision).open(encoding='utf-8-sig') as stream:
        for line in stream:
            match = re.match(r'\s*\{\s*"id"\s*:\s*("(?:[^"\\]|\\.)*")', line)
            if not match: raise ValueError('Frozen export requires ID-first JSONL records')
            if json.loads(match.group(1)) == identifier: selected.append(line)
    if len(selected) != 1: raise ValueError('Exactly one matching exported evaluation label required')
    return json.loads(selected[0])


def unavailable_metrics():
    return dict(TipValidAtK=0., AnyTipValidAtK=0., UniqueClassifiedTipValidAtK=0,
        UnknownTypeTipValidCount=0, DuplicateClassifiedTipValidCount=0,
        KnownReferenceTypeCoverageAtK=None, semantic_goal_accuracy=0., AnySemanticGoalAtK=0.,
        TipClearAtK=0., StartCorrectAtK=0., EventSequenceCorrectAtK=0., SelectedTipValidAtK=None,
        format_failure_count=4, known_reference_types=None, reference_count=None,
        evaluation_status='registered_input_unavailable; no invented geometry or reference labels')


class LabelEvaluator:
    def __init__(self, data, manifest): self.data, self.manifest = Path(data), manifest

    def __call__(self, identifier, row, prediction, expected_sha):
        from scripts.evaluate_observed_two_row import scene_metrics
        start = time.perf_counter()
        if digest(prediction) != expected_sha: raise ValueError('Prediction seal mismatch before label access')
        if row is None: return unavailable_metrics(), [], dict(evaluation_not_available=True)
        with np.load(prediction, allow_pickle=False) as archive:
            paths, opened, current = (archive[key].copy() for key in ('paths', 'gripper_open', 'current'))
        checked_file(self.data/'supervision.jsonl', {str((self.data/'supervision.jsonl').resolve()):self.manifest['output_files_sha256']['supervision.jsonl']})
        label = request_label(self.data/'supervision.jsonl', identifier)
        if label['id'] != identifier or label['parent_id'] != row['parent_id'] or label['split'] != 'DEV_MODEL':
            raise ValueError('Evaluation label identity mismatch')
        sources = self.manifest['source_files_sha256']
        observation = checked_file(label['observation'], sources)
        if observation != Path(row['image']).resolve().with_name('observation.npz'): raise ValueError('Evaluation current differs from forward observation')
        # Current values for failed forwards are still observed inputs, never answers.
        with np.load(observation, allow_pickle=False) as a:
            expected_current = np.r_[a['gripper_pose'].reshape(7), a['gripper_open'].reshape(1)].astype(np.float32)
        if np.isfinite(current).all() and not np.array_equal(current, expected_current): raise ValueError('Generated current state changed')
        with np.load(checked_file(label['verification_only'], sources), allow_pickle=False) as archive:
            geometry = {key:archive[key] for key in ('obstacle_centers', 'obstacle_halfsizes')}
        route_config = checked_file(label['route_config'], sources)
        if digest(route_config) != label['route_config_sha256']: raise ValueError('Route label config hash changed')
        config = json.loads(route_config.read_text())
        loaded = time.perf_counter()
        metrics, candidates = scene_metrics(paths, opened,
            dict(gripper_pose=expected_current[:7], gripper_open=expected_current[7]), geometry,
            label['semantic_targets'], label['route_types'], config)
        metrics['reference_count'] = len(label['routes'])
        finish = time.perf_counter()
        return metrics, candidates, dict(sealed_prediction_and_label_io_hash_ms=(loaded-start)*1000,
            two_row_validation_ms=(finish-loaded)*1000)


def compare_cache(run, config, rows, predictions, training_sources):
    """Saved-array-only diagnostic; never calls a model or changes candidates."""
    run = Path(run); reference_file = run/'dev_model/predictions.npz'
    summary = json.loads((run/'summary.json').read_text())
    if digest(reference_file) != summary['prediction_sha256']: raise ValueError('Cached best prediction SHA changed')
    with np.load(reference_file, allow_pickle=False) as a: saved = {k:a[k].copy() for k in a.files}
    ids = list(map(str, saved['scene_ids']))
    if len(set(ids)) != len(ids) or set(ids) != set(rows): raise ValueError('Cached best must cover every available DEV input')
    if saved['paths'].shape != (len(ids),4,24,3) or saved['gripper_open'].shape != (len(ids),4,24): raise ValueError('Cached output budget mismatch')
    reports = []
    for identifier in DEV_IDS:
        if identifier not in rows: continue
        row = rows[identifier]; index = ids.index(identifier); value = predictions[identifier]
        if str(saved['parent_ids'][index]) != row['parent_id']: raise ValueError('Cached prediction parent mismatch')
        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
        cache = checked_file(Path(config['cache_dir'])/(key+'.npz'), training_sources)
        with np.load(cache, allow_pickle=False) as a:
            for name in ('id','parent_id','split'):
                if str(a[name].item()) != row[name]: raise ValueError('Cached feature identity differs')
            if str(a['image_sha256'].item()) != digest(row['image']): raise ValueError('Cached image changed')
            valid = value['mean_hidden'].shape == a['mean_hidden'].shape and value['last_hidden'].shape == a['last_hidden'].shape
            feature = {name:float(np.max(np.abs(value[name]-a[name]))) if valid else None for name in ('mean_hidden','last_hidden')}
            tokens_equal = int(value['input_tokens']) == int(a['input_tokens'])
        before, after = saved['paths'][index], value['paths']
        finite = np.isfinite(after).all() and np.isfinite(value['gripper_open']).all()
        reports.append(dict(id=identifier, cache_file_sha256=digest(cache), feature_max_abs_difference=feature,
            input_tokens_equal=tokens_equal, all_generated_values_finite=bool(finite),
            xyz_max_abs_difference_m=float(np.max(np.abs(after-before))) if finite else None,
            endpoint_max_difference_m=float(np.linalg.norm(after[:,-1]-before[:,-1], axis=-1).max()) if finite else None,
            event_max_abs_difference=float(np.max(np.abs(value['gripper_open']-saved['gripper_open'][index]))) if finite else None,
            xyz_allclose_atol1e_6_rtol1e_5=bool(np.allclose(after,before,atol=1e-6,rtol=1e-5)),
            cache_prediction_row=index))
    return dict(saved_prediction_sha256=digest(reference_file), examples=len(reports), per_scene=reports,
        additional_model_requests=0, cached_feature_in_forward=False,
        policy='Numerical/decision mismatches are retained, not repaired, reselected or rerun; batched versus serial arithmetic can differ.')


def verify_training_artifacts(run, config):
    run = Path(run)
    path = run/'two_row_driver_receipt.json'; receipt = json.loads(path.read_text())
    if (receipt.get('protocol') != 'ordinary_two_row_frozen_qwen_saturation_v1'
            or receipt['actual_training_summary_sha256'] != digest(run/'summary.json')
            or receipt['export_sha256'] != config['two_row_export_sha256']
            or receipt['source_sha256'] != config['two_row_driver_sha256']):
        raise ValueError('Completed driver receipt does not bind this training result')
    hashes = {}
    for filename in ('dev_model/predictions.npz', 'dev_model/per_scene.json', 'dev_model/metrics.json'):
        entry = receipt['prediction_artifacts'][filename]; actual = run/filename
        if (Path(entry['path']).resolve() != actual.resolve() or entry['sha256'] != digest(actual)
                or entry['bytes'] != actual.stat().st_size):
            raise ValueError('Fixed best comparison artifact differs from driver receipt')
        hashes[filename] = entry['sha256']
    return dict(driver_receipt_sha256=digest(path), best_comparison_artifact_sha256=hashes)


def preflight(args):
    from scripts.export_two_row_observations import verify_export, selection_guard
    data, run, source = args.data.resolve(), args.run.resolve(), args.training_source.resolve()
    config = json.loads((run/'config.json').read_text()); summary = json.loads((run/'summary.json').read_text())
    status = json.loads((run/'status.json').read_text())
    if status.get('status') != 'completed' or status.get('exit_code') != 0 or status.get('step') != 1500:
        raise ValueError('Completed fixed1500 ordinary training required')
    head_options(config)
    if config.get('two_row_driver_protocol') != 'ordinary_two_row_frozen_qwen_saturation_v1': raise ValueError('Two-row driver required')
    if source.name != config['code_commit'] or not re.fullmatch('[0-9a-f]{40}', source.name): raise ValueError('Explicit actual immutable training source required')
    sources = {}
    for filename in DEPENDENCIES:
        current, original = ROOT/filename, source/filename
        if digest(current) != digest(original): raise ValueError('Training source dependency differs: '+filename)
        sources[filename] = digest(current)
    for key, filename in [('source_script_sha256','scripts/train_observed_geometry.py'),('two_row_driver_sha256','scripts/train_observed_two_row.py')]:
        if config[key] != sources[filename]: raise ValueError('Training configuration source SHA differs')
    for field, filename in [('observations','observations.jsonl'),('supervision','supervision.jsonl'),('cache_dir','qwen_cache')]:
        if Path(config[field]).resolve() != data/filename: raise ValueError('Run/export path differs: '+field)
    manifest, gate = verify_export(data)  # Mechanical metadata and integrity bytes, no label-array deserialization.
    selection_guard(manifest['selection'])
    if config['two_row_export_sha256'] != digest(data/'export_manifest.json'): raise ValueError('Training export changed')
    rows = select_observations([json.loads(line) for line in (data/'observations.jsonl').read_text().splitlines() if line.strip()],
        manifest['selection']['requested_parents']['TRAIN'])
    input_hashes = {}
    for row in rows.values():
        for path in input_paths(row,manifest['source_files_sha256']): input_hashes[str(path)] = digest(path)
    cache = json.loads((data/'qwen_cache/cache_config.json').read_text())
    if cache != config['cache_config']: raise ValueError('Actual Qwen cache configuration changed')
    if (cache['model'] != 'Qwen/Qwen3-VL-2B-Instruct' or cache['revision'] != REVISION or cache['processor'] != REVISION
            or cache['model_trainable_parameter_count'] != 0 or cache['manifest_sha256'] != digest(data/'observations.jsonl')
            or set(cache['input_contract']) != INPUT_KEYS or cache['dtype'] != 'torch.bfloat16'):
        raise ValueError('Pinned frozen real Qwen cache required')
    provenance = json.loads((args.model/'provenance.json').read_text())
    if provenance.get('model_id') != cache['model'] or provenance.get('revision') != REVISION or not provenance.get('all_hashes_verified'):
        raise ValueError('Official Qwen provenance mismatch')
    if digest(run/'best.pt') != summary['best_checkpoint_sha256']: raise ValueError('Fixed best checkpoint bytes changed')
    artifact_receipt = verify_training_artifacts(run, config)
    training_sources = json.loads((run/'source_hashes.json').read_text())
    # Hashes establish provenance; reference arrays themselves are never loaded.
    for path, sha in training_sources.items():
        if digest(path) != sha: raise ValueError('Original training source changed: '+path)
    receipt = dict(protocol=PROTOCOL, run=str(run), data=str(data), training_source=str(source),
        source_files_sha256=dict(sources, **{'scripts/evaluate_observed_two_row_online.py':digest(__file__)}),
        checkpoint_sha256=summary['best_checkpoint_sha256'], checkpoint_step=summary['best_step'],
        training_artifact_receipt=artifact_receipt,
        config_sha256=digest(run/'config.json'), training_summary_sha256=digest(run/'summary.json'),
        export_manifest_sha256=digest(data/'export_manifest.json'), model_provenance_sha256=digest(args.model/'provenance.json'),
        observed_input_hashes=input_hashes, live_gate_before=gate, requested_input_ids=DEV_IDS,
        available_input_ids=[i for i in DEV_IDS if i in rows], requested_requests=36, requested_candidates=144,
        no_labels_in_forward=True, no_cached_features_in_forward=True, checkpoint_selection='unchanged saved best only',
        authorized_gpu='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab', memory_fraction=.35, cpu_threads=1)
    return config, summary, manifest, rows, input_hashes, training_sources, receipt


def summarize(records):
    fields=('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','UnknownTypeTipValidCount',
        'DuplicateClassifiedTipValidCount','KnownReferenceTypeCoverageAtK','semantic_goal_accuracy','AnySemanticGoalAtK',
        'TipClearAtK','StartCorrectAtK','EventSequenceCorrectAtK','SelectedTipValidAtK')
    result={}
    for field in fields:
        values=[r['metrics'][field] for r in records if r['metrics'].get(field) is not None]
        result[field]=float(np.mean(values)) if values else None
    times=np.array([r['continuous_request_wall_ms'] for r in records if r['generation_attempted']])
    result.update(requested_requests=36, requested_candidate_slots=144, recorded_requests=len(records),
        actual_attempted_generation_requests=sum(r['generation_attempted'] for r in records),
        unavailable_or_unattempted_requests=sum(not r['generation_attempted'] for r in records),
        generation_failure_requests=sum(r['generation_error'] is not None for r in records),
        all_attempted_request_ms_median=float(np.median(times)) if len(times) else None,
        all_attempted_request_ms_p95=float(np.percentile(times,95)) if len(times) else None,
        first_attempted_request_ms=float(times[0]) if len(times) else None,
        coverage_evaluable_requests=sum(r['metrics'].get('KnownReferenceTypeCoverageAtK') is not None for r in records),
        requested_input_success_denominator=36, unknown_missing_reference_types_are_negative=False,
        SelectedValidAtK=None, robot_execution_success=None)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','model','training-source','output'): parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(argv); pipeline_start=time.perf_counter()
    args.output.mkdir(parents=True,exist_ok=False)
    try:
        config, train_summary, manifest, rows, input_hashes, training_sources, receipt=preflight(args)
        write_json(args.output/'preflight.json',receipt)
        if args.validate_only:
            write_json(args.output/'status.json',dict(status='metadata_preflight_only',gpu_executed=False));return 0
        import torch
        import transformers
        from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
        from routeset.observed_geometry import ObservedGeometryRouteHead
        from scripts.export_two_row_observations import verify_export
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Only physical GPU1 is authorized')
        if transformers.__version__!=config['cache_config']['transformers'] or torch.__version__!=config['cache_config']['torch']:
            raise ValueError('Exact original Qwen cache runtime required')
        torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35);torch.cuda.reset_peak_memory_stats()
        load_start=time.perf_counter()
        checkpoint=torch.load(args.run/'best.pt',map_location='cpu',weights_only=False)
        if checkpoint['config']!=config or checkpoint['step']!=train_summary['best_step']:
            raise ValueError('Checkpoint/config/selected step differs')
        backbone=Qwen3VLForConditionalGeneration.from_pretrained(args.model,torch_dtype=torch.bfloat16,
            local_files_only=True,attn_implementation='sdpa',device_map={'':'cuda'})
        backbone.requires_grad_(False).eval()
        processor=AutoProcessor.from_pretrained(args.model,local_files_only=True,min_pixels=64*32*32,
            max_pixels=config['cache_config']['max_pixels'])
        head=ObservedGeometryRouteHead(**head_options(config)).to('cuda')
        head.load_state_dict(checkpoint['model'],strict=True);head.requires_grad_(False).eval()
        del checkpoint
        torch.cuda.synchronize(); loading_seconds=time.perf_counter()-load_start
        # The generator's input registry contains only image/current hashes.
        generator=FrozenGenerator(backbone,processor,head,config['pixel_stride'],input_hashes)
        parameter_versions=tuple(p._version for m in (backbone,head) for p in m.parameters())
        evaluator=LabelEvaluator(args.data,manifest)
        sealed=args.output/'requests';sealed.mkdir();records=[];predictions={};blocked=None
        for identifier in DEV_IDS:
            append_json(args.output/'request_journal.jsonl',dict(id=identifier,event='started',candidate_slots=4))
            record,value=process_request(identifier,rows.get(identifier),generator,evaluator,sealed,blocked=blocked)
            records.append(record);predictions[identifier]=value
            append_json(args.output/'requests.jsonl',record)
            append_json(args.output/'request_journal.jsonl',dict(id=identifier,event='completed',prediction_sha256=record['prediction_sha256']))
            if record['generation_attempted'] and record['generation_error'] is not None:
                blocked='not_attempted_after_first_runtime_failure; no retry: '+record['generation_error']
            print(json.dumps({k:record[k] for k in ('id','generation_attempted','generation_error','continuous_request_wall_ms')}),flush=True)
        request_pipeline_seconds=time.perf_counter()-load_start-loading_seconds
        np.savez_compressed(args.output/'predictions.npz',paths=np.stack([predictions[i]['paths'] for i in DEV_IDS]),
            gripper_open=np.stack([predictions[i]['gripper_open'] for i in DEV_IDS]),scene_ids=np.array(DEV_IDS),
            parent_ids=np.array([i.rsplit('_target',1)[0] for i in DEV_IDS]))
        comparison=compare_cache(args.run,config,rows,predictions,training_sources)
        if verify_training_artifacts(args.run,config)!=receipt['training_artifact_receipt']:
            raise ValueError('Training comparison artifact changed during online evaluation')
        old_rows={r['scene_id']:r for r in json.loads((args.run/'dev_model/per_scene.json').read_text())}
        decision_fields=('TipValid','semantic_goal_correct','starts_at_current_state','tip_segments_clear',
            'event_state_sequence_correct','declared_passage_type')
        for r in records:
            if r['id'] not in old_rows:continue
            old=old_rows[r['id']]['tip_candidates'];new=r['candidates']
            if len(old)!=4 or len(new)!=4:raise ValueError('Four cached/online decision rows required')
            comparison_row=next(x for x in comparison['per_scene'] if x['id']==r['id'])
            comparison_row['candidate_decisions_identical']=all(json.dumps(a[k],sort_keys=True)==json.dumps(b[k],sort_keys=True)
                for a,b in zip(old,new) for k in decision_fields)
        write_json(args.output/'cache_comparison.json',comparison)
        _,final_gate=verify_export(args.data)
        if parameter_versions!=tuple(p._version for m in (backbone,head) for p in m.parameters()):
            raise ValueError('Frozen model parameter was updated during online evaluation')
        if digest(args.run/'best.pt')!=receipt['checkpoint_sha256']:raise ValueError('Original best checkpoint changed')
        for r in records:
            if digest(sealed/r['prediction_file'])!=r['prediction_sha256']:raise ValueError('A sealed request changed')
        result=dict(receipt,**summarize(records),status='completed_with_failures' if blocked else 'completed',
            model_loading_seconds=loading_seconds,metadata_startup_seconds=load_start-pipeline_start,
            serial_request_pipeline_seconds=request_pipeline_seconds,total_pipeline_wall_seconds=time.perf_counter()-pipeline_start,
            peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),live_gate_after=final_gate,
            prediction_sha256=digest(args.output/'predictions.npz'),cache_comparison_sha256=digest(args.output/'cache_comparison.json'),
            timing_scope='Continuous serial per-request IO + real Qwen + RGB-D + head + prediction sealing + same-request label IO/check; model startup and post-run saved-array consistency are separate. No feature cache, warmup, repair, retry or robot execution.',
            filesystem_cache_note='OS file pages may be warm; integrity preflight hashes input files.',
            validity_scope='Two-row tip-only2cm/goal3cm checks; no arm/IK/execution certification',
            additional_model_requests_for_consistency=0,source_training_cost_counted_again=False)
        result['gpu_hours_reserved']=result['total_pipeline_wall_seconds']/3600
        write_json(args.output/'summary.json',result)
        write_json(args.output/'status.json',dict(status=result['status'],exit_code=1 if blocked else 0,requests=36))
        return 1 if blocked else 0
    except BaseException as exc:
        write_json(args.output/'status.json',dict(status='failed',exit_code=1,error=repr(exc),
            elapsed_seconds=time.perf_counter()-pipeline_start,no_automatic_resume_or_retry=True))
        raise


if __name__=='__main__':raise SystemExit(main())
