"""Fixed four-search spatial-penalty adaptation of the frozen observation A*.

This is a traditional baseline, not a learned contribution or a homotopy
certificate. The scoped adapter changes only grid-edge reuse costs. Every
returned raw route participates, irrespective of all later checks or labels.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import re
import time
import traceback

import numpy as np
from PIL import Image

from scripts import evaluate_two_row_astar_v2 as control
from scripts import export_two_row_observations as exporter

PROTOCOL = 'observed_two_row_spatial_penalty_v1'
SIGMA_M = .05
_ACTIVE = False


def squared_polyline_distance(points, raw):
    """Exact Euclidean distance to continuous segments, without rasterization."""
    points, raw = np.asarray(points, dtype=np.float64), np.asarray(raw, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 3 or raw.ndim != 2 or raw.shape[1] != 3
            or len(raw) < 2 or not np.isfinite(points).all() or not np.isfinite(raw).all()):
        raise ValueError('Finite XYZ queries and a complete polyline required')
    best = np.full(len(points), np.inf)
    for first, second in zip(raw, raw[1:]):
        delta = second-first
        norm = float(delta @ delta)
        offset = points-first
        parameter = np.clip(np.einsum('ij,j->i', offset, delta)/norm, 0., 1.) if norm else np.zeros(len(points))
        residual = offset-parameter[:, None]*delta
        best = np.minimum(best, np.einsum('ij,ij->i', residual, residual))
    return best


def spatial_field(free, lower, raw_paths, voxel=.025, sigma=SIGMA_M):
    """Sum one Gaussian per earlier complete raw route on free grid vertices.

Blocked vertices are never read by the original search. Chunking bounds
temporary allocations, and changes neither the field nor the search budget.
"""
    if sigma != SIGMA_M or voxel != .025:
        raise ValueError('Prospectively fixed sigma .05m and voxel .025m required')
    free = np.asarray(free, dtype=bool)
    if free.ndim != 3 or free.size > 2_000_000:
        raise ValueError('Original bounded 3D grid required')
    field = np.zeros(free.shape, dtype=np.float64)
    flat = field.ravel()
    indices = np.flatnonzero(free)
    for start in range(0, len(indices), 16384):
        chosen = indices[start:start+16384]
        points = np.asarray(lower)+np.stack(np.unravel_index(chosen, free.shape), axis=-1)*voxel
        values = np.zeros(len(chosen))
        for raw in raw_paths:
            values += np.exp(-squared_polyline_distance(points, raw)/(2*sigma*sigma))
        flat[chosen] = values
    return field


class SpatialCosts:
    def __init__(self, used_edges, field):
        self.used_edges, self.field = used_edges, field

    def __getitem__(self, key):
        if key[0] in ('start', 'goal'):
            return self.used_edges[key]
        if key[0] != 'grid' or len(key) != 3:
            raise ValueError('Unregistered edge kind')
        return float((self.field[key[1]]+self.field[key[2]])*.5)


class SpatialSearch:
    """Track raw paths while checking the old caller's unmodified Counter."""
    def __init__(self, original, current, config):
        if (config['candidates'], config['horizon'], config['voxel_m'],
                config['used_edge_penalty_multiplier'], config['maximum_expanded_nodes_per_candidate'],
                config['search_deadline_seconds']) != (4, 24, .025, 4., 20000, 2.):
            raise ValueError('Frozen search constants changed')
        self.original, self.current, self.config = original, np.asarray(current[:3]).copy(), config
        self.raw_paths, self.expected, self.calls = [], Counter(), 0

    def __call__(self, free, lower, exact_goal, starts, goals, used_edges, start_permission, target_permission):
        if self.calls >= 4:
            raise ValueError('Four searches maximum; no hidden replacement')
        if dict(used_edges) != dict(self.expected):
            raise ValueError('Original Counter differs from all previously returned paths')
        self.calls += 1
        began = time.perf_counter()
        count = len(self.raw_paths)
        # No proxy at all in slot zero: exact original costs and tie ordering.
        if count:
            field = spatial_field(free, lower, self.raw_paths)
            costs = SpatialCosts(used_edges, field)
            audit = dict(field_max=float(field.max()), field_sha256=hashlib.sha256(field.tobytes()).hexdigest(),
                         evaluated_free_vertices=int(np.count_nonzero(free)), field_bytes=int(field.nbytes))
        else:
            costs, audit = used_edges, dict(field_max=0., field_sha256=None, evaluated_free_vertices=0, field_bytes=0)
        preprocess = time.perf_counter()-began
        cells, record = self.original(free, lower, exact_goal, starts, goals, costs, start_permission, target_permission)
        record['spatial_penalty'] = dict(previous_returned_raw_paths=count, sigma_m=SIGMA_M,
            preprocessing_seconds=preprocess, preprocessing_inside_original_search_deadline=False,
            search_plus_preprocessing_seconds=float(record['search_seconds'])+preprocess,
            all_returned_paths_included_without_postcheck_filter=True, **audit)
        if cells is not None:
            cells = [tuple(int(x) for x in cell) for cell in cells]
            raw = np.vstack([self.current, np.asarray(lower)+np.asarray(cells)*.025, exact_goal])
            self.raw_paths.append(raw)
            self.expected[('start', cells[0])] += 1
            self.expected[('goal', cells[-1])] += 1
            for first, second in zip(cells, cells[1:]):
                self.expected[('grid',)+tuple(sorted((first, second)))] += 1
        return cells, record


@contextmanager
def spatial_adapter(planner, current):
    """Single-thread scoped adaptation; restore even after an exception."""
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError('Nested/concurrent scoped planner adaptation prohibited')
    original = planner.astar_virtual
    adapter = SpatialSearch(original, current, planner.CONFIG)
    _ACTIVE = True
    planner.astar_virtual = adapter
    try:
        yield adapter
    finally:
        planner.astar_virtual = original
        _ACTIVE = False


def generate(planner, inputs, instruction, model, prior, arm):
    # Only observation tensors, language, and the shared TRAIN fit are accepted.
    if arm == 'edge':
        return planner.plan_observation(*inputs, instruction, model, prior)
    if arm != 'spatial':
        raise ValueError('Unknown baseline arm')
    with spatial_adapter(planner, inputs[6]) as adapter:
        paths, raw, record = planner.plan_observation(*inputs, instruction, model, prior)
    record['spatial_adapter'] = dict(protocol=PROTOCOL, actual_astar_calls=adapter.calls,
        previous_complete_raw_paths=len(adapter.raw_paths), sigma_m=SIGMA_M,
        preprocessing_seconds=sum(a.get('spatial_penalty', {}).get('preprocessing_seconds', 0.) for a in record['attempts']),
        cost='length * (1 + 4 * mean(endpoint field)); sum Gaussian exact-distance-to-each-prior-raw-polyline',
        virtual_edge_cost='unchanged original exact edge reuse', evaluation_labels_used=False)
    return paths, raw, record


def label_for(data, row):
    """Decode only this ID; never decode unrelated DEV label payloads for TRAIN."""
    with (Path(data)/'supervision.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            match = re.search(r'"id"\s*:\s*("(?:[^"\\]|\\.)*")', line)
            if match is None:
                raise ValueError('Supervision row lacks an ID')
            if json.loads(match.group(1)) != row['id']:
                continue
            label = json.loads(line)
            if label['parent_id'] != row['parent_id'] or label['split'] != row['split']:
                raise ValueError('Label identity changed')
            return label
    raise ValueError('Registered supervision row missing')


def fit_train(data, observations, manifest, planner):
    began = time.perf_counter()
    train = [r for r in observations if r['split'] == 'TRAIN']
    labels, hashes = {}, {}
    for row in train:
        label = label_for(data, row)
        pointer = str(Path(row['image']).with_name('observation.npz'))
        if label['observation'] != pointer:
            raise ValueError('Current observation pointer changed')
        labels[row['id']] = dict(observation=pointer, routes=list(label['routes']))
        for path in [row['image'], pointer]+label['routes']:
            control.checked(path, manifest, hashes)
    model, parents = planner.prototype.fit_prototypes(Path(data), train, labels)
    prior = planner.v1.fit_workspace(Path(data), train, labels)
    for fitted in (model['training_source_sha256'], prior['source_sha256']):
        if any(hashes.get(path) != value for path, value in fitted.items()):
            raise ValueError('Fitter read outside exact TRAIN source set')
    return model, prior, dict(elapsed_seconds=time.perf_counter()-began, parents=sorted(parents),
        observed_train_inputs=len(train), positive_reference_occurrences=sum(len(x['routes']) for x in labels.values()),
        source_files_sha256=hashes, semantic_targets_or_route_types_passed_to_fitters=False)


def load_current(row, manifest, hashes, planner):
    if set(row) != exporter.INPUT_KEYS or row['split'] not in ('TRAIN', 'DEV_MODEL'):
        raise ValueError('Strict registered observation input required')
    start = time.perf_counter()
    image = control.checked(row['image'], manifest, hashes)
    pointer = control.checked(image.with_name('observation.npz'), manifest, hashes)
    rgb = np.asarray(Image.open(image).convert('RGB'))
    with np.load(pointer, allow_pickle=False) as archive:
        if set(archive.files) != exporter.CURRENT_KEYS:
            raise ValueError('Observation contains undeclared fields')
        current = {key: archive[key].copy() for key in archive.files}
    io = time.perf_counter()-start
    start = time.perf_counter()
    color, xyz, valid = planner.prototype.observed_grid(rgb, current['depth'], current['camera_intrinsics'], current['camera_extrinsics'])
    state = np.r_[current['gripper_pose'], np.asarray(current['gripper_open']).reshape(1)]
    return (color, xyz, valid, current['depth'], current['camera_intrinsics'], current['camera_extrinsics'], state), current, dict(
        observation_io_hash_seconds=io, backprojection_seconds=time.perf_counter()-start)


def one_request(data, row, manifest, model, prior, planner, output, arm):
    start = time.perf_counter()
    hashes, output = {}, Path(output)
    output.mkdir(parents=True, exist_ok=False)
    control.write_json(output/'status.json', dict(status='running', id=row['id'], split=row['split'], arm=arm, reserved_slots=4))
    inputs, current, timing = load_current(row, manifest, hashes, planner)
    began = time.perf_counter()
    paths, raw, record = generate(planner, inputs, row['instruction'], model, prior, arm)
    control.validate_pool(paths, raw, record)
    timing['localization_grid_field_search_and_original_proxy_seconds'] = time.perf_counter()-began
    opened = np.full((4, 24), np.asarray(current['gripper_open']).item())
    began = time.perf_counter()
    prediction, raw_file = output/'predictions.npz', output/'raw_paths.npz'
    np.savez_compressed(prediction, paths=paths[None], gripper_open=opened[None], scene_ids=[row['id']], parent_ids=[row['parent_id']])
    np.savez_compressed(raw_file, **{'candidate_%d'%i: np.empty((0, 3)) if path is None else path for i, path in enumerate(raw)})
    phash, rhash = control.digest(prediction), control.digest(raw_file)
    control.write_json(output/'generation_seal.json', dict(id=row['id'], split=row['split'], arm=arm,
        prediction_sha256=phash, raw_paths_sha256=rhash, submitted_candidate_budget=4,
        raw_complete_paths=record['raw_complete_paths'], evaluation_labels_opened=False))
    timing['output_sealing_seconds'] = time.perf_counter()-began
    timing['observation_to_sealed_pool_seconds'] = time.perf_counter()-start
    began = time.perf_counter()
    label = label_for(data, row)
    with np.load(control.checked(label['verification_only'], manifest, hashes), allow_pickle=False) as archive:
        geometry = {key: archive[key].copy() for key in ('obstacle_centers', 'obstacle_halfsizes')}
    route_config = json.loads(control.checked(label['route_config'], manifest, hashes).read_text())
    metric, candidates = control.scene_metrics(paths, opened, current, geometry, label['semantic_targets'], label['route_types'], route_config)
    timing['label_io_and_two_row_check_seconds'] = time.perf_counter()-began
    timing['observation_to_checked_pool_seconds'] = time.perf_counter()-start
    result = dict(id=row['id'], parent_id=row['parent_id'], split=row['split'], arm=arm, metrics=metric, candidates=candidates,
        reference_count=len(label['routes']), known_reference_count=sum(t is not None for t in label['route_types']),
        generation=record, timing=timing, prediction_sha256=phash, raw_paths_sha256=rhash,
        source_files_sha256=hashes, scored_after_sealed_pool=True, no_output_filtering_or_repair=True)
    control.write_json(output/'result.json', result)
    control.write_json(output/'status.json', dict(status='completed', id=row['id'], split=row['split'], arm=arm,
        submitted_candidate_budget=4, raw_complete_paths=record['raw_complete_paths'], failed_slots=record['failed_slots']))
    return result, paths, opened


def validate_registration(config, manifest, observations, stage):
    if (config.get('protocol') != PROTOCOL or config.get('sigma_m') != SIGMA_M or config.get('voxel_m') != .025
            or config.get('grid_edge_penalty_multiplier') != 4 or config.get('candidates') != 4
            or config.get('horizon') != 24 or config.get('maximum_expanded_nodes_per_candidate') != 20000
            or config.get('search_deadline_seconds') != 2 or config.get('registered_train_indices') != list(range(32))
            or config.get('train_preflight_indices') != list(range(4)) or config.get('dev_indices') != list(range(64, 76))):
        raise ValueError('Fixed spatial experiment registration changed')
    if exporter.selection_guard(manifest['selection']) != 32:
        raise ValueError('Same fixed32 TRAIN export required')
    allowed = {'TRAIN': set('two_row_reach_%d'%(283200+i) for i in range(32)),
               'DEV_MODEL': set('two_row_reach_%d'%(283200+i) for i in range(64, 76))}
    seen = set()
    for row in observations:
        if (set(row) != exporter.INPUT_KEYS or row['split'] not in allowed or row['parent_id'] not in allowed[row['split']]
                or row['id'] not in [row['parent_id']+'_target%d'%t for t in range(3)] or row['id'] in seen):
            raise ValueError('Observation identity/role/whitelist changed')
        seen.add(row['id'])
    if stage == 'train_preflight':
        selected = [r for r in observations if r['split'] == 'TRAIN' and r['parent_id'] in sorted(allowed['TRAIN'])[:4]]
        expected = {p+'_target%d'%t for p in sorted(allowed['TRAIN'])[:4] for t in range(3)}
        if {r['id'] for r in selected} != expected:
            raise ValueError('Exactly registered first4 TRAIN parents /12 inputs required')
    elif stage == 'dev':
        selected = [r for r in observations if r['split'] == 'DEV_MODEL']
        expected = {p+'_target%d'%t for p in allowed['DEV_MODEL'] for t in range(3)}
        if {r['id'] for r in selected} != expected:
            raise ValueError('Exactly fixed36 DEV inputs required; no replacement')
    else:
        raise ValueError('Unknown stage')
    return sorted(selected, key=lambda r: r['id'])


def verify_export_metadata(data, config):
    """Closed mechanical gate + export hashes; raw files checked only on use."""
    path = Path(data)/'export_manifest.json'
    if control.digest(path) != config['export_manifest_sha256']:
        raise ValueError('Different fixed32 export')
    manifest = json.loads(path.read_text())
    if manifest['protocol'] != exporter.PROTOCOL:
        raise ValueError('Wrong export protocol')
    for name, value in manifest['output_files_sha256'].items():
        if name not in exporter.OUTPUT_FILES or control.digest(Path(data)/name) != value:
            raise ValueError('Export metadata changed')
    if set(manifest['output_files_sha256']) != set(exporter.OUTPUT_FILES):
        raise ValueError('Incomplete export metadata hashes')
    _, _, plans, _, missing = exporter.closed_prefix(manifest['source_dataset'], manifest['selection'])
    gate = exporter.collector.live_layout_gate(Path(manifest['source_dataset']))
    if missing or set(gate['blocked_parent_ids']) & {p['parent_id'] for p in plans}:
        raise ValueError('Current model-use gate blocked')
    return manifest, gate


def aggregate(records):
    result = {key: float(np.mean([r['metrics'][key] for r in records if r['metrics'][key] is not None]))
              if any(r['metrics'][key] is not None for r in records) else None for key in control.METRICS}
    latency = [r['timing']['observation_to_checked_pool_seconds'] for r in records]
    result.update(examples=len(records), submitted_candidate_budget=4*len(records),
        raw_complete_paths=sum(r['generation']['raw_complete_paths'] for r in records),
        failed_candidate_slots=sum(r['generation']['failed_slots'] for r in records),
        reference_evaluation_examples=sum(r['reference_count'] > 0 for r in records),
        known_reference_coverage_evaluation_examples=sum(r['metrics']['known_reference_types'] > 0 for r in records),
        request_latency_seconds=dict(first=latency[0], median=float(np.median(latency)), p95=float(np.quantile(latency, .95)), total=sum(latency)),
        field_preprocessing_seconds=sum(r['generation'].get('spatial_adapter', {}).get('preprocessing_seconds', 0.) for r in records))
    return result


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def fit_identity(model, prior):
    # The old fitter returns its wall time inside the model metadata. Only this
    # measured runtime is excluded; all actual parameters, rows, and hashes stay.
    parameters = dict(model)
    elapsed = parameters.pop('training_seconds')
    if not np.isfinite(elapsed) or elapsed < 0:
        raise ValueError('Finite actual fitter runtime required')
    return canonical(dict(prototype=parameters, workspace=prior))


def run(data, output, config_path, stage, preflight=None):
    data, output, config_path = Path(data), Path(output), Path(config_path)
    if output.exists():
        raise FileExistsError('Fresh output required; no replay/resume or automatic retries')
    began = time.perf_counter()
    config = json.loads(config_path.read_text())
    manifest, gate = verify_export_metadata(data, config)
    observations = [json.loads(line) for line in (data/'observations.jsonl').read_text().splitlines()]
    selected = validate_registration(config, manifest, observations, stage)
    planner = control.dependencies()
    source_paths = [Path(__file__), config_path, Path(control.__file__), Path(exporter.__file__)]
    source_paths += [Path(__file__).with_name(name) for name in list(control.FROZEN_SOURCES)+[
        'evaluate_observed_two_row.py', 'evaluate_observed_obstacles.py', 'collect_observed_two_row_pilot.py',
        'collect_obstacle_reach.py', 'collect_two_row_formal.py', 'snapshot_multitask_observations.py',
        'launch_observed_two_row_spatial_penalty_v1.sh']]
    source = {str(p): control.digest(p) for p in source_paths}
    source[str(data/'export_manifest.json')] = control.digest(data/'export_manifest.json')
    if stage == 'dev':
        if preflight is None:
            raise ValueError('Sealed TRAIN mechanical preflight required before DEV')
        previous = json.loads(Path(preflight).read_text())
        if (previous.get('protocol') != PROTOCOL or previous.get('stage') != 'train_preflight'
                or previous.get('status') != 'completed' or previous.get('config_sha256') != control.digest(config_path)
                or previous.get('mechanical_preflight_passed') is not True
                or any(previous.get('source_files_sha256', {}).get(k) != v for k, v in source.items())):
            raise ValueError('TRAIN preflight source/config/mechanical receipt differs')
        for path, value in previous['artifact_sha256'].items():
            if control.digest(path) != value:
                raise ValueError('TRAIN preflight artifact changed')
        source[str(Path(preflight))] = control.digest(preflight)
    model, prior, fit = fit_train(data, observations, manifest, planner)
    if fit['observed_train_inputs'] != 93 or len(fit['parents']) != 31:
        raise ValueError('Original fixed32 fit population (one missing parent) changed')
    source.update(fit['source_files_sha256'])
    fitted_sha = fit_identity(model, prior)
    if fitted_sha != config['original_shared_fit_sha256']:
        raise ValueError('Prototype/workspace differs from actual original fixed32 baseline')
    if stage == 'dev' and fitted_sha != previous['shared_fit_sha256']:
        raise ValueError('TRAIN fit changed after preflight')
    output.mkdir(parents=True)
    control.write_json(output/'fitted_train_model.json', dict(prototype=model, workspace=prior, fit=fit,
        canonical_sha256=fitted_sha, canonical_excludes_only='prototype.training_seconds'))
    records = {'edge': [], 'spatial': []} if stage == 'train_preflight' else {'spatial': []}
    pools = {arm: [] for arm in records}
    first_equal, raw_first_equal = [], []
    requested_requests = len(selected)*len(records)
    try:
        for row in selected:
            pair = {}
            for arm in records:
                result, paths, events = one_request(data, row, manifest, model, prior, planner, output/arm/row['id'], arm)
                records[arm].append(result)
                pools[arm].append((paths, events))
                source.update(result['source_files_sha256'])
                pair[arm] = paths
                print(json.dumps(dict(stage=stage, arm=arm, id=row['id'], failed_slots=result['generation']['failed_slots'],
                    seconds=result['timing']['observation_to_checked_pool_seconds'])), flush=True)
            if stage == 'train_preflight':
                equal = bool(np.array_equal(pair['edge'][0], pair['spatial'][0], equal_nan=True))
                with np.load(output/'edge'/row['id']/'raw_paths.npz') as a, np.load(output/'spatial'/row['id']/'raw_paths.npz') as b:
                    raw_equal = bool(np.array_equal(a['candidate_0'], b['candidate_0'], equal_nan=True))
                first_equal.append(dict(id=row['id'], equal=equal))
                raw_first_equal.append(dict(id=row['id'], equal=raw_equal))
                if not equal or not raw_equal:
                    raise ValueError('Zero-field first path differs (including possible time-budget jitter); stop, no retuning')
        for arm, pool in pools.items():
            np.savez_compressed(output/(arm+'_predictions.npz'), paths=np.stack([p[0] for p in pool]),
                gripper_open=np.stack([p[1] for p in pool]), scene_ids=[r['id'] for r in selected], parent_ids=[r['parent_id'] for r in selected])
        _, final_gate = verify_export_metadata(data, config)
        if any(control.digest(path) != value for path, value in source.items()):
            raise ValueError('Frozen source/input changed during baseline')
        report = dict(protocol=PROTOCOL, status='completed', stage=stage, config_sha256=control.digest(config_path),
            config=config, planner_config=planner.CONFIG, prototype_config=planner.prototype.CONFIG,
            requested_inputs_per_arm=len(selected), requested_candidate_slots=4*requested_requests,
            completed_requests=requested_requests, failed_requests=0, unattempted_requests=0,
            train_fit=fit, shared_fit_sha256=fitted_sha, registered_train_parents=32, actual_train_parents=31,
            input_ids=[r['id'] for r in selected], results={arm: aggregate(rows) for arm, rows in records.items()},
            per_request=records, first_h24_equal=first_equal, first_raw_equal=raw_first_equal,
            mechanical_preflight_passed=(stage == 'train_preflight' and all(x['equal'] for x in first_equal+raw_first_equal)),
            elapsed_seconds=time.perf_counter()-began, training_steps=0, new_qwen_calls=0, gpu_hours=0,
            prior_preflight_sha256=source.get(str(Path(preflight))) if preflight else None,
            source_files_sha256=source, initial_mechanical_gate=gate, final_mechanical_gate=final_gate,
            artifact_sha256={str(p): control.digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
            scope='Fixed K=4; field preprocessing is additional compute counted in request wall time; no fixed-time superiority claim',
            decision_scope='Mechanical preflight only, not a threshold tuned on TRAIN metrics; DEV is a separately launched fixed evaluation')
        control.write_json(output/'report.json', report)
        return report
    except Exception:
        completed = sum(len(rows) for rows in records.values())
        control.write_json(output/'failure.json', dict(status='failed', protocol=PROTOCOL, stage=stage,
            traceback=traceback.format_exc(), requested_requests=requested_requests, reserved_candidate_slots=4*requested_requests,
            completed_requests=completed, interrupted_request_upper_bound_slots=4,
            later_unattempted_requests=max(0, requested_requests-completed-1),
            automatic_retries=0, elapsed_seconds=time.perf_counter()-began))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('train_preflight', 'dev'), required=True)
    parser.add_argument('--preflight', type=Path)
    args = parser.parse_args()
    result = run(args.data, args.output, args.config, args.stage, args.preflight)
    print(json.dumps({k: result[k] for k in ('protocol', 'status', 'stage', 'completed_requests', 'elapsed_seconds')}))
