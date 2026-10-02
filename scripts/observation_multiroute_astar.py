"""Observation-only prototype endpoint plus four bounded voxel A* searches.

This is a traditional control, not a learned contribution or robot certificate.
Unknown space is blocked, except the explicitly recorded local current-tip and
predicted-target contact allowances. Evaluation labels are opened only after
all candidates have been emitted. Failed slots remain NaN in the K=4 budget.
"""
import argparse
from collections import Counter
import heapq
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy import ndimage

from scripts import observation_prototype_grounding as prototype


CONFIG = dict(voxel_m=.025, visible_surface_clearance_m=.02,
    train_workspace_margin_m=.06, maximum_grid_cells=2_000_000,
    current_tip_contact_radius_m=.04, target_contact_depth_m=.03,
    target_contact_radius_rule='half TRAIN prototype extent plus one voxel',
    maximum_expanded_nodes_per_candidate=20_000, search_deadline_seconds=2.,
    used_edge_penalty_multiplier=4., candidates=4, horizon=24,
    unknown_policy='blocked except recorded local current/target contact allowances',
    collision_scope='discrete visible-free-space proxy, not a continuous 2cm physical-clearance certificate; depth projection and surface/grid quantization errors are not jointly bounded',
    diagonal_policy='all touched neighbor voxels, including edge/corner ties, must be free')


def fit_workspace(dataset, observations, labels):
    """Bounds depend on TRAIN demonstration support, never DEV truth."""
    minimum, maximum, count = np.full(3, np.inf), np.full(3, -np.inf), 0
    hashes = {}
    for row in observations:
        if row['split'] != 'TRAIN':
            continue
        for name in labels[row['id']].get('routes', []):
            path = dataset / name
            with np.load(path, allow_pickle=False) as archive:
                xyz = archive['gripper_pose'][:, :3]
            if not np.isfinite(xyz).all():
                raise ValueError('nonfinite TRAIN trajectory in workspace prior')
            minimum = np.minimum(minimum, xyz.min(0))
            maximum = np.maximum(maximum, xyz.max(0))
            hashes[str(path)] = prototype.digest(path)
            count += 1
    if count == 0:
        raise ValueError('at least one TRAIN reference is required')
    margin = CONFIG['train_workspace_margin_m']
    return dict(lower=(minimum-margin).tolist(), upper=(maximum+margin).tolist(),
                train_routes=count, source_sha256=hashes, fit_scope='TRAIN paths only')


def selected_target_component(rgb, valid, instruction, model, details):
    if details.get('status') != 'predicted':
        return np.zeros(valid.shape, bool)
    item = model['prototypes'][instruction]
    distance = (((prototype.feature_colors(rgb)-np.asarray(item['center']))/
                 np.asarray(item['scale']))**2).mean(-1)
    components, _ = ndimage.label(valid & (distance <= item['threshold']),
                                  structure=np.ones((3, 3), np.uint8))
    return components == details['selected_component']


def grid_index(point, lower, voxel):
    return np.floor((np.asarray(point)-lower)/voxel+.5).astype(np.int64)


def build_grid(rgb, xyz, valid, depth, intrinsics, camera_to_world, current,
               endpoint, instruction, model, endpoint_details, prior):
    """Only unlabelled current observations and TRAIN-fitted quantities."""
    started = time.perf_counter()
    voxel = CONFIG['voxel_m']
    lower, upper = np.asarray(prior['lower']), np.asarray(prior['upper'])
    shape = tuple((np.ceil((upper-lower)/voxel).astype(int)+1).tolist())
    if np.prod(shape) > CONFIG['maximum_grid_cells']:
        raise ValueError('TRAIN workspace exceeds predeclared maximum grid cells')
    indices = np.indices(shape).reshape(3, -1).T
    world = lower + indices*voxel
    camera = (world-camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    projected = camera @ intrinsics.T
    z = camera[:, 2]
    safe_z = np.where(z > 0, z, 1.)
    pixels = np.rint(projected[:, :2]/safe_z[:, None]).astype(int)
    height, width = depth.shape
    in_view = (z > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    px = np.clip(pixels[:, 0], 0, width-1)
    py = np.clip(pixels[:, 1], 0, height-1)
    observed = in_view & valid[py, px]
    measured_depth = depth[py, px]
    visible_free = observed & (z < measured_depth-CONFIG['visible_surface_clearance_m'])
    component = selected_target_component(rgb, valid, instruction, model, endpoint_details)
    # Exempt only observed pixels in the predicted contact object, not arbitrary
    # depth points within a sphere around a supplied true endpoint.
    obstacle_points = xyz[valid & ~component]
    cells = grid_index(obstacle_points, lower, voxel)
    inside = np.all((cells >= 0) & (cells < np.asarray(shape)), axis=1)
    occupied = np.zeros(shape, bool)
    kept = cells[inside]
    if len(kept):
        occupied[tuple(kept.T)] = True
        distance = ndimage.distance_transform_edt(~occupied)*voxel
    else:
        distance = np.full(shape, np.inf)
    radius = CONFIG['visible_surface_clearance_m'] + np.sqrt(3)*voxel/2
    surface_blocked = distance <= radius
    target_radius = model['prototypes'][instruction]['extent_m']/2 + voxel
    target_permission = (observed & component[py, px] &
        (np.abs(z-measured_depth) <= CONFIG['target_contact_depth_m']) &
        (np.linalg.norm(world-endpoint, axis=-1) <= target_radius)).reshape(shape)
    start_permission = (np.linalg.norm(world-current[:3], axis=-1) <=
                        CONFIG['current_tip_contact_radius_m']).reshape(shape)
    # Start allowance corresponds only to the measured current tip vicinity.
    # Target permission still cannot remove another object's observed surface.
    ordinary_free = visible_free.reshape(shape) & ~surface_blocked
    free = (ordinary_free | target_permission) & ~surface_blocked
    free |= start_permission
    unknown = ~observed.reshape(shape) | (z.reshape(shape) > measured_depth.reshape(shape))
    metadata = dict(shape=list(shape), grid_nodes=int(np.prod(shape)), free_nodes=int(free.sum()),
        ordinary_free_nodes=int(ordinary_free.sum()), unknown_nodes=int(unknown.sum()),
        unknown_nodes_still_blocked=int((unknown & ~free).sum()),
        observed_surface_occupied_voxels=int(occupied.sum()),
        obstacle_observed_points=len(obstacle_points), obstacle_points_outside_workspace=int((~inside).sum()),
        target_component_pixels=int(component.sum()), lower=lower.tolist(),
        nominal_upper=upper.tolist(), effective_upper=(lower+(np.asarray(shape)-1)*voxel).tolist(),
        voxel_m=voxel, conservative_surface_inflation_m=float(radius),
        geometric_guarantee='continuous segment supercover is relative to this discrete proxy only, not true obstacles or full robot; cell-center depth projection and point quantization are approximations',
        contact_allowances=dict(current_tip_center=current[:3].tolist(), current_tip_radius_m=CONFIG['current_tip_contact_radius_m'],
            current_tip_region_voxels=int(start_permission.sum()), current_tip_newly_free_voxels=int((start_permission & ~ordinary_free).sum()),
            current_tip_unknown_voxels=int((start_permission & unknown).sum()),
            current_tip_voxel_indices=np.argwhere(start_permission).tolist(),
            predicted_endpoint=endpoint.tolist(), target_radius_m=float(target_radius), target_depth_shell_m=CONFIG['target_contact_depth_m'],
            target_region_voxels=int(target_permission.sum()), target_newly_free_voxels=int((target_permission & ~ordinary_free & ~surface_blocked).sum()),
            target_unknown_voxels=int((target_permission & unknown & ~surface_blocked).sum()),
            target_permitted_voxel_indices=np.argwhere(target_permission & ~surface_blocked).tolist()),
        grid_construction_seconds=time.perf_counter()-started)
    return free, lower, start_permission, target_permission, metadata


def neighbor_specification():
    result = []
    for delta in itertools.product((-1, 0, 1), repeat=3):
        if not any(delta):
            continue
        active = [axis for axis in range(3) if delta[axis]]
        touched = []
        for flags in itertools.product((0, 1), repeat=len(active)):
            if not any(flags):
                continue
            offset = [0, 0, 0]
            for axis, flag in zip(active, flags):
                offset[axis] = delta[axis]*flag
            touched.append(tuple(offset))
        result.append((delta, tuple(touched), float(np.linalg.norm(delta))))
    return result


NEIGHBORS = neighbor_specification()


def astar(free, start, goal, used_edges, contact_start=None, contact_target=None):
    """One bounded search is exactly one submitted route slot, even on failure."""
    begin = time.perf_counter()
    shape, voxel = free.shape, CONFIG['voxel_m']
    statistics = dict(expanded_nodes=0, neighbor_edges_examined=0, free_edges_examined=0,
        diagonal_supercover_voxels_examined=0, allowed_start_region_edge_checks=0,
        allowed_target_region_edge_checks=0, maximum_expanded_nodes=CONFIG['maximum_expanded_nodes_per_candidate'],
        search_deadline_seconds=CONFIG['search_deadline_seconds'])
    def finish(path, status):
        statistics.update(status=status, search_seconds=time.perf_counter()-begin,
                          complete_raw_paths_emitted=int(path is not None))
        return path, statistics
    if np.any(start < 0) or np.any(goal < 0) or np.any(start >= shape) or np.any(goal >= shape):
        return finish(None, 'endpoint_outside_train_workspace')
    start, goal = tuple(start), tuple(goal)
    if not free[start] or not free[goal]:
        return finish(None, 'start_or_predicted_endpoint_not_traversable')
    counter = itertools.count()
    queue = [(0., next(counter), start)]
    scores, previous, closed = {start: 0.}, {}, set()
    while queue:
        _, _, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal:
            path = [current]
            while path[-1] != start:
                path.append(previous[path[-1]])
            path.reverse()
            statistics['route_edges_in_start_allowance'] = sum(bool(contact_start[a] or contact_start[b]) for a, b in zip(path, path[1:])) if contact_start is not None else 0
            statistics['route_edges_in_target_allowance'] = sum(bool(contact_target[a] or contact_target[b]) for a, b in zip(path, path[1:])) if contact_target is not None else 0
            return finish(path, 'path_found')
        closed.add(current)
        statistics['expanded_nodes'] += 1
        if statistics['expanded_nodes'] >= CONFIG['maximum_expanded_nodes_per_candidate']:
            return finish(None, 'expanded_node_budget_exhausted')
        if statistics['expanded_nodes'] % 64 == 0 and time.perf_counter()-begin >= CONFIG['search_deadline_seconds']:
            return finish(None, 'search_time_budget_exhausted')
        for delta, touched, length in NEIGHBORS:
            statistics['neighbor_edges_examined'] += 1
            neighbor = tuple(current[axis]+delta[axis] for axis in range(3))
            if any(neighbor[axis] < 0 or neighbor[axis] >= shape[axis] for axis in range(3)):
                continue
            blocked = False
            for offset in touched:
                statistics['diagonal_supercover_voxels_examined'] += 1
                cell = tuple(current[axis]+offset[axis] for axis in range(3))
                if not free[cell]:
                    blocked = True
                    break
            if blocked:
                continue
            statistics['free_edges_examined'] += 1
            if contact_start is not None and (contact_start[current] or contact_start[neighbor]):
                statistics['allowed_start_region_edge_checks'] += 1
            if contact_target is not None and (contact_target[current] or contact_target[neighbor]):
                statistics['allowed_target_region_edge_checks'] += 1
            edge = tuple(sorted((current, neighbor)))
            cost = length*voxel*(1+CONFIG['used_edge_penalty_multiplier']*used_edges[edge])
            proposed = scores[current]+cost
            if proposed < scores.get(neighbor, np.inf):
                scores[neighbor], previous[neighbor] = proposed, current
                heuristic = np.linalg.norm(np.asarray(neighbor)-goal)*voxel
                heapq.heappush(queue, (proposed+heuristic, next(counter), neighbor))
    return finish(None, 'open_set_exhausted')


def segment_supercover(first, second, lower, voxel):
    """All voxels touched by a segment, including simultaneous boundary ties."""
    a, b = (np.asarray(first)-lower)/voxel+.5, (np.asarray(second)-lower)/voxel+.5
    delta = b-a
    breaks = [0., 1.]
    for axis in range(3):
        if abs(delta[axis]) < 1e-12:
            continue
        for plane in range(int(np.ceil(min(a[axis], b[axis]))), int(np.floor(max(a[axis], b[axis])))+1):
            parameter = (plane-a[axis])/delta[axis]
            if 0. < parameter < 1.:
                breaks.append(float(parameter))
    breaks = sorted(set(breaks))
    probes = breaks + [(x+y)/2 for x, y in zip(breaks, breaks[1:])]
    cells = set()
    for parameter in probes:
        position = a+parameter*delta
        choices = []
        for value in position:
            nearest = round(float(value))
            choices.append((nearest-1, nearest) if abs(value-nearest) <= 1e-8 else (int(np.floor(value)),))
        cells.update(itertools.product(*choices))
    return cells


def path_observed_clear(path, free, lower):
    checked = 0
    for first, second in zip(path, path[1:]):
        for cell in segment_supercover(first, second, lower, CONFIG['voxel_m']):
            checked += 1
            if any(cell[axis] < 0 or cell[axis] >= free.shape[axis] for axis in range(3)) or not free[cell]:
                return False, checked
    return True, checked


def resample(path, horizon=24):
    cumulative = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=-1))]
    keep = np.r_[True, np.diff(cumulative) > 1e-10]
    if cumulative[-1] <= 1e-10:
        return np.repeat(path[:1], horizon, axis=0)
    values = np.linspace(0, cumulative[-1], horizon)
    return np.stack([np.interp(values, cumulative[keep], path[keep, axis]) for axis in range(3)], -1)


def plan_observation(rgb, xyz, valid, depth, intrinsics, camera_to_world, current,
                     instruction, model, prior):
    started = time.perf_counter()
    endpoint, localization = prototype.predict(rgb, xyz, valid, instruction, model)
    localization_seconds = time.perf_counter()-started
    paths = np.full((CONFIG['candidates'], CONFIG['horizon'], 3), np.nan)
    raw_paths, attempts = [], []
    if endpoint is None:
        return paths, [None]*CONFIG['candidates'], dict(localization=localization,
            localization_seconds=localization_seconds, attempts=[dict(slot=i, status='localization_failure', complete_raw_paths_emitted=0) for i in range(CONFIG['candidates'])],
            submitted_candidate_budget=CONFIG['candidates'], request_generation_seconds=time.perf_counter()-started)
    try:
        free, lower, start_permission, target_permission, grid = build_grid(rgb, xyz, valid, depth,
            intrinsics, camera_to_world, current, endpoint, instruction, model, localization, prior)
    except ValueError as error:
        return paths, [None]*CONFIG['candidates'], dict(localization=localization, localization_seconds=localization_seconds,
            attempts=[dict(slot=i, status='grid_construction_rejected', error=str(error), complete_raw_paths_emitted=0) for i in range(CONFIG['candidates'])],
            submitted_candidate_budget=CONFIG['candidates'], request_generation_seconds=time.perf_counter()-started)
    start, goal = grid_index(current[:3], lower, CONFIG['voxel_m']), grid_index(endpoint, lower, CONFIG['voxel_m'])
    used_edges, earlier = Counter(), set()
    for slot in range(CONFIG['candidates']):
        cells, information = astar(free, start, goal, used_edges, start_permission, target_permission)
        information['slot'] = slot
        if cells is None:
            raw_paths.append(None)
            attempts.append(information)
            continue
        # The raw complete path remains one candidate even if subsequent
        # endpoint connectors or H24 resampling fail an observed-space check.
        emission_started = time.perf_counter()
        raw = np.vstack([current[:3], lower+np.asarray(cells)*CONFIG['voxel_m'], endpoint])
        sampled = resample(raw, CONFIG['horizon'])
        emission_seconds = time.perf_counter()-emission_started
        postprocess = time.perf_counter()
        raw_clear, raw_checks = path_observed_clear(raw, free, lower)
        sampled_clear, sampled_checks = path_observed_clear(sampled, free, lower)
        signature = tuple(cells)
        information.update(raw_point_count=len(raw), raw_observed_clear=bool(raw_clear), h24_observed_clear=bool(sampled_clear),
            resampling_changed_observed_clearance=bool(raw_clear != sampled_clear),
            raw_voxels_checked=raw_checks, h24_voxels_checked=sampled_checks,
            exact_grid_path_duplicate=signature in earlier, observed_path_check_seconds=time.perf_counter()-postprocess,
            path_materialization_resampling_seconds=emission_seconds,
            raw_length_m=float(np.linalg.norm(np.diff(raw, axis=0), axis=-1).sum()),
            h24_length_m=float(np.linalg.norm(np.diff(sampled, axis=0), axis=-1).sum()))
        paths[slot], raw_paths = sampled, raw_paths+[raw]
        earlier.add(signature)
        for first, second in zip(cells, cells[1:]):
            used_edges[tuple(sorted((first, second)))] += 1
        attempts.append(information)
    return paths, raw_paths, dict(localization=localization, localization_seconds=localization_seconds,
        grid=grid, attempts=attempts, submitted_candidate_budget=CONFIG['candidates'],
        request_generation_seconds=time.perf_counter()-started,
        failed_slots=sum(path is None for path in raw_paths), raw_complete_paths=sum(path is not None for path in raw_paths),
        filtering_or_repair=False)


def run(data, output, train_benchmark_only=False):
    if output.exists():
        raise FileExistsError('preserve existing run: '+str(output))
    observations = prototype.read_rows(data/'observations.jsonl')
    label_rows = prototype.read_rows(data/'supervision.jsonl')
    labels = {row['id']: row for row in label_rows}
    if len(labels) != len(label_rows) or len({row['id'] for row in observations}) != len(observations):
        raise ValueError('duplicate manifest IDs')
    parent_splits = {}
    for row in observations:
        if set(row) != prototype.INPUT_KEYS or labels[row['id']]['split'] != row['split']:
            raise ValueError('input contract or split mismatch')
        if parent_splits.setdefault(row['parent_id'], row['split']) != row['split']:
            raise ValueError('parent split leakage')
    training_started = time.perf_counter()
    model, train_parents = prototype.fit_prototypes(data, observations, labels)
    prior = fit_workspace(data, observations, labels)
    training_seconds = time.perf_counter()-training_started
    split = 'TRAIN' if train_benchmark_only else 'DEV_MODEL'
    chosen = sorted([row for row in observations if row['split'] == split], key=lambda row: row['id'])
    if train_benchmark_only:
        chosen = chosen[:1]
    elif train_parents & {row['parent_id'] for row in chosen}:
        raise ValueError('TRAIN/DEV parent overlap')
    if not chosen:
        raise ValueError('no requested observations')
    output.mkdir(parents=True)
    predictions, events, records, source_hashes = [], [], [], {}
    for row in chosen:
        request_started = time.perf_counter()
        # Only the observation pointer is taken from the manifest here. DEV
        # route/target/box truth is not passed to any generation function.
        observation_path = data/labels[row['id']]['observation']
        (rgb, xyz, valid), observation_files = prototype.load_observation(data, row, labels[row['id']]['observation'])
        with np.load(observation_path, allow_pickle=False) as archive:
            current = np.r_[archive['gripper_pose'], np.asarray(archive['gripper_open']).reshape(1)]
            depth, intrinsics, camera_to_world = archive['depth'], archive['camera_intrinsics'], archive['camera_extrinsics']
        preprocessing_seconds = time.perf_counter()-request_started
        paths, raw_paths, record = plan_observation(rgb, xyz, valid, depth, intrinsics, camera_to_world,
            current, row['instruction'], model, prior)
        record.update(id=row['id'], parent_id=row['parent_id'], preprocessing_seconds=preprocessing_seconds,
            total_observation_to_routes_seconds=time.perf_counter()-request_started)
        arrays = {f'candidate_{slot}': path if path is not None else np.empty((0, 3)) for slot, path in enumerate(raw_paths)}
        np.savez_compressed(output/(row['id']+'_raw.npz'), **arrays)
        records.append(record)
        predictions.append(paths)
        events.append(np.full(paths.shape[:2], current[-1]))
        source_hashes.update({str(path): prototype.digest(path) for path in observation_files})
        print(json.dumps(dict(id=row['id'], failed_slots=record.get('failed_slots', CONFIG['candidates']),
            request_seconds=record['total_observation_to_routes_seconds'], search_status=[item['status'] for item in record['attempts']])), flush=True)
        (output/'generation_progress.json').write_text(json.dumps(records, indent=2, allow_nan=False))
    prediction_file = output/'predictions.npz'
    np.savez_compressed(prediction_file, paths=np.stack(predictions), gripper_open=np.stack(events),
        scene_ids=np.asarray([row['id'] for row in chosen]), parent_ids=np.asarray([row['parent_id'] for row in chosen]))
    evaluation_started = time.perf_counter()
    # Historical evaluator helpers contain direct sibling imports. Resolve
    # those siblings from this exact prototype/source package, and import the
    # privileged evaluator only after all output candidates have been saved.
    sys.path.insert(0, str(Path(prototype.__file__).resolve().parent))
    try:
        from scripts.evaluate_observed_obstacles import evaluate
        metrics = evaluate(data, prediction_file, output/'tip_evaluation', split=split, allow_subset=train_benchmark_only)
    finally:
        sys.path.pop(0)
    report = dict(baseline='TRAIN prototype localization + observed voxel A* with used-edge penalty',
        config=CONFIG, prototype_config=prototype.CONFIG, split=split, subset_training_cost_diagnostic=train_benchmark_only,
        examples=len(chosen), submitted_candidate_budget=len(chosen)*CONFIG['candidates'],
        input_fields=['RGB', 'metric depth', 'camera calibration', 'current gripper pose/open', 'exact instruction'],
        privileged_inference_input=False, train_workspace_prior=prior, prototype_model=model,
        training_seconds=training_seconds, source_sha256=source_hashes, script_sha256=prototype.digest(__file__),
        records=records, fixed_tip_evaluation=metrics, evaluation_seconds=time.perf_counter()-evaluation_started,
        limitation='Unknown space is conservatively blocked except explicitly recorded local contact allowances. Grid clearance is a discrete proxy, not a continuous physical 2cm certificate: cell-center depth tests and surface/segment quantization errors are not jointly bounded. Visible points do not reconstruct hidden solids or certify full arm/IK/execution. Independent unchanged true-box evaluation may reject proxy-clear paths.',
        selection_or_repair=False, scored_only_after_all_candidates_emitted=True)
    (output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    return report


def self_test():
    free = np.ones((7, 7, 3), bool)
    free[3, :, :] = False
    free[3, 1, 1] = free[3, 5, 1] = True
    used = Counter()
    first, stats = astar(free, np.array([1, 3, 1]), np.array([5, 3, 1]), used)
    assert first is not None and stats['complete_raw_paths_emitted'] == 1
    for a, b in zip(first, first[1:]):
        used[tuple(sorted((a, b)))] += 1
    second, _ = astar(free, np.array([1, 3, 1]), np.array([5, 3, 1]), used)
    assert first != second
    # A diagonal touching either blocked corner neighbor is not an edge.
    corner = np.zeros((2, 2, 1), bool)
    corner[0, 0, 0] = corner[1, 1, 0] = True
    missing, result = astar(corner, np.array([0, 0, 0]), np.array([1, 1, 0]), Counter())
    assert missing is None and result['status'] == 'open_set_exhausted'
    cells = segment_supercover(np.zeros(3), np.array([1., 1., 0.]), np.zeros(3), 1.)
    assert (1, 0, 0) in cells and (0, 1, 0) in cells
    raw = np.array([[0., 0., 0.], [0., .1, 0.], [.1, .1, 0.]])
    sampled = resample(raw)
    assert sampled.shape == (24, 3) and np.array_equal(sampled[0], raw[0]) and np.array_equal(sampled[-1], raw[-1])
    all_blocked = np.zeros((2, 2, 2), bool)
    missing, _ = astar(all_blocked, np.zeros(3, int), np.ones(3, int), Counter())
    assert missing is None
    # A real observed depth plane supplies free space in front; points behind
    # it remain unknown/blocked without a start or predicted contact exception.
    depth = np.ones((2, 2))
    calibration = np.diag([2., 2., 1.])
    rgb, xyz, valid = prototype.observed_grid(np.zeros((2, 2, 3), np.uint8), depth, calibration, np.eye(4))
    model = {'prototypes': {'fixture': {'extent_m': .03}}}
    grid, lower, _, _, audit = build_grid(rgb, xyz, valid, depth, calibration, np.eye(4),
        np.array([-1., -1., -1., 0., 0., 0., 1., 1.]), np.array([-1., 0., 1.]), 'fixture', model,
        {'status': 'unsupported_exact_instruction'}, {'lower': [-.05, -.05, .9], 'upper': [.05, .05, 1.2]})
    assert grid[2, 2, 0] and not grid[2, 2, 10]
    assert audit['contact_allowances']['current_tip_region_voxels'] == 0
    assert audit['unknown_nodes_still_blocked'] > 0
    print('A* self-test passed: two passages, edge penalty, conservative diagonal, continuous supercover, H24 endpoints, failed search, observed plane and blocked unknown')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-benchmark-only', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.data is None or args.output is None:
        parser.error('--data and --output required')
    else:
        result = run(args.data, args.output, args.train_benchmark_only)
        print(json.dumps(dict(examples=result['examples'], budget=result['submitted_candidate_budget'], metrics=result['fixed_tip_evaluation'])), flush=True)


if __name__ == '__main__':
    main()
