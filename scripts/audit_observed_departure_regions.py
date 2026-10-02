"""TRAIN-only bounded connected-boundary audit, with no complete route search.

Build all regions from shared observed A* v2 inputs before opening positive
reference paths/types for evaluation. No true obstacle boxes are read.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from scripts import observation_multiroute_astar as grid_api
from scripts import observation_multiroute_astar_v2 as attachment_api
from scripts import observation_prototype_grounding as prototype


PARENTS = {'obstacle_reach_%06d' % number for number in range(272000, 272008)}
ALL_PRIOR_PARENTS = {'obstacle_reach_%06d' % number for number in range(272000, 272032)}
CONFIG = dict(radii_m=[.15, .25, .35], voxel_m=.025, maximum_admitted_nodes=20000,
    scope='first8 predeclared old32 TRAIN parents, all instructions and positive references',
    radius_metric='Euclidean distance from measured current tip; not goal distance or reference arc length',
    boundary='reachable in-radius free nodes with a conservative valid grid edge to an out-of-radius free node',
    adjacency='shared26-neighbor edges; all supercover touched voxels must be free',
    region='connected components of the induced boundary graph, without arbitrary angular splitting',
    node_limit='reject before graph construction if the entire admitted in-radius free-node set exceeds20000',
    reference_mapping='last sqrt(3)*voxel length of the segment before first exact sphere exit; full grid supercover intersects frozen boundary',
    inference_fields=['RGB', 'depth', 'camera calibration', 'current state', 'instruction', 'unchanged TRAIN prototype/workspace'],
    complete_routes_searched=0, true_boxes_read=False)


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def selected_rows(path):
    # Filter explicit parent text before decoding; no other role content is used.
    rows = []
    for line in Path(path).read_text().splitlines():
        if any('"'+parent+'"' in line for parent in PARENTS):
            row = json.loads(line)
            if row['parent_id'] not in PARENTS or row['split'] != 'TRAIN':
                raise ValueError('unexpected parent/role')
            rows.append(row)
    return rows


def departure_regions(free, lower, current_xyz, starts, radius, node_limit=20000):
    began = time.perf_counter()
    voxel = CONFIG['voxel_m']
    shape = np.asarray(free.shape)
    coordinates = np.argwhere(free)
    distance = np.linalg.norm(lower+coordinates*voxel-current_xyz, axis=1)
    nodes = coordinates[distance <= radius+1e-12]
    count = len(nodes)
    info = dict(radius_m=radius, in_radius_free_nodes=count, admitted_nodes=0,
        candidate_neighbor_edges=0, supercover_voxel_checks=0, valid_in_radius_directed_edges=0,
        valid_outward_directed_edges=0, reachable_nodes=0, boundary_nodes=0, regions=None,
        complete_routes_searched=0)
    def finish(status, cells=None, labels=None):
        info.update(status=status, seconds=time.perf_counter()-began)
        return (np.empty((0, 3), np.int64) if cells is None else cells,
                np.empty(0, np.int32) if labels is None else labels, info)
    if count > node_limit:
        return finish('admitted_node_budget_exhausted')
    if not starts:
        return finish('no_original_rule_start_attachment')
    if not count:
        return finish('no_in_radius_free_nodes')
    info['admitted_nodes'] = count
    node_index = np.full(free.shape, -1, dtype=np.int32)
    node_index[tuple(nodes.T)] = np.arange(count)
    seed_ids = [int(node_index[index]) for index in starts if node_index[index] >= 0]
    if not seed_ids:
        return finish('start_attachments_outside_radius')
    rows, cols, outward = [], [], np.zeros(count, bool)
    # This constructs a bounded graph, never predecessor chains or goal paths.
    for delta, touched, _ in grid_api.NEIGHBORS:
        neighbor = nodes+np.asarray(delta)
        good = np.all((neighbor >= 0) & (neighbor < shape), axis=1)
        info['candidate_neighbor_edges'] += count
        for offset in touched:
            location = nodes+np.asarray(offset)
            inside = np.all((location >= 0) & (location < shape), axis=1)
            safe = np.clip(location, 0, shape-1)
            good &= inside & free[tuple(safe.T)]
            info['supercover_voxel_checks'] += count
        source = np.flatnonzero(good)
        destination = node_index[tuple(neighbor[good].T)]
        inside_radius = destination >= 0
        rows.append(source[inside_radius]); cols.append(destination[inside_radius])
        outward[source[~inside_radius]] = True
        info['valid_outward_directed_edges'] += int((~inside_radius).sum())
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), np.uint8), (rows, cols)), shape=(count, count)).tocsr()
    info['valid_in_radius_directed_edges'] = len(rows)
    _, components = connected_components(graph, directed=False)
    reachable = np.isin(components, np.unique(components[seed_ids]))
    boundary = np.flatnonzero(reachable & outward)
    info.update(reachable_nodes=int(reachable.sum()), boundary_nodes=len(boundary))
    if not len(boundary):
        info.update(regions=0, region_sizes=[])
        return finish('no_reachable_outer_boundary')
    number, labels = connected_components(graph[boundary][:, boundary], directed=False)
    labels = labels.astype(np.int32)+1
    info.update(regions=int(number), region_sizes=np.bincount(labels)[1:].tolist())
    return finish('completed', nodes[boundary], labels)


def first_exit(path, center, radius):
    for index, (first, second) in enumerate(zip(path, path[1:])):
        if np.linalg.norm(second-center) <= radius+1e-12:
            continue
        direction, offset = second-first, first-center
        a, b, c = float(direction@direction), float(2*offset@direction), float(offset@offset-radius**2)
        if a <= 1e-20:
            continue
        discriminant = max(0., b*b-4*a*c)
        roots = [r for r in ((-b-np.sqrt(discriminant))/(2*a), (-b+np.sqrt(discriminant))/(2*a)) if -1e-10 <= r <= 1+1e-10]
        if roots:
            fraction = float(np.clip(max(roots), 0., 1.))
            return index, fraction, first+fraction*direction
    return None


def map_reference(path, current, radius, lower, cells, labels, free):
    found = first_exit(path, current, radius)
    if found is None:
        return dict(leaves_radius=False, assigned_regions=[], unambiguous=False)
    segment, fraction, exit_point = found
    direction = path[segment+1]-path[segment]
    length = float(np.linalg.norm(direction))
    before = path[segment]+max(0., fraction-np.sqrt(3)*CONFIG['voxel_m']/max(length, 1e-20))*direction
    touched = grid_api.segment_supercover(before, exit_point, lower, CONFIG['voxel_m'])
    lookup = {tuple(cell): int(label) for cell, label in zip(cells, labels)}
    assigned = sorted({lookup[cell] for cell in touched if cell in lookup})
    prefix = np.vstack([path[:segment+1], exit_point])
    clear, checked = grid_api.path_observed_clear(prefix, free, lower)
    return dict(leaves_radius=True, first_exit_segment=segment, first_exit_xyz=exit_point.tolist(),
        assigned_regions=assigned, unambiguous=len(assigned) == 1, geometric_boundary_covered=bool(assigned),
        prefix_observed_proxy_clear=bool(clear), prefix_grid_voxels_checked=checked,
        covered_and_prefix_proxy_clear=bool(assigned and clear), mapping_supercover_voxels=len(touched))


def run(data, output, prior_report, prior_hash):
    if output.exists():
        raise FileExistsError('fresh output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise RuntimeError('hide CUDA for this CPU-only audit')
    if prototype.digest(prior_report) != prior_hash:
        raise ValueError('TRAIN prior report hash mismatch')
    frozen = json.loads(prior_report.read_text())
    if frozen.get('split') != 'TRAIN' or not frozen.get('subset_training_cost_diagnostic'):
        raise ValueError('reuse the original TRAIN probe report, not a DEV report')
    if frozen['config'] != attachment_api.CONFIG or frozen['prototype_config'] != prototype.CONFIG:
        raise ValueError('shared planner configuration changed')
    if (prototype.digest(grid_api.__file__) != frozen['shared_grid_script_sha256']
            or prototype.digest(attachment_api.__file__) != frozen['script_sha256']):
        raise ValueError('shared observed A* source differs from frozen prior report')
    prior, model = frozen['train_workspace_prior'], frozen['prototype_model']
    if model['training_parents'] != 32 or prior['train_routes'] != 181:
        raise ValueError('old32 TRAIN-fitted quantities required')
    if any(row['id'].rsplit('_target', 1)[0] not in ALL_PRIOR_PARENTS for row in model['training_rows']):
        raise ValueError('prior fitted outside authorized old32 TRAIN')
    observations = sorted(selected_rows(data/'observations.jsonl'), key=lambda row: row['id'])
    pointers = {row['id']: row['observation'] for row in selected_rows(data/'supervision.jsonl')}
    if len(observations) != 24 or {row['parent_id'] for row in observations} != PARENTS:
        raise ValueError('all first8 TRAIN parents/24 instructions required')
    if any(set(row) != prototype.INPUT_KEYS for row in observations):
        raise ValueError('observation whitelist violated')
    output.mkdir(parents=True)
    source_root = Path(__file__).resolve().parents[1]
    provenance = dict(config=CONFIG, code_commit=os.environ.get('CODE_COMMIT'), pid=os.getpid(),
        prior_report=str(prior_report), prior_report_sha256=prior_hash,
        source_sha256={str(Path(module.__file__).relative_to(source_root)): prototype.digest(module.__file__)
                       for module in (grid_api, attachment_api, prototype)}, script_sha256=prototype.digest(__file__),
        manifest_sha256={name: prototype.digest(data/name) for name in ('observations.jsonl', 'supervision.jsonl')},
        parents=sorted(PARENTS), all_regions_written_before_reference_evaluation=True)
    write(output/'config.json', provenance)
    started, records = time.perf_counter(), []
    for row in observations:
        began = time.perf_counter()
        (rgb, xyz, valid), files = prototype.load_observation(data, row, pointers[row['id']])
        with np.load(data/pointers[row['id']], allow_pickle=False) as archive:
            current = np.r_[archive['gripper_pose'], np.asarray(archive['gripper_open']).reshape(1)]
            depth, intrinsics, camera = archive['depth'], archive['camera_intrinsics'], archive['camera_extrinsics']
        endpoint, localization = prototype.predict(rgb, xyz, valid, row['instruction'], model)
        record = dict(id=row['id'], parent_id=row['parent_id'], source_sha256={str(p): prototype.digest(p) for p in files},
            localization=localization, radii=[])
        arrays = dict(current=current, radii=np.asarray(CONFIG['radii_m']))
        if endpoint is None:
            record['status'] = 'localization_failed'
        else:
            free, lower, _, _, grid = grid_api.build_grid(rgb, xyz, valid, depth, intrinsics, camera, current,
                endpoint, row['instruction'], model, localization, prior)
            component = grid_api.selected_target_component(rgb, valid, row['instruction'], model, localization)
            rays = dict(depth=depth, valid=valid, component=component, intrinsics=intrinsics, camera_to_world=camera,
                current=current, endpoint=endpoint, target_radius=grid['contact_allowances']['target_radius_m'])
            starts, audit = attachment_api.attachments(current[:3], 'start', free, lower,
                attachment_api.CONFIG['current_tip_contact_radius_m'], xyz[valid & ~component], rays)
            record.update(status='generated', grid=grid, start_attachments=audit)
            arrays.update(free=free, lower=lower)
            for number, radius in enumerate(CONFIG['radii_m']):
                cells, labels, statistics = departure_regions(free, lower, current[:3], starts, radius)
                arrays['cells%d' % number], arrays['labels%d' % number] = cells, labels
                record['radii'].append(statistics)
        destination = output/(row['id']+'_regions.npz')
        np.savez_compressed(destination, **arrays)
        record.update(artifact_sha256=prototype.digest(destination), generation_seconds=time.perf_counter()-began)
        records.append(record)
        write(output/'generation_progress.json', records)
        print(json.dumps(dict(id=row['id'], status=record['status'], regions=[r['regions'] for r in record['radii']],
            seconds=record['generation_seconds'])), flush=True)
    generation = dict(records=records, seconds=time.perf_counter()-started, complete_routes_searched=0,
        all_24_observation_outputs_saved=True, reference_paths_opened=False, true_boxes_opened=False)
    write(output/'generation_complete.json', generation)
    # Strict phase barrier: only now read positive trajectories and type labels.
    labels = {row['id']: row for row in selected_rows(data/'supervision.jsonl')}
    evaluations = []
    for record in records:
        label = labels[record['id']]
        if len(label['routes']) != len(label['route_types']):
            raise ValueError('positive reference/type mismatch')
        with np.load(output/(record['id']+'_regions.npz'), allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        for reference, (name, route_type) in enumerate(zip(label['routes'], label['route_types'])):
            path = data/name
            if path.parent.name != record['parent_id']:
                raise ValueError('reference escaped selected TRAIN parent')
            with np.load(path, allow_pickle=False) as archive:
                xyz = archive['gripper_pose'][:, :3].copy()
            if not np.isfinite(xyz).all() or np.linalg.norm(xyz[0]-arrays['current'][:3]) > .005:
                raise ValueError('reference must be finite and start at the observed tip')
            row = dict(id=record['id'], parent_id=record['parent_id'], reference=reference,
                path=str(path), path_sha256=prototype.digest(path), known_type=route_type, radii=[])
            for number, radius in enumerate(CONFIG['radii_m']):
                if record['status'] != 'generated':
                    result = dict(generation_failed=True, leaves_radius=bool(first_exit(xyz, arrays['current'][:3], radius)), assigned_regions=[])
                else:
                    result = map_reference(xyz, arrays['current'][:3], radius, arrays['lower'],
                        arrays['cells%d' % number], arrays['labels%d' % number], arrays['free'])
                    result['generation_failed'] = record['radii'][number]['regions'] is None
                row['radii'].append(result)
            evaluations.append(row)
    write(output/'reference_evaluation.json', evaluations)
    summary = summarize(records, evaluations)
    summary.update(total_seconds=time.perf_counter()-started, generation_seconds=generation['seconds'],
        artifact_sha256={p.name: prototype.digest(p) for p in output.glob('*.npz')},
        generation_complete_sha256=prototype.digest(output/'generation_complete.json'),
        reference_evaluation_sha256=prototype.digest(output/'reference_evaluation.json'))
    write(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)


def summarize(records, evaluations):
    by_radius = {}
    for number, radius in enumerate(CONFIG['radii_m']):
        counts = [r['radii'][number]['regions'] if r['status'] == 'generated' else None for r in records]
        eligible = [r for r in evaluations if r['radii'][number]['leaves_radius']]
        covered = [r for r in eligible if len(r['radii'][number]['assigned_regions']) == 1]
        different, separated, collapsed, same, fragmented = 0, 0, 0, 0, 0
        for index, first in enumerate(covered):
            if first['known_type'] is None:
                continue
            for second in covered[index+1:]:
                if first['id'] != second['id'] or second['known_type'] is None:
                    continue
                distinct = first['radii'][number]['assigned_regions'] != second['radii'][number]['assigned_regions']
                if first['known_type'] != second['known_type']:
                    different += 1; separated += int(distinct); collapsed += int(not distinct)
                else:
                    same += 1; fragmented += int(distinct)
        by_radius[str(radius)] = dict(instructions=len(records), region_count_histogram=dict(Counter(str(c) for c in counts)),
            generation_failed=sum(c is None for c in counts), multi_region_instructions=sum(c is not None and c > 1 for c in counts),
            positive_references=len(evaluations), reference_exits=len(eligible), no_radius_exit=len(evaluations)-len(eligible),
            geometric_boundary_covered=sum(bool(r['radii'][number]['assigned_regions']) for r in eligible),
            unambiguous_reference_exits=len(covered),
            covered_and_prefix_proxy_clear=sum(r['radii'][number].get('covered_and_prefix_proxy_clear', False) for r in eligible),
            known_different_type_mapped_pairs=different, separated_different_type_pairs=separated, collapsed_different_type_pairs=collapsed,
            known_same_type_mapped_pairs=same, fragmented_same_type_pairs=fragmented,
            maximum_regions=max((c for c in counts if c is not None), default=None))
    no_branch = all(r['multi_region_instructions'] == 0 for r in by_radius.values())
    failed = any(r['generation_failed'] > 0 for r in by_radius.values())
    decision = ('inconclusive_no_observed_multi_regions_with_generation_failures' if failed else
                'reject_this_boundary_representation_no_multi_region_space') if no_branch else (
                'inspect_coverage_and_fragmentation_before_any_allocator; multiple_components_alone_are_not_evidence')
    return dict(config=CONFIG, parents=len(PARENTS), observations=len(records), radii=by_radius,
        decision=decision,
        limitation='observed partial-space proxy only; local components are not certified global route/homotopy classes; reference sets incomplete; unknown types retained')


def self_test():
    free = np.ones((13, 13, 13), bool)
    lower = np.array([-.15, -.15, -.15]); current = np.zeros(3); starts = {(6, 6, 6): {}}
    cells, labels, info = departure_regions(free, lower, current, starts, .10)
    assert info['regions'] == 1 and info['complete_routes_searched'] == 0
    free[6, :, :] = False; free[6, 5:8, 5:8] = True
    _, _, split = departure_regions(free, lower, current, starts, .10)
    assert split['regions'] == 2
    _, _, capped = departure_regions(np.ones_like(free), lower, current, starts, .10, node_limit=3)
    assert capped['status'] == 'admitted_node_budget_exhausted' and capped['admitted_nodes'] == 0 and capped['regions'] is None
    hit = first_exit(np.array([[0., 0., 0.], [.2, 0., 0.]]), current, .10)
    np.testing.assert_allclose(hit[2], [.1, 0, 0], atol=1e-12)
    mapped = map_reference(np.array([[0., 0., 0.], [.2, 0., 0.]]), current, .10, lower, cells, labels, np.ones_like(free))
    assert mapped['unambiguous'] and mapped['covered_and_prefix_proxy_clear']
    assert first_exit(np.array([[0., 0., 0.], [.02, 0., 0.]]), current, .10) is None
    json.dumps(dict(region=info, split=split, capped=capped, mapped=mapped), allow_nan=False)
    print('departure self-test passed: connected shell, genuine split, finite node cap, exact first exit, preserved no-exit denominator')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path); parser.add_argument('--output', type=Path)
    parser.add_argument('--prior-report', type=Path); parser.add_argument('--prior-report-sha256')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif None in (args.data, args.output, args.prior_report, args.prior_report_sha256):
        parser.error('all data/output/prior provenance arguments required')
    else:
        run(args.data, args.output, args.prior_report, args.prior_report_sha256)


if __name__ == '__main__':
    main()
