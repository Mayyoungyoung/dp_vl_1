"""Fixed first16 TRAIN correspondence opportunity; saved pools only, no forward.

Route types are incomplete positive relations, not a one-to-one correspondence
or an exhaustive mode inventory. Geometry comparisons never modify trajectories.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import os
from pathlib import Path
import re
import time

import numpy as np

PROTOCOL = 'two_row_first16_train_cross_goal_opportunity_v1'
PARENTS = ['two_row_reach_%d' % i for i in range(283200, 283216)]
IDS = {p + '_target%d' % t for p in PARENTS for t in range(3)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_lines(path):
    """Decode only registered TRAIN records; other JSONL payloads stay opaque.

    The entire metadata file has already been hash-checked. Its exporter writes
    id first. Reject noncanonical headers rather than parse another role's labels.
    """
    rows = {}
    with Path(path).open(encoding='utf-8') as stream:
        for line in stream:
            match = re.match(r'^\{"id": "([^"]+)"[,}]', line)
            if not match:
                raise ValueError('Canonical frozen export id-first JSONL required')
            identifier = match.group(1)
            if identifier not in IDS:
                continue
            row = json.loads(line)
            if (identifier in rows or row.get('id') != identifier or row.get('split') != 'TRAIN'
                    or row.get('parent_id') != identifier.rsplit('_target', 1)[0]):
                raise ValueError('Duplicate or non-TRAIN registered identity')
            rows[identifier] = row
    if set(rows) != IDS:
        raise ValueError('All original48 TRAIN inputs required; no replacement/subset')
    return rows


def train_path(name, corpus, parent):
    if parent not in PARENTS: raise ValueError('Unregistered TRAIN parent')
    base = Path(corpus).resolve()/'parents'/'TRAIN'/parent
    if base.resolve() != base: raise ValueError('Symlink changes selected TRAIN parent')
    path = Path(name)
    try: path.resolve().relative_to(base)
    except ValueError: raise ValueError('Raw read outside selected TRAIN parent')
    return path


def arc_sample(xyz, count=33):
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError('Finite xyz polyline required')
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    if arc[-1] <= 0:
        return np.repeat(xyz[:1], count, axis=0)
    keep = np.r_[True, np.diff(arc) > 0]
    return np.stack([np.interp(np.linspace(0, arc[-1], count), arc[keep], xyz[keep, j]) for j in range(3)], -1)


def describe(xyz, row_x, post_top, long_threshold=2.):
    """All plane intersections, deduplicated only at the same shared vertex.

    A touch and a segment lying in the plane remain explicit ambiguity. A
    backward/repeated crossing is retained even if the official type is known.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        return dict(finite=False, simple_forward=False, ambiguity=['nonfinite_or_malformed'], crossings=[])
    lengths = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    arc = np.r_[0., np.cumsum(lengths)]
    events, coplanar = [], []
    for row, x in enumerate(row_x):
        signs = np.sign(xyz[:, 0] - x)
        signs[np.abs(xyz[:, 0] - x) <= 1e-10] = 0
        for i in range(len(xyz) - 1):
            if signs[i] == signs[i+1] == 0:
                coplanar.append(dict(row=row, segment=i))
            elif signs[i] * signs[i+1] < 0:
                t = float((x-xyz[i, 0]) / (xyz[i+1, 0]-xyz[i, 0]))
                point = xyz[i] + t*(xyz[i+1]-xyz[i])
                events.append(dict(row=row, vertex_parameter=i+t, xyz=point.tolist(),
                    direction=int(signs[i+1]), arc_fraction=float((arc[i]+t*lengths[i])/arc[-1]) if arc[-1] else 0.))
        for i in np.flatnonzero(signs == 0):
            if (i > 0 and signs[i-1] == 0) or (i+1 < len(xyz) and signs[i+1] == 0):
                continue  # represented by explicit coplanar interval(s)
            direction = int(signs[i+1]) if 0 < i < len(xyz)-1 and signs[i-1]*signs[i+1] < 0 else 0
            events.append(dict(row=row, vertex_parameter=int(i), xyz=xyz[i].tolist(), direction=direction,
                arc_fraction=float(arc[i]/arc[-1]) if arc[-1] else 0.))
    events.sort(key=lambda e: (e['vertex_parameter'], e['row']))
    counts = [sum(e['row'] == row for e in events) for row in range(2)]
    reasons = []
    if coplanar: reasons.append('coplanar_segment')
    if any(e['direction'] == 0 for e in events): reasons.append('tangent_or_endpoint_touch')
    if any(n == 0 for n in counts): reasons.append('missing_row_crossing')
    if any(n > 1 for n in counts): reasons.append('multiple_row_crossings')
    if [e['row'] for e in events] != [0, 1]: reasons.append('non_single_forward_row_order')
    if any(e['direction'] < 0 for e in events): reasons.append('backward_crossing')
    if not xyz[0, 0] < row_x[0] < row_x[1] < xyz[-1, 0]: reasons.append('endpoints_do_not_straddle_rows')
    simple = not reasons
    sampled = arc_sample(xyz)
    u = np.linspace(0., 1., len(sampled))[:, None]
    residual = sampled - ((1-u)*xyz[0] + u*xyz[-1])
    return dict(finite=True, length_m=float(arc[-1]), long_arc=bool(arc[-1] > long_threshold),
        maximum_z_m=float(xyz[:, 2].max()), simple_forward=simple, ambiguity=reasons,
        crossings=events, crossing_counts=counts, coplanar_segments=coplanar,
        crossing_yz=[[e['xyz'][1], e['xyz'][2]-post_top] for e in events] if simple else None,
        chord_residual=residual.tolist())


def distance(a, b):
    if not (a['simple_forward'] and b['simple_forward']):
        return None
    delta = np.asarray(a['crossing_yz']) - b['crossing_yz']
    return dict(crossing_yz_m=float(np.linalg.norm(delta, axis=1).mean()),
        lateral_m=float(np.abs(delta[:, 0]).mean()), height_m=float(np.abs(delta[:, 1]).mean()),
        chord_residual_m=float(np.linalg.norm(np.asarray(a['chord_residual'])-b['chord_residual'], axis=1).mean()))


def mean_values(values):
    return {key: float(np.mean([v[key] for v in values])) for key in values[0]} if values else None


def balanced_pair(left, right):
    """Equal type-pair weights inside each parent/target pair, not route count."""
    groups = defaultdict(list)
    known_pairs = 0
    for a, b in itertools.product(left, right):
        if a['type'] is None or b['type'] is None:
            continue  # unknown never becomes a negative pair
        known_pairs += 1
        key = (tuple(a['type']), tuple(b['type']))
        value = distance(a['descriptor'], b['descriptor'])
        if value is not None: groups[key].append(value)
    same, different, cells = [], [], []
    for (a, b), values in sorted(groups.items()):
        value = mean_values(values)
        (same if a == b else different).append(value)
        cells.append(dict(left_type=a, right_type=b, route_pairs=len(values), mean=value))
    common = {tuple(x['type']) for x in left if x['type'] is not None} & {tuple(x['type']) for x in right if x['type'] is not None}
    return dict(known_common_types=sorted(common), known_common_count=len(common),
        unknown_routes_left=sum(r['type'] is None for r in left), unknown_routes_right=sum(r['type'] is None for r in right),
        all_route_pairs=len(left)*len(right), known_route_pairs=known_pairs,
        simple_known_route_pairs=sum(len(v) for v in groups.values()),
        same=mean_values(same), different=mean_values(different), type_pair_cells=cells)


def summarize_pairs(pairs, thresholds):
    eligible = [r for r in pairs if r['same'] is not None and r['different'] is not None]
    ratios = [r['same']['crossing_yz_m']/max(r['different']['crossing_yz_m'], 1e-12) for r in eligible]
    wins = [r['same']['crossing_yz_m'] < r['different']['crossing_yz_m'] for r in eligible]
    rich = [r for r in eligible if r['known_common_count'] >= 2]
    gates = dict(enough_pairs=len(eligible) >= thresholds['minimum_eligible_target_pairs'],
        enough_parents=len({r['parent_id'] for r in eligible}) >= thresholds['minimum_eligible_parents'],
        enough_multi_type_pairs=len(rich) >= thresholds['minimum_two_common_type_pairs'],
        median_ratio=bool(ratios) and float(np.median(ratios)) <= thresholds['maximum_median_same_different_ratio'],
        fraction_smaller=bool(wins) and float(np.mean(wins)) >= thresholds['minimum_fraction_same_smaller'])
    # Every one of the48 target pairs is reported, including ineligible ones.
    by_parent = defaultdict(list)
    for row in eligible: by_parent[row['parent_id']].append(row)
    return dict(requested_unordered_target_pairs=48, actual_target_pairs=len(pairs), eligible_pairs=len(eligible),
        eligible_parents=len({r['parent_id'] for r in eligible}), multiple_common_type_pairs=len(rich),
        pair_equal_mean_same=mean_values([r['same'] for r in eligible]),
        pair_equal_mean_different=mean_values([r['different'] for r in eligible]),
        parent_equal_mean_same=mean_values([mean_values([r['same'] for r in rows]) for rows in by_parent.values()]),
        parent_equal_mean_different=mean_values([mean_values([r['different'] for r in rows]) for rows in by_parent.values()]),
        median_same_different_ratio=float(np.median(ratios)) if ratios else None,
        fraction_same_smaller=float(np.mean(wins)) if wins else None, screens=gates,
        reference_structure_screen_passed=all(gates.values()))


def load_pool(path, expected_digest):
    if digest(path) != expected_digest: raise ValueError('Frozen prediction SHA mismatch')
    with np.load(path, allow_pickle=False) as a:
        ids, parents = a['scene_ids'].astype(str), a['parent_ids'].astype(str)
        xyz, events = a['paths'].copy(), a['gripper_open'].copy()
    if (ids.shape != (48,) or parents.shape != (48,) or len(set(ids)) != 48 or set(ids) != IDS
            or xyz.shape != (48, 4, 24, 3) or events.shape != (48, 4, 24)
            or any(p != i.rsplit('_target', 1)[0] for i, p in zip(ids, parents))):
        raise ValueError('Exact48 TRAIN conditions and all192 K4/H24 slots required')
    return {i: (xyz[n], events[n]) for n, i in enumerate(ids)}


def prediction_screen(records, thresholds):
    eligible = [r for r in records if r['semantic_correct'] and r['mean_excess_m'] is not None]
    groups = {}
    for collision in (False, True):
        group = [r for r in eligible if r['collision'] == collision]
        parents = defaultdict(list)
        for r in group: parents[r['id'].rsplit('_target', 1)[0]].append(r['mean_excess_m'])
        groups[str(collision)] = dict(candidates=len(group), parents=len(parents),
            mean_excess_m=float(np.mean([r['mean_excess_m'] for r in group])) if group else None,
            parent_equal_excess_m=float(np.mean([np.mean(v) for v in parents.values()])) if group else None,
            beyond_envelope=sum(r['mean_excess_m'] > 0 for r in group))
    clear, hit = groups['False'], groups['True']
    difference = hit['parent_equal_excess_m']-clear['parent_equal_excess_m'] if hit['candidates'] and clear['candidates'] else None
    screens = dict(enough_colliding=hit['candidates'] >= thresholds['minimum_colliding_candidates'],
        enough_clear=clear['candidates'] >= thresholds['minimum_clear_candidates'],
        enough_colliding_parents=hit['parents'] >= thresholds['minimum_parents_per_group'],
        enough_clear_parents=clear['parents'] >= thresholds['minimum_parents_per_group'],
        absolute_excess=hit['parent_equal_excess_m'] is not None and hit['parent_equal_excess_m'] >= thresholds['minimum_collision_excess_m'],
        association=difference is not None and difference >= thresholds['minimum_collision_minus_clear_m'],
        fraction_excess=bool(hit['candidates']) and hit['beyond_envelope']/hit['candidates'] >= thresholds['minimum_collision_fraction_excess'])
    return dict(all_candidate_slots=len(records), semantic_correct_slots=sum(r['semantic_correct'] for r in records),
        eligible_semantic_correct_slots=len(eligible), unknown_type_slots=sum(not r['known_type'] for r in records),
        simple_forward_slots=sum(r['simple_forward'] for r in records), collision_groups=groups,
        parent_mean_collision_minus_clear_m=difference, screens=screens, prediction_gap_screen_passed=all(screens.values()))


def audit(config_path, output):
    # Imports are lazy: pure local tests never require Torch or a simulator.
    from scripts import export_two_row_observations as exporter
    from scripts.evaluate_observed_two_row import scene_metrics
    from scripts.collect_observed_two_row_pilot import crossing_signature
    from routeset.observed_route_head import resample_event_segments
    import torch
    torch.set_num_threads(1)
    config_path, output = Path(config_path), Path(output)
    if output.exists(): raise FileExistsError('Fresh opportunity output required')
    config = json.loads(config_path.read_text()); started = time.perf_counter(); hashes = {}
    if config['protocol'] != PROTOCOL or config['train_parent_ids'] != PARENTS:
        raise ValueError('Fixed first16 registration required')
    def checked(path, expected):
        path = Path(path); value = digest(path)
        if value != expected: raise ValueError('Registered source SHA mismatch: '+str(path))
        hashes[str(path)] = value
        return path
    for name, value in config['frozen_artifacts_sha256'].items(): checked(name, value)
    data = Path(config['data']); run = Path(config['run']); last = Path(config['last_train_audit'])
    manifest = json.loads((data/'export_manifest.json').read_text())
    if manifest['protocol'] != exporter.PROTOCOL or exporter.selection_guard(manifest['selection']) != 16:
        raise ValueError('Original prefix28 export required')
    for name, value in manifest['output_files_sha256'].items(): checked(data/name, value)
    # Whole metadata bytes are checked; no DEV raw path is hashed/opened.
    corpus = Path(manifest['source_dataset']); exporter.collector.verify_corpus(corpus)
    gate = exporter.collector.live_layout_gate(corpus)
    if set(PARENTS) & set(gate['blocked_parent_ids']): raise ValueError('TRAIN mechanical layout gate blocked')
    observations, labels = selected_lines(data/'observations.jsonl'), selected_lines(data/'supervision.jsonl')
    summary = json.loads((run/'summary.json').read_text()); receipt = json.loads((run/'two_row_driver_receipt.json').read_text())
    last_report = json.loads((last/'report.json').read_text())
    best_sha = config['frozen_artifacts_sha256'][str(run/'train/predictions.npz')]
    last_sha = config['frozen_artifacts_sha256'][str(last/'last_train/predictions.npz')]
    if (receipt['actual_training_summary_sha256'] != digest(run/'summary.json')
            or receipt['export_sha256'] != digest(data/'export_manifest.json')
            or receipt['prediction_artifacts']['train/predictions.npz']['sha256'] != best_sha
            or summary['best_step'] != 500 or summary['last_step'] != 1500
            or last_report['export_manifest_sha256'] != digest(data/'export_manifest.json')
            or last_report['checkpoint_sha256'] != summary['last_checkpoint_sha256']
            or last_report['stages']['last_train']['prediction_sha256'] != last_sha):
        raise ValueError('Original best500/last1500 TRAIN lineage required')
    pools = {'best': load_pool(run/'train/predictions.npz', best_sha),
             'last': load_pool(last/'last_train/predictions.npz', last_sha)}
    def train_file(name, parent):
        path = train_path(name, corpus, parent)
        return checked(path, manifest['source_files_sha256'][str(path)])
    refs = {'raw': {}, 'model_H24': {}}; source_rows = []; candidates = {'best': [], 'last': []}; parent_scales = {}
    for identifier in sorted(IDS):
        label = labels[identifier]; parent = label['parent_id']
        if len(label['routes']) != len(label['route_types']): raise ValueError('Reference/type alignment changed')
        cfg = json.loads(train_file(label['route_config'], parent).read_text())
        if digest(label['route_config']) != label['route_config_sha256']: raise ValueError('Route config changed')
        with np.load(train_file(label['observation'], parent), allow_pickle=False) as a:
            current = {k: a[k].copy() for k in ('gripper_pose', 'gripper_open')}
        with np.load(train_file(label['verification_only'], parent), allow_pickle=False) as a:
            geometry = {k: a[k].copy() for k in ('obstacle_centers', 'obstacle_halfsizes')}
        # Hash RGB without decoding: establish same-image semantics within parent.
        train_file(observations[identifier]['image'], parent)
        top = cfg['post_base_z'] + cfg['post_size_xyz'][2]
        parent_scales[parent] = dict(tip_clearance_m=.02,
            middle_opening_width_after_clearance_m=[float(y[1]-y[0]-cfg['post_size_xyz'][1]-.04) for y in cfg['post_y']])
        for version in refs: refs[version][identifier] = []
        for number, (name, mode) in enumerate(zip(label['routes'], label['route_types'])):
            with np.load(train_file(name, parent), allow_pickle=False) as a:
                pose, opened = a['gripper_pose'].copy(), a['gripper_open'].copy()
            h24, event24 = resample_event_segments(pose, opened, 24)
            actual = crossing_signature(pose[:, :3], cfg)
            if (None if actual is None else list(actual)) != mode: raise ValueError('Stored positive type differs from original classifier')
            versions = [('raw', pose[:, :3], opened), ('model_H24', h24, event24)]
            for version, xyz, events in versions:
                metric, outcome = scene_metrics(xyz[None], events[None], current, geometry, label['semantic_targets'], label['route_types'], cfg)
                descriptor = describe(xyz, cfg['row_x'], top, config['long_arc_threshold_m'])
                actual_type = outcome[0]['declared_passage_type']
                actual_type = list(actual_type) if actual_type is not None else None
                row = dict(reference=number, source=str(name), type=actual_type, original_raw_type=mode,
                    type_changed_by_representation=actual_type != mode, descriptor=descriptor)
                refs[version][identifier].append(row)
                source_rows.append(dict(id=identifier, parent_id=parent, representation=version,
                    tip_valid=metric['TipValidAtK'] == 1, **row))
        for stage, pool in pools.items():
            xyz, opened = pool[identifier]
            _, outcomes = scene_metrics(xyz, opened, current, geometry, label['semantic_targets'], label['route_types'], cfg)
            for k, (path, outcome) in enumerate(zip(xyz, outcomes)):
                descriptor = describe(path, cfg['row_x'], top, config['long_arc_threshold_m'])
                candidates[stage].append(dict(id=identifier, parent_id=parent, candidate=k,
                    descriptor=descriptor, outcome=outcome))
    for parent in PARENTS:
        ids = [parent+'_target%d' % t for t in range(3)]
        if len({digest(observations[i]['image']) for i in ids}) != 1 or len({labels[i]['observation'] for i in ids}) != 1:
            raise ValueError('Same-parent three-goal observation/current identity required')
    if sum(len(v) for v in refs['raw'].values()) != 285:
        raise ValueError('All285 original positive references required')
    for stage, expected in [('best', summary['train_metrics']), ('last', last_report['metrics'])]:
        for key, field in [('TipValidAtK','TipValid'), ('semantic_goal_accuracy','semantic_goal_correct'),
                           ('TipClearAtK','tip_segments_clear')]:
            actual = float(np.mean([r['outcome'][field] for r in candidates[stage]]))
            if not np.isclose(actual, expected[key], rtol=0., atol=1e-12):
                raise ValueError('Original TRAIN prediction metrics changed: '+stage+'/'+key)
    pairs, summaries = {}, {}
    for version in refs:
        pairs[version] = [dict(parent_id=p, left_id=a, right_id=b, **balanced_pair(refs[version][a], refs[version][b]))
            for p in PARENTS for a, b in itertools.combinations([p+'_target%d' % t for t in range(3)], 2)]
        for row in pairs[version]:
            row['physical_scales'] = parent_scales[row['parent_id']]
            row['same_mean_offset_over_2cm'] = row['same']['crossing_yz_m']/.02 if row['same'] else None
            width = min(parent_scales[row['parent_id']]['middle_opening_width_after_clearance_m'])
            row['same_mean_offset_over_min_middle_opening'] = row['same']['crossing_yz_m']/width if row['same'] and width > 0 else None
        summaries[version] = summarize_pairs(pairs[version], config['opportunity_screen'])
        descriptors = [r['descriptor'] for rows in refs[version].values() for r in rows]
        summaries[version]['all_positive_reference_counts'] = dict(total=len(descriptors),
            simple_forward=sum(r['simple_forward'] for r in descriptors), long_arc=sum(r.get('long_arc',False) for r in descriptors),
            simple_forward_long_arc=sum(r['simple_forward'] and r.get('long_arc',False) for r in descriptors),
            ambiguous=sum(not r['simple_forward'] for r in descriptors),
            ambiguity_reasons=dict(Counter(reason for r in descriptors for reason in r['ambiguity'])))
    # Descriptive association: same-type cross-goal reference envelope only.
    # No query-slot correspondence; candidate index is provenance, never matching.
    associations = {}
    for stage, rows in candidates.items():
        records = []
        for candidate in rows:
            source = candidate['id']; mode = candidate['outcome']['declared_passage_type']; comparisons = []
            if mode is not None:
                own = [r for r in refs['model_H24'][source] if r['type'] == list(mode)]
                for target in [candidate['parent_id']+'_target%d' % t for t in range(3) if source != candidate['parent_id']+'_target%d' % t]:
                    other = [r for r in refs['model_H24'][target] if r['type'] == list(mode)]
                    baseline = [v['crossing_yz_m'] for a, b in itertools.product(own, other) if (v := distance(a['descriptor'], b['descriptor'])) is not None]
                    observed = [v['crossing_yz_m'] for b in other if (v := distance(candidate['descriptor'], b['descriptor'])) is not None]
                    if baseline and observed:
                        comparisons.append(dict(other_id=target, reference_p95_m=float(np.percentile(baseline, 95)),
                            prediction_mean_m=float(np.mean(observed)), excess_m=float(np.mean(observed)-np.percentile(baseline, 95))))
            record = dict(id=source, candidate=candidate['candidate'], semantic_correct=candidate['outcome']['semantic_goal_correct'],
                collision=not candidate['outcome']['tip_segments_clear'], known_type=mode is not None,
                simple_forward=candidate['descriptor']['simple_forward'], comparisons=comparisons,
                mean_excess_m=float(np.mean([r['excess_m'] for r in comparisons])) if comparisons else None)
            records.append(record)
        associations[stage] = dict(**prediction_screen(records, config['prediction_screen']), records=records,
            inference='Descriptive, candidate-correlated association; unknown/ambiguous/no-common-reference remain unevaluable, never negative.')
    final_gate = exporter.collector.live_layout_gate(corpus)
    if set(PARENTS) & set(final_gate['blocked_parent_ids']): raise ValueError('Final TRAIN gate blocked')
    if any(digest(p) != h for p, h in hashes.items()): raise ValueError('Frozen inputs changed during audit')
    result = dict(protocol=PROTOCOL, requested_parents=16, requested_inputs=48, unordered_target_pairs=48,
        directed_target_pairs=96, positive_references=sum(len(v) for v in refs['raw'].values()),
        reference_summary=summaries, reference_pairs=pairs, references=source_rows,
        prediction_association=associations, prediction_candidates=candidates,
        joint_opportunity_screen_passed=all(v['reference_structure_screen_passed'] for v in summaries.values()) and associations['last']['prediction_gap_screen_passed'],
        source_files_sha256=hashes, config_sha256=digest(config_path), script_sha256=digest(__file__),
        audit_dependency_sha256={name: digest(Path(__file__).resolve().parents[1]/name) for name in (
            'scripts/export_two_row_observations.py', 'scripts/collect_two_row_formal.py',
            'scripts/evaluate_observed_two_row.py', 'scripts/evaluate_observed_obstacles.py',
            'scripts/collect_observed_two_row_pilot.py', 'routeset/observed_route_head.py')},
        initial_gate=gate, final_gate=final_gate, elapsed_seconds=time.perf_counter()-started,
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        torch_threads=torch.get_num_threads(), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), gpu_hours=0.,
        new_forward_requests=0, new_candidate_states=0, repaired_paths=0, dev_raw_opened=False, locked_raw_opened=False,
        selection_note='Opportunity screen only. No inferential statistics, transport guarantee, new method efficacy or authorization to train.')
    output.mkdir(parents=True)
    exporter.write_json(output/'report.json', result)
    exporter.write_json(output/'artifact_index.json', {'report.json': digest(output/'report.json')})
    print(json.dumps(dict(reference_summary=summaries, positive_references=result['positive_references'])))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); audit(args.config, args.output)
