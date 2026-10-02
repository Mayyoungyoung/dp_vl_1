"""Audit sealed TRAIN spatial/edge A* pools; never import or call a planner.

The analyzer consumes saved checker decisions and arrays only. It does not
reopen collection labels, reinterpret unknown types, or certify map coverage.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np


PROTOCOL = 'observed_spatial_penalty_saved_train_pair_v1'
GENERATION_PROTOCOL = 'observed_two_row_spatial_penalty_v1'
METRICS = ('TipValidAtK', 'AnyTipValidAtK', 'UniqueClassifiedTipValidAtK',
           'UnknownTypeTipValidCount', 'DuplicateClassifiedTipValidCount',
           'KnownReferenceTypeCoverageAtK', 'semantic_goal_accuracy',
           'AnySemanticGoalAtK', 'TipClearAtK', 'StartCorrectAtK',
           'EventSequenceCorrectAtK', 'endpoint_error_m', 'mean_path_length_m')
CHECKS = ('finite_xyz', 'finite_event_values', 'semantic_goal_correct',
          'starts_at_current_state', 'tip_segments_clear', 'event_state_sequence_correct')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal_number(first, second, label):
    require((first is None and second is None) or
            (first is not None and second is not None and
             np.isfinite(first) and np.isfinite(second) and
             np.isclose(first, second, atol=1e-12, rtol=1e-12)), label)


def verify_artifacts(report, run, binary_root):
    """Map original POSIX paths to copied trees without touching server data."""
    files = report['artifact_sha256']
    fitted = [PurePosixPath(p) for p in files if PurePosixPath(p).name == 'fitted_train_model.json']
    require(len(fitted) == 1, 'Exactly one sealed fitted model required')
    original = fitted[0].parent
    verified = {}
    for path, digest in files.items():
        remote = PurePosixPath(path)
        require(remote.is_absolute() and '..' not in remote.parts, 'Invalid original POSIX artifact path')
        try:
            relative = remote.relative_to(original)
        except ValueError as error:
            raise ValueError('Artifact outside sealed run') from error
        local = (binary_root if relative.suffix == '.npz' else run).joinpath(*relative.parts)
        require(local.is_file() and sha(local) == digest, 'Sealed artifact hash differs: '+str(relative))
        verified[relative.as_posix()] = digest
    require('fitted_train_model.json' in verified, 'Missing sealed fit')
    return verified


def summarize_candidates(candidates, paths, events, raw, attempts):
    require(len(candidates) == len(attempts) == len(raw) == 4 and
            paths.shape == (4, 24, 3) and events.shape == (4, 24), 'Exactly four complete slots required')
    modes, outcomes, searches, slots = Counter(), Counter(), Counter(), []
    valid = unknown = complete = 0
    for k, (candidate, attempt, trajectory) in enumerate(zip(candidates, attempts, raw)):
        require(candidate['candidate'] == attempt['slot'] == k, 'Candidate order changed')
        finite, event_finite = bool(np.isfinite(paths[k]).all()), bool(np.isfinite(events[k]).all())
        require(finite == candidate['finite_xyz'] and event_finite == candidate['finite_event_values'],
                'Saved finite flags differ from arrays')
        require(candidate['TipValid'] == all(candidate[key] for key in CHECKS), 'TipValid flags disagree')
        mode = candidate['declared_passage_type']
        require(mode is None or (len(mode) == 2 and all(x in ('negative_y', 'middle', 'positive_y', 'over') for x in mode)),
                'Unknown type must remain null; invalid known type')
        require(candidate['classified_tip_valid'] == bool(candidate['TipValid'] and mode is not None),
                'Classified validity flags disagree')
        if len(trajectory):
            require(trajectory.ndim == 2 and trajectory.shape[1] == 3 and len(trajectory) >= 2 and
                    np.isfinite(trajectory).all() and finite and attempt['complete_raw_paths_emitted'] == 1,
                    'Returned raw path and completed slot disagree')
            complete += 1
        else:
            require(trajectory.shape == (0, 3) and np.isnan(paths[k]).all() and
                    attempt['complete_raw_paths_emitted'] == 0, 'Missing path must keep its NaN slot')
        if candidate['TipValid']:
            valid += 1
            outcome = 'tip_valid'
            if mode is None:
                unknown += 1
            else:
                modes[tuple(mode)] += 1
        elif not finite:
            outcome = 'no_complete_path:'+attempt['status']
        else:
            outcome = 'finite_invalid:'+','.join(key for key in CHECKS if not candidate[key])
        outcomes[outcome] += 1
        searches[attempt['status']] += 1
        slots.append(dict(slot=k, outcome=outcome, search_status=attempt['status'], candidate=candidate,
                          exact_grid_duplicate=attempt.get('exact_grid_path_duplicate', False),
                          spatial_penalty=attempt.get('spatial_penalty')))
    return dict(valid=valid, known_unique=len(modes), known_duplicate=sum(modes.values())-len(modes),
                unknown_valid=unknown, raw_complete=complete, failed_generation=4-complete,
                finite_invalid=complete-valid, outcomes=dict(outcomes), search_statuses=dict(searches),
                classified_valid_types=[dict(type=list(k), count=v) for k, v in sorted(modes.items())], slots=slots)


def analyze_arm(arm, records, report, run, binary_root, verified):
    ids = report['input_ids']
    by_id = {row['id']: row for row in records}
    require(len(records) == len(by_id) == len(ids) and set(by_id) == set(ids), 'Missing, extra, or repeated request')
    combined_name = arm+'_predictions.npz'
    require(combined_name in verified, 'Unsealed combined pool')
    with np.load(binary_root/combined_name, allow_pickle=False) as pool:
        pool_ids, parents = list(pool['scene_ids'].astype(str)), list(pool['parent_ids'].astype(str))
        paths, events = pool['paths'].copy(), pool['gripper_open'].copy()
    require(len(pool_ids) == len(set(pool_ids)) == 12 and set(pool_ids) == set(ids) and
            paths.shape == (12, 4, 24, 3) and events.shape == (12, 4, 24), 'Full 12-request pool required')
    cases = []
    for identifier in ids:
        record, i = by_id[identifier], pool_ids.index(identifier)
        parent = identifier.rsplit('_target', 1)[0]
        require(record['split'] == 'TRAIN' and record['arm'] == arm and
                record['parent_id'] == parents[i] == parent, 'Role, arm or parent changed')
        q, b = run/arm/identifier, binary_root/arm/identifier
        for name in ('result.json', 'status.json', 'generation_seal.json', 'predictions.npz', 'raw_paths.npz'):
            require('/'.join((arm, identifier, name)) in verified, 'Request artifact not sealed')
        require(read(q/'result.json') == record and read(q/'status.json')['status'] == 'completed',
                'Report row differs from completed saved request')
        seal = read(q/'generation_seal.json')
        require(seal['id'] == identifier and seal['split'] == 'TRAIN' and seal['arm'] == arm and
                seal['evaluation_labels_opened'] is False and seal['submitted_candidate_budget'] == 4,
                'Prediction-before-label seal changed')
        for field, name in [('prediction_sha256', 'predictions.npz'), ('raw_paths_sha256', 'raw_paths.npz')]:
            require(record[field] == seal[field] == sha(b/name), 'Request pool hash differs')
        with np.load(b/'predictions.npz', allow_pickle=False) as one:
            require(one['paths'].shape == (1, 4, 24, 3) and one['gripper_open'].shape == (1, 4, 24) and
                    one['scene_ids'].tolist() == [identifier] and one['parent_ids'].tolist() == [parent],
                    'Per-request pool identity or shape differs')
            require(np.array_equal(paths[i], one['paths'][0], equal_nan=True) and
                    np.array_equal(events[i], one['gripper_open'][0], equal_nan=True), 'Combined pool differs')
        with np.load(b/'raw_paths.npz', allow_pickle=False) as raw:
            require(set(raw.files) == {'candidate_%d'%k for k in range(4)}, 'Raw pool slots differ')
            raw_paths = [raw['candidate_%d'%k].copy() for k in range(4)]
        generation = record['generation']
        require(generation['submitted_candidate_budget'] == 4, 'Candidate budget changed')
        result = summarize_candidates(record['candidates'], paths[i], events[i], raw_paths, generation['attempts'])
        require(result['raw_complete'] == generation['raw_complete_paths'] == seal['raw_complete_paths'] and
                result['failed_generation'] == generation['failed_slots'], 'Failure denominator differs')
        for key, value in [('TipValidAtK', result['valid']/4), ('AnyTipValidAtK', float(result['valid'] > 0)),
                           ('UniqueClassifiedTipValidAtK', result['known_unique']),
                           ('DuplicateClassifiedTipValidCount', result['known_duplicate']),
                           ('UnknownTypeTipValidCount', result['unknown_valid'])]:
            equal_number(record['metrics'][key], value, key+' differs from all slots')
        for key, flag in [('semantic_goal_accuracy', 'semantic_goal_correct'), ('TipClearAtK', 'tip_segments_clear'),
                          ('StartCorrectAtK', 'starts_at_current_state'), ('EventSequenceCorrectAtK', 'event_state_sequence_correct')]:
            equal_number(record['metrics'][key], np.mean([c[flag] for c in record['candidates']]), key+' flags differ')
        field_seconds = sum(a.get('spatial_penalty', {}).get('preprocessing_seconds', 0.) for a in generation['attempts'])
        if arm == 'spatial':
            equal_number(field_seconds, generation['spatial_adapter']['preprocessing_seconds'], 'Field time differs')
            prior_complete = calls = 0
            for attempt in generation['attempts']:
                penalty = attempt.get('spatial_penalty')
                if penalty is not None:
                    require(penalty['previous_returned_raw_paths'] == prior_complete and penalty['sigma_m'] == .05 and
                            penalty['all_returned_paths_included_without_postcheck_filter'] is True and
                            penalty['preprocessing_inside_original_search_deadline'] is False, 'Prior path field audit differs')
                    calls += 1
                prior_complete += attempt['complete_raw_paths_emitted']
            require(calls == generation['spatial_adapter']['actual_astar_calls'] <= 4 and
                    prior_complete == generation['spatial_adapter']['previous_complete_raw_paths'], 'Actual calls differ')
        timing = record['timing']
        require(0 <= field_seconds <= timing['localization_grid_field_search_and_original_proxy_seconds'] <=
                timing['observation_to_checked_pool_seconds'], 'Field costs excluded or negative')
        cases.append(dict(id=identifier, parent_id=parent, metrics=record['metrics'], **result,
                          request_seconds=timing['observation_to_checked_pool_seconds'], field_seconds=field_seconds))
    for key in METRICS:
        values = [case['metrics'][key] for case in cases if case['metrics'][key] is not None]
        equal_number(report['results'][arm][key], np.mean(values) if values else None, 'Aggregate '+key+' differs')
    totals = {key: sum(c[key] for c in cases) for key in ('valid', 'known_unique', 'known_duplicate', 'unknown_valid',
                                                        'raw_complete', 'failed_generation', 'finite_invalid')}
    outcomes = Counter()
    for c in cases:
        outcomes.update(c['outcomes'])
    return dict(cases=cases, totals=totals, outcomes=dict(outcomes), saved_metrics=report['results'][arm],
                field_seconds=sum(c['field_seconds'] for c in cases), request_seconds=sum(c['request_seconds'] for c in cases))


def paired_cases(edge, spatial):
    left, right = ({c['id']: c for c in arm['cases']} for arm in (edge, spatial))
    require(left.keys() == right.keys(), 'Pairing by observation ID required')
    return [dict(id=i, parent_id=left[i]['parent_id'],
                 delta={key: right[i][key]-left[i][key] for key in
                        ('valid', 'known_unique', 'known_duplicate', 'unknown_valid', 'failed_generation', 'request_seconds', 'field_seconds')},
                 edge=left[i], spatial=right[i]) for i in sorted(left)]


def analyze(run, binary_root):
    run, binary_root = Path(run), Path(binary_root)
    report = read(run/'report.json')
    expected = sorted('two_row_reach_%d_target%d'%(p, t) for p in range(283200, 283204) for t in range(3))
    require(report['protocol'] == GENERATION_PROTOCOL and report['status'] == 'completed' and
            report['stage'] == 'train_preflight' and report['mechanical_preflight_passed'] is True and
            sorted(report['input_ids']) == expected and len(report['input_ids']) == 12,
            'Only completed fixed first4 TRAIN /12-input preflight supported')
    require(set(report['per_request']) == set(report['results']) == {'edge', 'spatial'} and
            report['requested_inputs_per_arm'] == 12 and report['requested_candidate_slots'] == 96 and
            report['completed_requests'] == 24 and report['failed_requests'] == report['unattempted_requests'] == 0,
            'Full paired request/slot accounting required')
    require(report['new_qwen_calls'] == report['training_steps'] == report['gpu_hours'] == 0,
            'Unexpected training or model budget')
    verified = verify_artifacts(report, run, binary_root)
    fit = read(run/'fitted_train_model.json')
    require(fit['canonical_sha256'] == report['shared_fit_sha256'] == report['config']['original_shared_fit_sha256'],
            'Shared fit identity differs')
    arms = {arm: analyze_arm(arm, report['per_request'][arm], report, run, binary_root, verified) for arm in ('edge', 'spatial')}
    for name in ('first_h24_equal', 'first_raw_equal'):
        rows = report[name]
        require(len(rows) == 12 and sorted(r['id'] for r in rows) == expected and all(r['equal'] for r in rows),
                'Complete first-path equality receipt required')
    for i in expected:
        for file, key in [('predictions.npz', 'paths'), ('raw_paths.npz', 'candidate_0')]:
            with np.load(binary_root/'edge'/i/file, allow_pickle=False) as a, np.load(binary_root/'spatial'/i/file, allow_pickle=False) as b:
                first, second = (a[key][0, 0], b[key][0, 0]) if key == 'paths' else (a[key], b[key])
                require(np.array_equal(first, second, equal_nan=True), 'Actual first-path arrays differ')
    pairs = paired_cases(arms['edge'], arms['spatial'])
    return dict(protocol=PROTOCOL, source_report_sha256=sha(run/'report.json'),
                source_files_sha256=report['source_files_sha256'], verified_artifacts=verified,
                requested_conditions=12, paired_candidate_slots=96, arms=arms, all_pairs=pairs,
                per_parent=[dict(parent_id=p, delta={key: sum(row['delta'][key] for row in pairs if row['parent_id'] == p)
                              for key in pairs[0]['delta']}) for p in sorted({r['parent_id'] for r in pairs})],
                known_unique_delta=arms['spatial']['totals']['known_unique']-arms['edge']['totals']['known_unique'],
                known_reference_coverage_delta=report['results']['spatial']['KnownReferenceTypeCoverageAtK']-
                    report['results']['edge']['KnownReferenceTypeCoverageAtK'] if
                    all(report['results'][a]['KnownReferenceTypeCoverageAtK'] is not None for a in arms) else None,
                shared_fit_seconds=report['train_fit']['elapsed_seconds'], pipeline_seconds=report['elapsed_seconds'],
                analysis_source_sha256=sha(__file__), new_planner_calls=0, new_model_calls=0, labels_reopened=False,
                interpretation='Saved TRAIN association only. If no gain, admissibility of other known positive routes in the original observed map remains untested. No retuning or automatic DEV launch.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--binary-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.run, args.binary_root)
    require(not args.output.exists(), 'Fresh derived output required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('requested_conditions', 'paired_candidate_slots', 'known_unique_delta')}))
