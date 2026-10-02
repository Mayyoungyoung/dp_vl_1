"""Reproduce format-only counts from an immutable complete independent4 journal.

No text repairs, partial route scoring, observation/geometry/label/model access.
The enclosing two-method generation run is still partial at this snapshot.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import statistics

EXPECTED_SHA = '66fe9c78d4326708134e58ad1ccaa24a7e2cdd29634fb59a7febe61dbf000b43'
ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(values):
    return dict(count=len(values), min=min(values), median=statistics.median(values),
                mean=statistics.mean(values), max=max(values)) if values else dict(count=0)


def main():
    source = ROOT/'requests.jsonl'
    if digest(source) != EXPECTED_SHA:
        raise ValueError('Immutable journal differs from recorded server/local SHA')
    rows = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    if len(rows) != 96 or len({(r['scene_id'], r['request']) for r in rows}) != 96:
        raise ValueError('Complete independent4 24-scene journal required')
    scene_calls = Counter(r['scene_id'] for r in rows)
    if len(scene_calls) != 24 or set(scene_calls.values()) != {4}:
        raise ValueError('Every scene must have all four original calls')
    if any(r['method'] != 'independent4' or r['repeat'] != 0 or r['k'] != 1 or r['max_new_tokens'] != 512 for r in rows):
        raise ValueError('Unexpected generation configuration')
    counters, horizons, point_lengths, scalar_types, event_values = Counter(), Counter(), Counter(), Counter(), Counter()
    eos, errors, tokens_by_class, detail = Counter(), Counter(), defaultdict(list), []
    for row in rows:
        counters['requests'] += 1
        counters['recorded_failure'] += row['failure'] is not None
        counters['recorded_overgeneration'] += row['parse']['budget_exceeded']
        counters['recorded_strict_format_slots'] += row['parse']['format_valid_candidates']
        cap = row['tokens']['reached_token_limit']
        counters['reached_token_limit'] += cap
        eos[str(row['tokens']['last_output_token_id'])] += 1
        entry = {key: row[key] for key in ('scene_id', 'parent_id', 'request', 'decoding_seed')}
        entry.update(raw_text_sha256=hashlib.sha256(row['text'].encode()).hexdigest(),
                     output_tokens=row['tokens']['output_tokens'], reached_token_limit=cap,
                     last_output_token_id=row['tokens']['last_output_token_id'], parse_errors=row['parse']['errors'])
        try:
            parsed = json.loads(row['text'])  # unchanged complete text only
        except (ValueError, RecursionError) as error:
            counters['json_invalid'] += 1
            category = 'json_invalid_at_cap' if cap else 'json_invalid_not_at_cap'
            errors[getattr(error, 'msg', str(error))] += 1
            entry.update(category=category, json_error=str(error), json_error_position=getattr(error, 'pos', None))
        else:
            counters['json_valid'] += 1
            if not isinstance(parsed, list) or len(parsed) != 1 or not isinstance(parsed[0], list):
                raise ValueError('Actual journal changed: expected every valid JSON to contain one route')
            counters['valid_json_exactly_one_route'] += 1
            route = parsed[0]; length = len(route); horizons[str(length)] += 1
            bad_points, noninteger_points, range_bad, event_bad = 0, 0, 0, 0
            for point in route:
                counters['points_in_valid_json'] += 1
                if not isinstance(point, list):
                    bad_points += 1; continue
                point_lengths[str(len(point))] += 1
                scalar_types.update(type(value).__name__ for value in point)
                if len(point) != 4:
                    bad_points += 1; continue
                if any(type(value) is not int for value in point):
                    noninteger_points += 1; continue
                range_bad += any(abs(value) > 10000 for value in point[:3])
                event_bad += point[3] not in (0, 1)
                event_values[str(point[3])] += 1
            counters['bad_point_structure'] += bad_points
            counters['noninteger_points'] += noninteger_points
            counters['protocol_coordinate_range_bad_points'] += range_bad
            counters['protocol_event_range_bad_points'] += event_bad
            category = 'exact_horizon' if length == 24 else ('short_horizon' if length < 24 else 'long_horizon')
            entry.update(category=category, route_count=1, horizon=length, bad_point_structure=bad_points,
                         noninteger_points=noninteger_points, protocol_coordinate_range_bad_points=range_bad,
                         protocol_event_range_bad_points=event_bad)
        counters[category] += 1
        tokens_by_class[category].append(row['tokens']['output_tokens'])
        detail.append(entry)
    generation = json.loads((ROOT/'generation_summary.json').read_text())
    if generation['strict_finite_slots'] != counters['recorded_strict_format_slots']:
        raise ValueError('Copied completed method summary disagrees with journal')
    result = dict(protocol='vlm_independent_format_partial_snapshot_v1',
        scope='Complete 96-request independent4 journal only; enclosing same-checkpoint whole4 comparison was still running. No geometric scoring or result substitution.',
        source_commit='8356b09b8436d00a4f98a76ffdd8c7d84033e6ca',
        source_server='/home/wzy/dpvlm/route_set_v1/runs/vlm_route_sft_autoregressive_v1/best_seed0/repeat0/independent4/requests.jsonl',
        requests_sha256=EXPECTED_SHA, server_sha_before_and_after_copy_agree=True,
        copied_generation_summary_sha256=digest(ROOT/'generation_summary.json'), analyzer_sha256=digest(Path(__file__)),
        methods_analyzed=['independent4'], sampling_repeats=1, parents=8, scenes=24, counters=dict(counters),
        horizon_distribution_valid_json=dict(sorted(horizons.items(), key=lambda kv: int(kv[0]))),
        point_length_distribution_valid_json=dict(point_lengths), scalar_types_valid_json=dict(scalar_types),
        event_value_distribution_valid_json=dict(event_values), final_token_ids=dict(eos), json_error_classes=dict(errors),
        output_tokens_all=describe([r['tokens']['output_tokens'] for r in rows]),
        output_tokens_by_class={key: describe(value) for key, value in sorted(tokens_by_class.items())},
        median_four_request_scene_seconds=generation['generation_ms_median']/1000,
        input_or_geometry_files_opened=0, repaired_texts=0, model_calls=0,
        interpretation='Most recorded format failures are fully parsed one-route integer sequences of the wrong length. Exact length/grammar constraints are a standard prospective decoding control, not a posthoc path repair or method contribution. Format success is not geometric/semantic validity.',
        per_request=detail)
    (ROOT/'diagnostic.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in ('counters', 'horizon_distribution_valid_json', 'output_tokens_all', 'json_error_classes')}, indent=2))


if __name__ == '__main__':
    main()
