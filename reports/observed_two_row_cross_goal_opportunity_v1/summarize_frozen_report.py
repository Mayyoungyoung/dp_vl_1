"""Summarize the sealed TRAIN-only audit; no model, data loader or server access."""
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPORT_SHA = 'b1a29cb7227746f8b6ed3135ca23b9860220c1d519bee66bbb56e0bcb9631949'


def main():
    raw = (ROOT / 'analysis/report.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REPORT_SHA
    r = json.loads(raw)
    assert (r['requested_parents'], r['requested_inputs'], r['positive_references']) == (16, 48, 285)
    result = dict(source_report_sha256=REPORT_SHA, scope='Descriptive aggregation of the already sealed report only',
                  new_model_requests=0, new_paths=0, references={}, predictions={})
    for version in ('raw', 'model_H24'):
        rows = [v for v in r['references'] if v['representation'] == version]
        pairs = r['reference_pairs'][version]
        assert len(rows) == 285 and len(pairs) == 48
        counts = Counter(('same' if p['same'] else 'none') + '+' + ('different' if p['different'] else 'none') for p in pairs)
        result['references'][version] = dict(total_references=len(rows), known=sum(v['type'] is not None for v in rows),
            unknown=sum(v['type'] is None for v in rows), tip_valid=sum(v['tip_valid'] for v in rows),
            type_changes=sum(v['type_changed_by_representation'] for v in rows), pair_availability=dict(counts),
            all_cross_goal_route_pairs=sum(p['all_route_pairs'] for p in pairs),
            both_known_route_pairs=sum(p['known_route_pairs'] for p in pairs),
            simple_both_known_route_pairs=sum(p['simple_known_route_pairs'] for p in pairs),
            ineligible_pairs=[dict(parent_id=p['parent_id'], left_id=p['left_id'], right_id=p['right_id'],
                missing_same=p['same'] is None, missing_different=p['different'] is None) for p in pairs if p['same'] is None or p['different'] is None],
            original_summary=r['reference_summary'][version])
    for stage in ('best', 'last'):
        rows = r['prediction_association'][stage]['records']
        assert len(rows) == len(r['prediction_candidates'][stage]) == 192
        categories, annotated = Counter(), []
        for v in rows:
            if not v['semantic_correct']: reason = 'incorrect_goal'
            elif not v['known_type']: reason = 'unknown_type'
            elif not v['simple_forward']: reason = 'ambiguous_crossing'
            elif v['mean_excess_m'] is None: reason = 'no_evaluable_positive_correspondence'
            else: reason = 'eligible'
            categories[reason] += 1
            annotated.append(dict(id=v['id'], candidate=v['candidate'], category=reason))
        candidates = r['prediction_candidates'][stage]
        summary = {k: v for k, v in r['prediction_association'][stage].items() if k != 'records'}
        result['predictions'][stage] = dict(total_slots=192, mutually_exclusive_exclusion_order=list((
            'incorrect_goal', 'unknown_type', 'ambiguous_crossing', 'no_evaluable_positive_correspondence', 'eligible')),
            exclusion_counts=dict(categories), exclusion_records=annotated, original_summary=summary,
            tip_valid=sum(v['outcome']['TipValid'] for v in candidates),
            semantic_correct=sum(v['outcome']['semantic_goal_correct'] for v in candidates),
            tip_clear=sum(v['outcome']['tip_segments_clear'] for v in candidates),
            eligible_collision_records=[v for v in rows if v['semantic_correct'] and v['mean_excess_m'] is not None and v['collision']])
    status = json.loads((ROOT / 'analysis.status.json').read_text())
    assert status['status'] == 'completed' and status['exit_code'] == 0
    result['budget'] = dict(diagnostic_body_seconds=r['elapsed_seconds'],
        recorded_process_seconds=(datetime.fromisoformat(status['end_utc']) - datetime.fromisoformat(status['start_utc'])).total_seconds(),
        gpu_hours=r['gpu_hours'], new_forward_requests=r['new_forward_requests'], new_candidate_states=r['new_candidate_states'],
        repaired_paths=r['repaired_paths'], cpu_affinity=r['cpu_affinity'], torch_threads=r['torch_threads'])
    result['joint_opportunity_screen_passed'] = r['joint_opportunity_screen_passed']
    result['decision'] = 'Stop this cross-goal auxiliary mechanism experiment under its preregistered gate; do not widen thresholds or consult DEV.'
    (ROOT / 'DERIVED_SUMMARY.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
