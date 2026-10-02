"""Summarize a completed TRAIN-only coverage audit without opening model/data.

This post-processing preserves its raw source and does not rerun inference,
select thresholds, or treat privileged reference replacement as model gain.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def analyze(folder):
    folder = Path(folder)
    summary = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
    rows = json.loads((folder / 'per_scene.json').read_text(encoding='utf-8'))
    if summary['split'] != 'TRAIN' or summary['raw_dev_or_locked_files_opened']:
        raise ValueError('A completed TRAIN-only audit is required')
    if len(rows) != summary['observations']:
        raise ValueError('Incomplete per-scene audit')
    affected = [row for row in rows if row['coverage']['duplicate_replacement_reference_oracle_gain'] > 0]
    ids = {row['scene_id'] for row in affected}
    def quantiles(items):
        values = [r['assignment']['second_minus_best'] for r in items
                  if r['assignment'] is not None and r['assignment']['second_minus_best'] is not None]
        return dict(count=len(values), levels=[0, .25, .5, .75, 1],
                    values=np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else [])
    assignments = Counter()
    details = []
    known_assignments = {}
    for label, group in [('all', rows), ('affected', affected),
                         ('other', [r for r in rows if r['scene_id'] not in ids])]:
        counts = Counter()
        for row in group:
            if row['assignment'] is None:
                continue
            for j, ref_idx in enumerate(row['assignment']['reference_indices']):
                ref = row['references'][ref_idx]
                typ = ref['declared_passage_type']
                if not typ:
                    continue
                pred = row['candidates'][j]
                counts['known_reference_assigned'] += 1
                counts['same_declared_type'] += pred['declared_passage_type'] == typ
                counts['collision'] += not pred['tip_segments_clear']
                counts['valid_same_type'] += pred['TipValid'] and pred['declared_passage_type'] == typ
        known_assignments[label] = dict(counts)
    for row in affected:
        missing = {tuple(t) for t in row['coverage']['missing_supported_known_types']}
        for j, ref_idx in enumerate(row['assignment']['reference_indices']):
            typ = row['references'][ref_idx]['declared_passage_type']
            if tuple(typ or []) not in missing:
                continue
            pred = row['candidates'][j]
            category = ('valid_other_type' if pred['classified_tip_valid'] else
                        'valid_unknown' if pred['TipValid'] else 'invalid')
            assignments['assigned_missing_known_type'] += 1
            assignments[category] += 1
            assignments['collision'] += not pred['tip_segments_clear']
            assignments['semantic_wrong'] += not pred['semantic_goal_correct']
            details.append(dict(scene_id=row['scene_id'], parent_id=row['parent_id'],
                                candidate=j, reference_index=ref_idx, reference_type=typ,
                                predicted_type=pred['declared_passage_type'], category=category,
                                cost=row['assignment']['cost_matrix'][j][ref_idx]))
    interpolation = {str(alpha): Counter() for alpha in (.25, .5, .75)}
    for row in rows:
        for probe in row['interpolation_probes']:
            if probe['reference_type_relation'] == 'different_known':
                counts = interpolation[str(probe['alpha'])]
                counts['probes'] += 1
                counts['collisions'] += not probe['tip_check']['tip_segments_clear']
    output = dict(protocol='observed_train_reference_coverage_postprocess_v1',
        source_sha256={name:hashlib.sha256((folder / name).read_bytes()).hexdigest()
                       for name in ('summary.json', 'per_scene.json')},
        source_analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        affected_instructions=len(affected), affected_parents=len({r['parent_id'] for r in affected}),
        affected_reference_counts=dict(Counter(str(r['reference_count']) for r in affected)),
        affected_duplicate_slots=sum(r['coverage']['valid_classified_duplicate_candidates'] for r in affected),
        affected_missing_types=dict(Counter(t[0] for r in affected
            for t in r['coverage']['missing_supported_known_types'])),
        oracle_gain_per_instruction=summary['duplicate_replacement_reference_oracle_gain'] / len(rows),
        assignment_gap_quantiles=dict(all=quantiles(rows), affected=quantiles(affected),
            other=quantiles([r for r in rows if r['scene_id'] not in ids])),
        known_reference_assignments=known_assignments,
        assignments_to_missing_known_types=dict(assignments),
        missing_assignment_details=details,
        different_known_interpolation_by_alpha={key:dict(value) for key,value in interpolation.items()},
        limitations='Post-hoc TRAIN associations from one DEV-selected checkpoint; not causal evidence. '
                    'Unknowns remain unknown. Oracle opportunity is not attainable model gain. '
                    'No model, data manifests, DEV, locked content, or new candidate pools opened.')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier post-processing; use a fresh output')
    result = analyze(args.audit)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({key:result[key] for key in ('affected_instructions','affected_parents',
        'affected_duplicate_slots','assignments_to_missing_known_types')}))
