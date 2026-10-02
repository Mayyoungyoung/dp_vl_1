from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from scripts import analyze_observed_spatial_penalty_astar as audit


def fixture():
    paths = np.zeros((4, 24, 3)); paths[3] = np.nan
    opened = np.ones((4, 24))
    raw = [np.zeros((2, 3)) for _ in range(3)]+[np.empty((0, 3))]
    candidates = []
    for k in range(4):
        flags = {name: True for name in audit.CHECKS}
        if k == 3:
            flags.update(finite_xyz=False, semantic_goal_correct=False,
                         starts_at_current_state=False, tip_segments_clear=False)
        candidates.append(dict(candidate=k, **flags, TipValid=k < 3,
                               declared_passage_type=['middle', 'middle'] if k < 2 else None,
                               classified_tip_valid=k < 2))
    attempts = [dict(slot=k, status='path_found' if k < 3 else 'no_attachment',
                     complete_raw_paths_emitted=int(k < 3)) for k in range(4)]
    return candidates, paths, opened, raw, attempts


def test_unknown_remains_valid_duplicate_and_missing_slot_counted():
    result = audit.summarize_candidates(*fixture())
    assert result['valid'] == 3 and result['unknown_valid'] == 1
    assert result['known_unique'] == result['known_duplicate'] == 1
    assert result['failed_generation'] == 1 and result['raw_complete'] == 3
    assert result['outcomes']['no_complete_path:no_attachment'] == 1


def test_finite_wrong_goal_separate_from_generation_failure():
    data = fixture(); data[0][1].update(semantic_goal_correct=False, TipValid=False, classified_tip_valid=False)
    result = audit.summarize_candidates(*data)
    assert result['finite_invalid'] == 1 and result['failed_generation'] == 1
    assert result['outcomes']['finite_invalid:semantic_goal_correct'] == 1


@pytest.mark.parametrize('corruption', ['missing_slot', 'false_valid', 'raw_mismatch', 'event_mismatch'])
def test_corrupt_candidate_accounting_rejected(corruption):
    c, p, e, r, a = fixture()
    if corruption == 'missing_slot':
        c.pop()
    elif corruption == 'false_valid':
        c[3]['TipValid'] = True
    elif corruption == 'raw_mismatch':
        r[0] = np.empty((0, 3))
    else:
        e[0, 0] = np.nan
    with pytest.raises(ValueError):
        audit.summarize_candidates(c, p, e, r, a)


def test_pair_by_id_not_record_order():
    def row(i, number):
        return dict(id=i, parent_id='parent', **{k: number for k in
            ('valid', 'known_unique', 'known_duplicate', 'unknown_valid', 'failed_generation', 'request_seconds', 'field_seconds')})
    edge = dict(cases=[row('a', 1), row('b', 3)])
    spatial = dict(cases=[row('b', 4), row('a', 1)])
    result = audit.paired_cases(edge, spatial)
    assert [r['delta']['valid'] for r in result] == [0, 1]
    with pytest.raises(ValueError):
        audit.paired_cases(edge, dict(cases=[row('a', 1)]))


def test_posix_artifact_mapping_and_hash_rejection(tmp_path):
    run, binary = tmp_path/'report', tmp_path/'binary'
    run.mkdir(); binary.mkdir()
    (run/'fitted_train_model.json').write_text('{}')
    np.savez(binary/'edge_predictions.npz', paths=np.zeros((1, 4, 24, 3)))
    files = {f'/home/wzy/runs/example/{name}': audit.sha(path/name) for path, name in
             [(run, 'fitted_train_model.json'), (binary, 'edge_predictions.npz')]}
    report = dict(artifact_sha256=files)
    assert len(audit.verify_artifacts(report, run, binary)) == 2
    bad = deepcopy(report); bad['artifact_sha256']['/elsewhere/not_part_of_run'] = '0'*64
    with pytest.raises(ValueError, match='outside'):
        audit.verify_artifacts(bad, run, binary)
    (run/'fitted_train_model.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='hash'):
        audit.verify_artifacts(report, run, binary)
