"""Finite TRAIN-only probe cannot masquerade as DEV or retry failed calls."""
import json
import numpy as np
import pytest
from routeset.vlm_sft_train_probe import PARENTS, generate_train_probe, validate_probe_samples


def samples():
    return [dict(id=p+'_target0', parent_id=p, split='TRAIN', instruction='reach red',
                 image_path='unopened.png', observation_path='unopened.npz') for p in PARENTS]


def test_exact_eight_calls_failure_and_truncated_prefix_remain_nan_without_retry(tmp_path):
    calls = []
    def generator(sample, k, seed, cap):
        calls.append((k, cap))
        if len(calls) == 1: raise RuntimeError('deliberate dead end')
        if len(calls) == 2: return dict(text='[[[0,0,0,1]', tokens=dict(reached_token_limit=True))
        return dict(text=json.dumps([[[0, 0, 0, 1]]*24]), tokens=dict(reached_token_limit=False))
    result = generate_train_probe(samples(), generator, tmp_path/'probe')[0]
    assert calls == [(1, 512)]*8 and result['requested_candidate_slots'] == result['charged_candidate_slots'] == 8
    assert result['failed_requests'] == 1 and result['reached_token_limit'] == 1 and result['strict_finite_slots'] == 6
    with np.load(tmp_path/'probe/predictions.npz') as saved:
        assert np.isnan(saved['paths'][:2]).all() and saved['paths'].shape == (8, 1, 24, 3)
    rows = [json.loads(line) for line in (tmp_path/'probe/requests.jsonl').read_text().splitlines()]
    assert all(r['split'] == 'TRAIN' for r in rows)
    assert rows[1]['text'] == '[[[0,0,0,1]'


def test_dev_relabel_answer_fields_parent_substitution_and_output_overwrite_rejected(tmp_path):
    for field, value in [('split', 'DEV_MODEL'), ('id', PARENTS[0]+'_target1'), ('answer', 'forbidden')]:
        rows = samples(); rows[0][field] = value
        with pytest.raises(ValueError): validate_probe_samples(rows)
    with pytest.raises(ValueError): validate_probe_samples(samples()[:-1])
    with pytest.raises(FileExistsError): generate_train_probe(samples(), None, tmp_path)
