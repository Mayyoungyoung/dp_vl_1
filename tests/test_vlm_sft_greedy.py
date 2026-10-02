"""Finite greedy control retains old decoding and all failed requested slots."""
from copy import deepcopy
import json

import numpy as np
import pytest

from routeset.vlm_sft_train_probe import (PARENTS, decoding_kwargs, generate_train_probe,
    validate_greedy_scope)
from scripts.analyze_vlm_greedy_train8_pair import (CHECKPOINT_SHA, endpoint_metrics,
    validate_pair_receipts)


def samples():
    return [dict(id=p+'_target0',parent_id=p,split='TRAIN',instruction='reach red',
                 image_path='unopened.png',observation_path='unopened.npz') for p in PARENTS]


def test_old_decoding_exact_and_greedy_has_no_sampling_controls():
    assert decoding_kwargs() == dict(do_sample=True,temperature=.7,top_p=.9,top_k=0,num_beams=1,
        num_return_sequences=1,repetition_penalty=1.,use_cache=True)
    greedy = decoding_kwargs(True)
    assert greedy['do_sample'] is False
    assert all(greedy[k] is None for k in ('temperature','top_p','top_k'))
    assert greedy['num_beams'] == greedy['num_return_sequences'] == 1
    for scope,seed,repeats in [('dev24_comparison',0,1),('train8_preflight',1,1),('train8_preflight',0,2)]:
        with pytest.raises(ValueError): validate_greedy_scope(True,scope,seed,repeats)
    validate_greedy_scope(False,'dev24_comparison',3,5)


def test_exact_eight_seed_matched_calls_without_hidden_candidates(tmp_path):
    calls = []
    def generate(sample,k,seed,cap):
        calls.append((sample['id'],k,seed,cap))
        return dict(text=json.dumps([[[0,0,0,1]]*24]),tokens={})
    old = generate_train_probe(samples(),generate,tmp_path/'old')[0]; old_calls = calls.copy(); calls.clear()
    new = generate_train_probe(samples(),generate,tmp_path/'new',greedy=True)[0]
    assert calls == old_calls and len(calls) == 8
    assert all(c[1] == 1 and c[3] == 512 for c in calls)
    assert new['attempted_autoregressive_calls'] == new['charged_candidate_slots'] == old['charged_candidate_slots'] == 8
    assert new['budget_unattempted_slots'] == 0 and new['strict_finite_slots'] == 8
    rows = [json.loads(x) for x in (tmp_path/'new/requests.jsonl').read_text().splitlines()]
    assert all(r['request_attempted'] and r['method'] == 'train8_constrained_greedy_k1' for r in rows)


def test_time_limit_retains_unattempted_slots_and_no_retry(tmp_path):
    clock = [0.];calls = []
    def generate(sample,k,seed,cap):
        calls.append(sample['id']); clock[0] = 181.
        return dict(text=json.dumps([[[0,0,0,1]]*24]),tokens={})
    result = generate_train_probe(samples(),generate,tmp_path/'probe',greedy=True,clock=lambda:clock[0])[0]
    assert len(calls) == result['attempted_autoregressive_calls'] == 1
    assert result['requested_candidate_slots'] == result['charged_candidate_slots'] == 8
    assert result['budget_unattempted_slots'] == result['failed_requests'] == 7
    assert result['request_loop_budget_overshoot_seconds'] == 1.
    with np.load(tmp_path/'probe/predictions.npz') as saved:
        assert np.isnan(saved['paths'][1:]).all()
    rows = [json.loads(x) for x in (tmp_path/'probe/requests.jsonl').read_text().splitlines()]
    assert all(r['failure']['type'] == 'TimeoutError' and not r['request_attempted'] and r['text'] == '' for r in rows[1:])


def receipts():
    old = dict(status='completed',evaluation_scope='train8_preflight',split='TRAIN',examples=8,seed=0,repeats=1,
        checkpoint_sha256=CHECKPOINT_SHA,checkpoint_step=1500,config=dict(fingerprint='same'),
        generation_input_sha256={'file':'same'},sampling_plan_sha256='same',training_summary_sha256='same',
        export_manifest_sha256='same',source_sha256={k:'same' for k in ('routeset/vlm_route_grammar.py',
            'routeset/vlm_sft_data.py','routeset/vlm_route_serialization.py','scripts/train_observed_lora.py')},
        syntax_constraint=dict(protocol='same',compact_json=True,k_exact=True,horizon=24,
            coordinate_integer_range=[-10000,10000],event_values=[0,1],posthoc_repair=False,vocabulary=dict(sha256='same')),
        decoding=decoding_kwargs())
    new = deepcopy(old)
    new.update(decoding=decoding_kwargs(True),decoding_mode='greedy',request_loop_limit_seconds=180.,
        generation_use_model_defaults=False,effective_generation_mode='greedy_search',
        effective_generation_config=dict(decoding_kwargs(True),max_new_tokens=512))
    return old,new


def test_pair_rejects_information_budget_or_effective_decoder_drift():
    old,new = receipts(); validate_pair_receipts(old,new)
    for key,value in [('config',{}),('generation_input_sha256',{}),('checkpoint_sha256','changed'),
                      ('generation_use_model_defaults',True),('request_loop_limit_seconds',300.)]:
        changed = deepcopy(new); changed[key] = value
        with pytest.raises(ValueError): validate_pair_receipts(old,changed)
    for key,value in [('do_sample',True),('temperature',.7),('num_beams',4),('max_new_tokens',1024)]:
        changed = deepcopy(new);changed['effective_generation_config'][key] = value
        with pytest.raises(ValueError): validate_pair_receipts(old,changed)


def test_pair_missing_slot_never_disappears_from_useful_control_denominator():
    paths = np.zeros((8,1,24,3)); events = np.ones((8,1,24))
    targets = [dict(centers=[[0,0,0],[1,0,0]],target_index=0,tolerance=.03)]*8
    result = endpoint_metrics(paths,events,targets)
    assert result['correct_requested_target_count'] == 8 and result['predeclared_useful_control']
    paths[0] = np.nan
    result = endpoint_metrics(paths,events,targets)
    assert result['requested_slots'] == 8 and result['strict_format_finite'] == 7
    assert result['correct_requested_target_count'] == 7
    assert result['endpoint_error_mean_m_over_finite_only'] == 0.
    assert result['all_eight_mean_endpoint_error_m'] is None and not result['predeclared_useful_control']


def test_closed_eight_slots_preserve_format_failure_and_cap(tmp_path):
    calls = []
    def generate(sample,k,seed,cap):
        calls.append(1)
        if len(calls)==1: raise RuntimeError('real callback failure')
        return dict(text='[[[0,0,0,1]',tokens=dict(reached_token_limit=True))
    result = generate_train_probe(samples(),generate,tmp_path/'failure',greedy=True)[0]
    assert len(calls)==8 and result['attempted_autoregressive_calls']==8
    assert result['strict_finite_slots']==0 and result['charged_candidate_slots']==8
    assert result['failed_requests']==1 and result['reached_token_limit']==7
