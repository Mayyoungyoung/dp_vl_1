"""Pure continuation lineage and paired-probe rejection tests; zero model calls."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from routeset.vlm_sft_continuation import continuation_accounting
from routeset.vlm_sft_continued_eval import (SOURCE_COMMIT, ORIGINAL_FILES, CONTINUATION_SOURCES, PROBE_PROTOCOL,
    validate_receipts, validate_saved_checkpoint, validate_inherited_metadata)
from scripts.analyze_vlm_sft_continued_train8 import validate_pair

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    # Small actual metadata only; never images/routes or actual model weights.
    folder = ROOT/'reports/vlm_route_sft_continuation_v1/gpu_launch'
    lineage = json.loads((folder/'continuation_source.json').read_text(encoding='utf-8'))
    config = json.loads((folder/'config.json').read_text(encoding='utf-8'))
    original = json.loads((ROOT/'reports/vlm_route_sft_v1/training/summary.json').read_text(encoding='utf-8'))
    summary = deepcopy(original)
    summary.update(step=6000,planned_steps=6000,code_commit=SOURCE_COMMIT,best_step=5750,
        best_dev_token_nll=.4,elapsed_seconds=2200.,continuation=lineage,best_is_inherited=False,
        original_source_files_unchanged=True,added_gradient_audit=deepcopy(original['gradient_audit']))
    summary['artifacts_sha256']['best.pt'] = 'new-best'
    summary['exposure']['TRAIN'].update(requests=6000,candidate_slots=15000)
    summary['continuation_accounting'] = continuation_accounting(lineage,summary['exposure'],summary['elapsed_seconds'])
    return summary,config,lineage


def test_actual_original_metadata_and_registered_new_source_accepted():
    summary,config,lineage = fixture()
    assert lineage['source_file_sha256'] == ORIGINAL_FILES
    assert all(config['source_sha256'][n] == h for n,h in CONTINUATION_SOURCES.items())
    assert validate_receipts(summary,config,lineage,'new-best')['added_exposure']['TRAIN']['candidate_slots'] == 11250


def test_inherited_incomplete_or_unregistered_result_cannot_generate():
    summary,config,lineage = fixture()
    for key,value in [('best_is_inherited',True),('status','running'),('step',5999),('planned_steps',6250),
                      ('code_commit','other'),('best_step',1500),('original_source_files_unchanged',False)]:
        wrong = deepcopy(summary); wrong[key] = value
        with pytest.raises(ValueError): validate_receipts(wrong,config,lineage,'new-best')
    with pytest.raises(ValueError): validate_receipts(summary,config,lineage,'different-bytes')
    wrong = deepcopy(summary); wrong['added_gradient_audit'].pop(next(iter(wrong['added_gradient_audit'])))
    with pytest.raises(ValueError): validate_receipts(wrong,config,lineage,'new-best')
    wrong = deepcopy(summary);wrong['continuation_accounting']['added_elapsed_seconds'] -= 1.
    with pytest.raises(ValueError): validate_receipts(wrong,config,lineage,'new-best')


def test_changed_data_source_or_original_checkpoint_rejected():
    summary,config,lineage = fixture()
    for key,value in [('lr',.0002),('data_fingerprint','changed'),('eval_every',500)]:
        wrong=deepcopy(config);wrong[key]=value
        with pytest.raises(ValueError):validate_receipts(summary,wrong,lineage,'new-best')
    wrong=deepcopy(config);wrong['source_sha256']['routeset/vlm_sft_loss.py']='changed'
    with pytest.raises(ValueError):validate_receipts(summary,wrong,lineage,'new-best')
    wrong=deepcopy(config);wrong['source_sha256']['scripts/train_vlm_route_sft.py']='other-trainer'
    with pytest.raises(ValueError):validate_receipts(summary,wrong,lineage,'new-best')
    changed=deepcopy(lineage);changed['source_file_sha256']['last.pt']='different-origin'
    wrong=deepcopy(summary);wrong['continuation']=changed
    with pytest.raises(ValueError):validate_receipts(wrong,config,changed,'new-best')


def test_checkpoint_internal_config_lineage_step_and_selection_checked():
    summary,config,lineage=fixture()
    saved=dict(protocol=summary['protocol'],config=config,continuation=lineage,best_is_inherited=False,
               step=5750,best_step=5750,best_nll=.4)
    validate_saved_checkpoint(saved,summary,config,lineage)
    for key,value in [('config',{}),('continuation',{}),('step',6000),('best_nll',.399),('best_is_inherited',True)]:
        wrong=deepcopy(saved);wrong[key]=value
        with pytest.raises(ValueError):validate_saved_checkpoint(wrong,summary,config,lineage)


def test_inherited_cost_checked_against_actual_original_summary():
    _,_,lineage=fixture()
    original=json.loads((ROOT/'reports/vlm_route_sft_v1/training/summary.json').read_text(encoding='utf-8'))
    validate_inherited_metadata(lineage,original)
    for key,value in [('inherited_elapsed_seconds',500.),('inherited_best_step',1250),('inherited_exposure',{})]:
        wrong=deepcopy(lineage);wrong[key]=value
        with pytest.raises(ValueError):validate_inherited_metadata(wrong,original)


def pair_fixture():
    old=json.loads((ROOT/'reports/vlm_route_greedy_train8_v1/train8/summary.json').read_text(encoding='utf-8'))
    summary,config,lineage=fixture()
    receipt=dict(selected_checkpoint_sha256='new-best',selected_step=5750,training_summary_sha256='completed6000')
    new=deepcopy(old)
    new.update(config=config,checkpoint_sha256='new-best',checkpoint_step=5750,protocol=PROBE_PROTOCOL,
        evaluation_scope='continued_train8_preflight',continuation_lineage=receipt,
        training_summary_sha256='completed6000')
    return old,new,config,receipt


def test_pair_allows_only_training_extension_and_same_actual_decoder_inputs():
    old,new,config,receipt=pair_fixture();validate_pair(old,new,config,receipt)
    for key,value in [('generation_input_sha256',{}),('requested_calls',9),('requested_slots',7),
                      ('request_loop_limit_seconds',240.),('generation_use_model_defaults',True),
                      ('evaluation_scope','dev24_comparison'),('checkpoint_sha256','other')]:
        wrong=deepcopy(new);wrong[key]=value
        with pytest.raises(ValueError):validate_pair(old,wrong,config,receipt)
    wrong=deepcopy(new);wrong['effective_generation_config']['num_beams']=2
    with pytest.raises(ValueError):validate_pair(old,wrong,config,receipt)
    wrong=deepcopy(new);wrong['source_sha256']['routeset/vlm_sft_train_probe.py']='changed'
    with pytest.raises(ValueError):validate_pair(old,wrong,config,receipt)
