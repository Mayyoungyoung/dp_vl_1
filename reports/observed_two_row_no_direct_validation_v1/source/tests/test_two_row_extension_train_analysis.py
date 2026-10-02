import json
from pathlib import Path
import shutil

import pytest

from scripts import analyze_two_row_extension_train as audit

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    root=tmp_path_factory.mktemp('extension_analysis')/'corpus'
    audit.extension.prepare(ROOT/'configs/observed_two_row_extension288_registered_v1.json',root)
    return root


@pytest.fixture
def corpus(prepared,tmp_path):
    root=tmp_path/'corpus';shutil.copytree(prepared,root);return root


def close_failed(corpus,count):
    value,_=audit.extension.verify_corpus(corpus)
    for plan in value['parent_plan'][:count]:
        data=corpus/'parents'/'TRAIN'/plan['parent_id']
        row=audit.extension.old.mechanical_closure(data,plan,{},1.,1)
        audit.extension.old.atomic_write(corpus/'closures'/('%03d.json'%plan['index']),row)


@pytest.mark.parametrize('count',[0,16,31,33,255,288,True,'32'])
def test_only_registered_train_prefixes(count):
    with pytest.raises(ValueError,match='fixed TRAIN prefixes'):audit.prefix_indices(count)


def test_missing_closure_blocks_before_raw_or_output(corpus,tmp_path,monkeypatch):
    monkeypatch.setattr(audit.TrainReader,'__init__',lambda *args:pytest.fail('raw read before closure gate'))
    with pytest.raises(ValueError,match='entire fixed TRAIN prefix'):
        audit.analyze(corpus,tmp_path/'out',32,plots=False)
    assert not (tmp_path/'out').exists()


def test_live_mechanical_snapshot_keeps_unfinished_budget_separate(corpus,monkeypatch):
    close_failed(corpus,2)
    monkeypatch.setattr(audit.TrainReader,'__init__',lambda *args:pytest.fail('mechanical cannot read raw'))
    row=audit.mechanical_snapshot(corpus,32)
    assert row['closed_train_parents']==2 and row['requested_slots']==864
    assert row['closed_parent_unattempted_lower']==54
    assert row['unfinished_parent_requested_slots']==810
    assert row['missing_closure_indices']==list(range(2,32))
    assert not row['all_raw_opened'] and row['new_dev_raw_sealed']


@pytest.mark.parametrize('count',[32,64,128,256])
def test_all_failed_prefix_preserves_every_parent_condition_slot(corpus,tmp_path,monkeypatch,count):
    close_failed(corpus,count)
    original=Path.open
    def guard(path,*args,**kwargs):
        if 'DEV_MODEL' in path.parts or 'TEST_LOCKED' in path.parts:
            pytest.fail('sealed raw was opened')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guard)
    result=audit.analyze(corpus,tmp_path/'out',count,plots=False)
    assert result['requested_parents']==count and result['requested_conditions']==3*count
    assert result['requested_slots']==27*count and result['slot_status_counts']=={'unattempted':27*count}
    assert result['unattempted_lower']==27*count and result['missing_observation_count']==3*count
    assert result['accepted_per_requested_slot']==0 and result['accepted_length_m'] is None
    assert len(result['conditions'])==3*count and result['distinct_known_type_histogram']=={0:3*count}
    assert len(json.loads((tmp_path/'out/all_requested_slots.json').read_text()))==27*count
    assert result['selected_train_indices']==list(range(count))
    assert result['new_dev_raw_sealed'] and result['reference_types_are_incomplete']


def test_closure_hash_mutation_rejected_before_raw(corpus,tmp_path,monkeypatch):
    close_failed(corpus,32)
    path=corpus/'closures/000.json';row=json.loads(path.read_text());row['role']='DEV_MODEL'
    path.write_text(json.dumps(row))
    monkeypatch.setattr(audit.TrainReader,'__init__',lambda *args:pytest.fail('raw opened on bad gate'))
    with pytest.raises(ValueError):audit.analyze(corpus,tmp_path/'out',32,plots=False)


def test_historical_trace_and_type_helpers_are_reused_without_redefinition():
    assert audit.TrainReader is audit.historical.TrainReader
    assert audit.trace_diagnostics is audit.historical.trace_diagnostics
    assert audit.plot_target is audit.historical.plot_target
    assert audit.indexed_slots is audit.historical.indexed_slots
    assert audit.condition_summary is audit.historical.condition_summary
