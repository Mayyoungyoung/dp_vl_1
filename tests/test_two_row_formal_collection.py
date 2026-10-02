import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest
from scripts import collect_two_row_formal as formal

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    path=tmp_path_factory.mktemp('formal_source')/'corpus'
    formal.prepare(ROOT/'configs/observed_two_row_formal116_registered_v1.json',path)
    return path


@pytest.fixture
def corpus(prepared,tmp_path):
    path=tmp_path/'corpus';shutil.copytree(prepared,path);return path


def test_fixed_corpus_role_source_and_parent_budget(corpus):
    value,manifest=formal.verify_corpus(corpus)
    assert value['requested_routes']==3132 and len(value['shards'][0])==58
    p=value['parent_plan'][100];cfg=formal.parent_config(value,p,manifest)
    assert cfg['role']=='TEST_LOCKED' and cfg['registered_parent']['config']['split']=='TEST_LOCKED'
    assert cfg['requested_route_proposals']==27 and cfg['requested_setup_actions']==0
    assert 'saved_world' not in cfg['registered_parent']


def test_registration_byte_change_refuses_resume(corpus):
    path=corpus/'registration.json';path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='SHA changed'):formal.verify_corpus(corpus)


def test_source_change_refuses_resume(corpus,monkeypatch):
    original=formal.source_hashes
    monkeypatch.setattr(formal,'source_hashes',lambda p:dict(original(p),unexpected='changed'))
    with pytest.raises(ValueError,match='source differs'):formal.verify_corpus(corpus)


def test_phase_barrier_requires_entire_prior_cohorts():
    assert formal.prior_phase_indices(0)==[]
    assert formal.prior_phase_indices(64)==list(range(16))
    assert formal.prior_phase_indices(16)==list(range(16))+list(range(64,76))
    assert formal.prior_phase_indices(100)==list(range(76))


def ledger(path,parent,events):
    path.parent.mkdir(parents=True,exist_ok=True)
    for event,attempt in events:
        formal.physical.legacy.append_json(path,dict(event=event,parent_id=parent,input_id=parent+'_target0',attempt=attempt))


def test_interruption_preserves_issued_slot_and_remaining_budget(tmp_path):
    parent='two_row_reach_283200';path=tmp_path/'slot_ledger.jsonl'
    ledger(path,parent,[('started',0),('completed',0),('started',1)])
    p=dict(parent_id=parent,index=0,role='TRAIN',collection_allowed=True,registration_eligible=True,registered_geometry_1mm_sha256='h')
    row=formal.mechanical_closure(tmp_path,p,{},None,None)
    assert row['attempted_lower']==row['attempted_upper']==2
    assert row['completed_slots']==1 and row['unattempted_lower']==row['unattempted_upper']==25
    assert row['worker_elapsed_unknown'] and not row['model_eligible']


def test_partial_ledger_line_has_one_slot_uncertainty(tmp_path):
    parent='two_row_reach_283200';path=tmp_path/'slot_ledger.jsonl'
    ledger(path,parent,[('started',0),('completed',0)])
    with path.open('a') as stream:stream.write('{"event":"start')
    assert formal.slot_counts(path,parent)==(1,1,True)


def test_duplicate_slot_rejected(tmp_path):
    path=tmp_path/'slot_ledger.jsonl';parent='two_row_reach_283200'
    ledger(path,parent,[('started',0),('completed',0),('started',0)])
    with pytest.raises(ValueError,match='duplicate'):formal.slot_counts(path,parent)


def test_worker_failure_before_data_is_closed_at_zero_budget(tmp_path):
    p=dict(parent_id='two_row_reach_283200',index=0,role='TRAIN',collection_allowed=True,registration_eligible=True,registered_geometry_1mm_sha256='h')
    row=formal.mechanical_closure(tmp_path/'missing',p,{},1.,1)
    assert row['status']=='worker_runtime_failed' and row['attempted_upper']==0 and row['unattempted_lower']==27


def save_closure(corpus,index,actual=None):
    value,_=formal.verify_corpus(corpus);p=value['parent_plan'][index]
    actual=actual or p['registered_geometry_1mm_sha256'];data=corpus/'parents'/p['role']/p['parent_id']
    data.mkdir(parents=True,exist_ok=True)
    formal.batch.write(data/'mechanical_receipt.json',dict(role=p['role'],started_slots=0,completed_slots=0,
        layouts=[dict(parent_id=p['parent_id'],actual_geometry_1mm_sha256=actual,initial_observation_saved=True)]))
    row=formal.mechanical_closure(data,p,{},1.,0)
    formal.atomic_write(corpus/'closures'/('%03d.json'%index),row)
    return data,row


def test_closed_mechanical_hash_change_blocks_gate(corpus):
    data,row=save_closure(corpus,0)
    path=data/'mechanical_receipt.json';path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='evidence changed'):formal.live_layout_gate(corpus)


def test_closed_wrong_role_rejected(corpus):
    data,row=save_closure(corpus,0);row['role']='TEST_LOCKED'
    formal.atomic_write(corpus/'closures/000.json',row)
    with pytest.raises(ValueError,match='index/role'):formal.live_layout_gate(corpus)


def test_actual_geometry_mismatch_and_duplicate_block_whole_group(corpus):
    _,a=save_closure(corpus,0);_,b=save_closure(corpus,64,a['actual_geometry_1mm_sha256'])
    result=formal.live_layout_gate(corpus)
    assert result['blocked_parent_ids']==[a['parent_id'],b['parent_id']]
    assert result['duplicate_groups'][0]['roles']==['DEV_MODEL','TRAIN']


def test_locked_gate_reads_only_mechanical_metadata(corpus,monkeypatch):
    data,_=save_closure(corpus,100);(data/'front.png').write_bytes(b'never open locked pixels')
    original=Path.open
    def guarded(path,*args,**kwargs):
        if path.name=='front.png':raise AssertionError('locked raw opened')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    assert formal.live_layout_gate(corpus)['raw_locked_opened'] is False


def test_closed_parent_is_not_relaunched_and_live_shard_refused(corpus,tmp_path,monkeypatch):
    save_closure(corpus,0)
    monkeypatch.setattr(formal.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not start worker')))
    assert formal.run_shard(corpus,tmp_path/'runs',0,Path('python'),max_new_parents=0)==0
    with formal.shard_lock(corpus/'shards/0/coordinator.lock'):
        with pytest.raises(RuntimeError,match='still alive'):
            formal.run_shard(corpus,tmp_path/'runs',0,Path('python'),resume=True,max_new_parents=0)
