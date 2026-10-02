import copy
import json
from pathlib import Path
import shutil

import pytest
from scripts import collect_two_row_extension as collect

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def prepared(tmp_path_factory):
    path=tmp_path_factory.mktemp('extension_source')/'corpus'
    collect.prepare(ROOT/'configs/observed_two_row_extension288_registered_v1.json',path)
    return path


@pytest.fixture
def corpus(prepared,tmp_path):
    path=tmp_path/'corpus';shutil.copytree(prepared,path);return path


def save_closure(corpus,value,index,actual=None):
    p=value['parent_plan'][index];data=corpus/'parents'/p['role']/p['parent_id']
    data.mkdir(parents=True,exist_ok=True)
    collect.batch.write(data/'mechanical_receipt.json',dict(role=p['role'],started_slots=0,completed_slots=0,
        layouts=[dict(parent_id=p['parent_id'],actual_geometry_1mm_sha256=actual or p['registered_geometry_1mm_sha256'],initial_observation_saved=True)]))
    row=collect.old.mechanical_closure(data,p,{},1.,0)
    collect.old.atomic_write(corpus/'closures'/('%03d.json'%index),row)
    return data,row


def test_old116_registration_cannot_be_passed_as_new288():
    with pytest.raises(ValueError,match='wrong extension288'):
        collect.load_registration(ROOT/'configs/observed_two_row_formal116_registered_v1.json')


def test_prepare_is_fresh_and_never_calls_simulator(corpus,monkeypatch):
    monkeypatch.setattr(collect.physical,'run_collection',lambda *a:pytest.fail('prepare cannot simulate'))
    value,manifest=collect.verify_corpus(corpus)
    assert value['requested_routes']==7776 and manifest['internal_budget_bytes']==8*1024**3
    assert not manifest['training_authorized'] and not manifest['raw_dev_analysis_authorized']
    with pytest.raises(FileExistsError):
        collect.prepare(ROOT/'configs/observed_two_row_extension288_registered_v1.json',corpus)


@pytest.mark.parametrize('field,new', [('requested_routes',3132),('raw_dev_analysis_authorized',True),('internal_budget_bytes',9*1024**3)])
def test_manifest_budget_and_seal_cannot_change(corpus,field,new):
    path=corpus/'corpus_manifest.json';x=json.loads(path.read_text());x[field]=new
    collect.old.atomic_write(path,x)
    with pytest.raises(ValueError,match='protocol, budget'):collect.verify_corpus(corpus)


def test_registration_bytes_and_source_change_refuse_resume(corpus,monkeypatch):
    path=corpus/'registration.json';old=path.read_bytes();path.write_bytes(old+b'\n')
    with pytest.raises(ValueError,match='SHA changed'):collect.verify_corpus(corpus)
    path.write_bytes(old);original=collect.source_hashes
    monkeypatch.setattr(collect,'source_hashes',lambda p:dict(original(p),changed='source'))
    with pytest.raises(ValueError,match='source differs'):collect.verify_corpus(corpus)


def test_stage_identity_and_entire_prior_prefix_are_required(corpus):
    value,_=collect.verify_corpus(corpus)
    assert collect.stage_indices('train32')==list(range(32))
    assert collect.stage_indices('train256')==list(range(256))
    assert collect.stage_indices('dev32')==list(range(256,288))
    with pytest.raises(ValueError,match='unknown'):collect.stage_indices('train48')
    for i in range(31):save_closure(corpus,value,i)
    with pytest.raises(ValueError,match='entire prior'):collect.require_stage_prerequisites(corpus,value,'train64')
    save_closure(corpus,value,31)
    assert len(collect.require_stage_prerequisites(corpus,value,'train64'))==32
    with pytest.raises(ValueError,match='entire prior'):collect.require_stage_prerequisites(corpus,value,'dev32')


def test_dev_is_separate_stage_after_all256_not_just_own_shard(corpus,monkeypatch):
    value,_=collect.verify_corpus(corpus)
    for i in range(256):
        if i!=255:save_closure(corpus,value,i)
    monkeypatch.setattr(collect.subprocess,'run',lambda *a,**k:pytest.fail('must not issue DEV worker'))
    with pytest.raises(ValueError,match='255'):
        collect.run_shard(corpus,corpus.parent/'runs',0,Path('python'),'dev32',max_new_parents=0)
    save_closure(corpus,value,255)
    assert len(collect.require_stage_prerequisites(corpus,value,'dev32'))==256


def test_child_worker_cannot_bypass_dev_gate_or_parent_root(corpus):
    value,_=collect.verify_corpus(corpus);p=value['parent_plan'][256]
    with pytest.raises(ValueError,match='registered parent root'):
        collect.worker(corpus,256,corpus.parent/'wrong')
    with pytest.raises(ValueError,match='entire prior'):
        collect.worker(corpus,256,corpus/'parents'/p['role']/p['parent_id'])


def test_worker_reuses_physical_entry_without_changing_old_protocol(corpus,monkeypatch):
    value,manifest=collect.verify_corpus(corpus);p=value['parent_plan'][0]
    cfg=collect.parent_config(value,p,manifest)
    collect.old.atomic_write(corpus/'parent_configs/000.json',cfg)
    seen=[]
    def physical(config,config_path,output,source_loader):
        seen.append(config)
        assert config['protocol']==collect.PARENT_PROTOCOL and config['requested_setup_actions']==0
        assert config['registered_parent']['config']['preparation_xyz']==[]
        assert callable(source_loader)
        return {'fatal_error':None,'shutdown_error':None}
    monkeypatch.setattr(collect.physical,'run_collection',physical)
    result=collect.worker(corpus,0,corpus/'parents/TRAIN'/p['parent_id'])
    assert not result['fatal_error'] and len(seen)==1


def test_issued_interrupted_slots_are_closed_and_never_replayed(corpus,tmp_path,monkeypatch):
    value,_=collect.verify_corpus(corpus);p=value['parent_plan'][0]
    data=corpus/'parents/TRAIN'/p['parent_id'];data.mkdir(parents=True)
    for event,attempt in [('started',0),('completed',0),('started',1)]:
        collect.physical.legacy.append_json(data/'slot_ledger.jsonl',dict(event=event,parent_id=p['parent_id'],input_id=p['parent_id']+'_target0',attempt=attempt))
    monkeypatch.setattr(collect.subprocess,'run',lambda *a,**k:pytest.fail('interrupted parent must not replay'))
    assert collect.run_shard(corpus,tmp_path/'runs',0,Path('python'),'train32',max_new_parents=1)==1
    row=json.loads((corpus/'closures/000.json').read_text())
    assert row['attempted_lower']==row['attempted_upper']==2 and row['completed_slots']==1
    assert row['unattempted_lower']==row['unattempted_upper']==25 and row['worker_elapsed_unknown']
    assert collect.run_shard(corpus,tmp_path/'runs',0,Path('python'),'train32',resume=True,max_new_parents=0)==0


def test_pre_run_failure_closes_zero_attempts_and_stops_shard(corpus,tmp_path,monkeypatch):
    run=tmp_path/'runs';status=run/'parents/shard0/parent_000.status.json'
    collect.old.atomic_write(status,dict(status='completed',exit_code=1,start_utc='2026-10-02T01:00:00+00:00',end_utc='2026-10-02T01:00:01+00:00'))
    monkeypatch.setattr(collect.subprocess,'run',lambda *a,**k:pytest.fail('failed worker must not replay'))
    with pytest.raises(RuntimeError,match='closure preserved'):
        collect.run_shard(corpus,run,0,Path('python'),'train32',max_new_parents=1)
    row=json.loads((corpus/'closures/000.json').read_text())
    assert row['attempted_upper']==0 and row['unattempted_lower']==27 and row['worker_elapsed_seconds']==1.


def test_live_shard_lock_refuses_duplicate(corpus,tmp_path):
    with collect.old.shard_lock(corpus/'shards/0/coordinator.lock'):
        with pytest.raises(RuntimeError,match='still alive'):
            collect.run_shard(corpus,tmp_path/'runs',0,Path('python'),'train32',resume=True,max_new_parents=0)


def test_closed_hash_mutation_and_role_change_block_gate(corpus):
    value,_=collect.verify_corpus(corpus);data,row=save_closure(corpus,value,0)
    path=data/'mechanical_receipt.json';path.write_bytes(path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='evidence changed'):collect.live_layout_gate(corpus)
    row['role']='DEV_MODEL';collect.old.atomic_write(corpus/'closures/000.json',row)
    with pytest.raises(ValueError,match='index/role'):collect.live_layout_gate(corpus)


def test_new_actual_mismatch_and_cross_role_duplicates_close_all(corpus):
    value,_=collect.verify_corpus(corpus);_,a=save_closure(corpus,value,0)
    _,b=save_closure(corpus,value,256,a['actual_geometry_1mm_sha256'])
    gate=collect.live_layout_gate(corpus)
    assert gate['blocked_parent_ids']==[a['parent_id'],b['parent_id']]
    assert gate['duplicate_groups'][0]['roles']==['DEV_MODEL','TRAIN']


def test_gate_never_opens_new_dev_raw(corpus,monkeypatch):
    value,_=collect.verify_corpus(corpus);data,_=save_closure(corpus,value,256)
    (data/'front.png').write_bytes(b'sealed pixels');(data/'attempts.jsonl').write_text('sealed outcomes')
    original=Path.open
    def guarded(path,*a,**k):
        if path.name in ('front.png','attempts.jsonl'):raise AssertionError('sealed content opened')
        return original(path,*a,**k)
    monkeypatch.setattr(Path,'open',guarded)
    assert not collect.live_layout_gate(corpus)['raw_dev_or_old_reserved_opened']


def test_internal_budget_pause_prevents_new_parent_and_retains_all_data(corpus,tmp_path,monkeypatch):
    run=tmp_path/'runs'
    monkeypatch.setattr(collect,'directory_bytes',lambda p:collect.BUDGET_BYTES//2)
    monkeypatch.setattr(collect.subprocess,'run',lambda *a,**k:pytest.fail('budget pause must not simulate'))
    with pytest.raises(collect.InternalBudgetPause):
        collect.run_shard(corpus,run,0,Path('python'),'train32',max_new_parents=1)
    assert not (corpus/'closures/000.json').exists()
    assert json.loads((corpus/'budget_latest.json').read_text())['next_parent_reserve_bytes']==2*collect.NEXT_PARENT_RESERVE_BYTES


def test_budget_cannot_hide_output_in_symlink_directory(tmp_path,monkeypatch):
    link=tmp_path/'linked'
    monkeypatch.setattr(collect.os,'walk',lambda *a,**k:[(str(tmp_path),['linked'],[])])
    original=Path.is_symlink
    monkeypatch.setattr(Path,'is_symlink',lambda p:p==link or original(p))
    with pytest.raises(ValueError,match='symlink'):collect.directory_bytes(tmp_path)


def test_prepare_and_stage_cli_do_not_require_or_start_simulation(corpus,monkeypatch):
    monkeypatch.setattr(collect,'runtime_guard',lambda:pytest.fail('mechanical stage gate should not initialize simulation'))
    assert collect.main(['check-stage','--corpus',str(corpus),'--stage','train32'])==0


def test_gpu_and_single_core_runtime_guard(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    with pytest.raises(ValueError,match='hidden GPU'):collect.runtime_guard()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','')
    monkeypatch.setattr(collect.os,'sched_getaffinity',lambda pid:{2,3},raising=False)
    with pytest.raises(ValueError,match='one CPU'):collect.runtime_guard()
    monkeypatch.setattr(collect.os,'sched_getaffinity',lambda pid:{2})
    collect.runtime_guard()
