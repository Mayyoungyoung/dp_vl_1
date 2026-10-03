"""Protocol, crash accounting, and actual serial accumulation/resume checks."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from routeset.observed_qwen_continuation import (POLICY, RequestJournal, SerialTrainer,
    atomic_torch_save, check_config, composite_evaluator, digest, draw_plan,
    exclusive_lock, reconcile_sealed_pool, seal_draw_plan, validate_policy,
    recovered_elapsed, should_pause, verify_pool, write_json)


def test_production_policy_and_exposure():
    p=json.loads(Path('configs/observed_two_row_lora_continuation_v1.json').read_text())
    assert validate_policy(p)==POLICY
    assert p['steps']*p['accumulation']==96000
    assert 96000*p['candidates']==384000
    assert p['steps']//p['eval_every']==12
    assert 12*36+285==717


@pytest.mark.parametrize('field,value',[('steps',3001),('accumulation',4),('lora_lr',1e-4),
    ('head_sha256','0'*64),('eval_every',500),('new_dev_allowed',True)])
def test_production_policy_rejects_changes(field,value):
    changed=dict(POLICY);changed[field]=value
    with pytest.raises(ValueError):validate_policy(changed)


def test_draw_plan_deterministic_and_sealed(tmp_path):
    a=draw_plan(['a','b','c'],4,32);b=draw_plan(['a','b','c'],4,32)
    assert a==b and len(a['indices'])==4 and len(a['indices'][0])==32
    path=tmp_path/'shared.json';assert seal_draw_plan(path,a)==seal_draw_plan(path,b)
    with pytest.raises(ValueError):seal_draw_plan(path,draw_plan(['b','a','c'],4,32))


def test_journal_crash_issued_call_is_not_replayed(tmp_path):
    p=tmp_path/'requests.jsonl';j=RequestJournal(p);boundary=j.snapshot()
    j.issue('train_tail','1:0:a')
    recovered=RequestJournal(p)
    assert recovered.counts=={'train_tail':1}
    with pytest.raises(ValueError):recovered.require_boundary(boundary)
    with pytest.raises(ValueError):recovered.issue('train_tail','1:0:a')
    with p.open('ab') as f:f.write(b'{"incomplete":')
    with pytest.raises(ValueError):RequestJournal(p)


def test_journal_tamper_rejected(tmp_path):
    p=tmp_path/'requests.jsonl';j=RequestJournal(p);j.issue('optimizer','1')
    p.write_text(p.read_text().replace('optimizer','something'))
    with pytest.raises(ValueError):RequestJournal(p)


def make_pool(path,identity,before,after):
    path.mkdir()
    for name in ['tail_features.npz','predictions.npz','per_scene.json']:(path/name).write_bytes(b'fixture')
    write_json(path/'metrics.json',{'selection_score':.1})
    receipt=dict(identity=identity,journal_before=before,journal_after=after,
        metrics={'selection_score':.1},artifacts={p.name:digest(p) for p in path.iterdir()})
    write_json(path/'pool_receipt.json',receipt)
    return receipt


def test_complete_pool_can_reconcile_without_forward(tmp_path):
    j=RequestJournal(tmp_path/'requests.jsonl');before=j.snapshot()
    ids=['a','b','c'];identity=dict(step=250,ids=ids,requests=3)
    for kind in ['dev_tail','dev_head']:
        for identifier in ids:j.issue(kind,'250:'+identifier)
    pool=tmp_path/'pool';receipt=make_pool(pool,identity,before,j.snapshot())
    assert verify_pool(pool,identity)==receipt
    reconcile_sealed_pool(j,before,receipt)
    assert len(j.records)==6
    j.issue('train_tail','251:0:a')
    with pytest.raises(ValueError):reconcile_sealed_pool(j,before,receipt)


def test_incomplete_or_changed_pool_rejected(tmp_path):
    j=RequestJournal(tmp_path/'j');identity=dict(step=250,ids=['a'],requests=1)
    receipt=make_pool(tmp_path/'pool',identity,j.snapshot(),j.snapshot())
    receipt['artifacts']={};write_json(tmp_path/'pool/pool_receipt.json',receipt)
    with pytest.raises(ValueError):verify_pool(tmp_path/'pool',identity)


def test_extra_or_wrong_eval_span_rejected(tmp_path):
    j=RequestJournal(tmp_path/'j');before=j.snapshot()
    j.issue('dev_tail','250:a');j.issue('optimizer','250')
    receipt=dict(identity=dict(step=250,ids=['a'],requests=1),journal_before=before,journal_after=j.snapshot())
    with pytest.raises(ValueError):reconcile_sealed_pool(j,before,receipt)


def test_sealed_evaluation_recovery_cost_is_not_free():
    assert recovered_elapsed(100.,3.5)==103.5
    with pytest.raises(ValueError):recovered_elapsed(100.,float('nan'))
    with pytest.raises(ValueError):recovered_elapsed(100.,-1.)


@pytest.mark.parametrize('boundary',[2,25])
def test_administrative_pause_does_not_change_budget(boundary):
    assert not should_pause(boundary-1,boundary,3000)
    assert should_pause(boundary,boundary,3000)
    assert not should_pause(boundary,None,3000)
    assert not should_pause(3000,3000,3000)
    with pytest.raises(ValueError):should_pause(boundary+1,boundary,3000)
    assert 'stop_after' not in POLICY


def test_contexts_restore_on_exception(tmp_path):
    old=lambda x:x;new=lambda x:None;ordinary=SimpleNamespace(verify_export=old)
    with pytest.raises(RuntimeError):
        with composite_evaluator(ordinary,new):
            assert ordinary.verify_export is new
            raise RuntimeError('checker failed')
    assert ordinary.verify_export is old
    with pytest.raises(RuntimeError):
        with exclusive_lock(tmp_path/'lock'):
            with pytest.raises(FileExistsError):
                with exclusive_lock(tmp_path/'lock'):pass
            raise RuntimeError('body failed')
    assert not (tmp_path/'lock').exists()


@pytest.mark.parametrize('change',[{'arm':'lora'},{'lora_lr':1e-4},{'prefix_corpus_sha256':'other'},
    {'draw_plan_sha256':'other'},{'source_sha256':{'helper':'changed'}},{'eval_every':500}])
def test_resume_config_rejects_protocol_changes(change):
    config=dict(POLICY,arm='frozen',prefix_corpus_sha256='a',draw_plan_sha256='b',source_sha256={'helper':'c'})
    with pytest.raises(ValueError):check_config(dict(config,**change),config)


def tiny(tmp_path,arm):
    torch=pytest.importorskip('torch');torch.set_num_threads(1)
    from routeset.common import seed_all
    seed_all(11)
    head=torch.nn.Linear(3,4)
    adapters={'tail.lora_A':torch.nn.Parameter(torch.randn(3,2)*.1),
              'tail.lora_B':torch.nn.Parameter(torch.zeros(2,3))} if arm=='lora' else {}
    cfg=dict(POLICY,steps=4,seed=11,arm=arm)
    plan=draw_plan(['a','b','c'],4,32,100000)
    trainer=SerialTrainer(head,adapters,cfg,plan);journal=RequestJournal(tmp_path/'journal.jsonl')
    def loss(identifier,step,micro,rng):
        key='%d:%d:%s'%(step,micro,identifier)
        journal.issue('train_tail',key)
        x=torch.randn(1,3)+float(rng.normal())
        if adapters:x=x+x@adapters['tail.lora_A']@adapters['tail.lora_B']
        journal.issue('train_head',key)
        return (head(x)-ord(identifier)/100).square().mean()
    return torch,trainer,journal,loss


def normalize_numpy(value):
    # The shared tensor comparator handles Torch tensors, not NumPy's global
    # RNG tuple. Preserve all NumPy RNG values while making equality explicit.
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:normalize_numpy(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(normalize_numpy(v) for v in value)
    return value


@pytest.mark.parametrize('arm',['frozen','lora'])
def test_torch_B1_accum32_continuous4_vs2_restore2_exact(tmp_path,arm):
    from routeset.qwen_prefix_replay import differences
    torch,a,ja,fa=tiny(tmp_path/'continuous',arm)
    for _ in range(4):a.train_step(fa,ja)
    expected=a.state_dict(ja,0.)
    torch,b,jb,fb=tiny(tmp_path/'resumed',arm)
    while not should_pause(b.step,2,4):b.train_step(fb,jb)
    saved=b.state_dict(jb,0.);p=tmp_path/'boundary.pt';atomic_torch_save(p,saved)
    torch,c,jc,fc=tiny(tmp_path/'resumed',arm)
    c.load_state_dict(torch.load(p,map_location='cpu',weights_only=False),jc)
    for _ in range(2):c.train_step(fc,jc)
    actual=c.state_dict(jc,0.)
    assert not differences(normalize_numpy(expected),normalize_numpy(actual))
    assert jc.counts=={'train_tail':128,'train_head':128,'optimizer':4}
    assert actual['candidate_path_states']==512 and actual['draw_position']==128


def test_torch_fault_after_optimizer_or_checkpoint_rename_no_replay(tmp_path,monkeypatch):
    torch,t,j,loss=tiny(tmp_path/'run','lora')
    before=t.state_dict(j,0.);p=tmp_path/'last.pt';atomic_torch_save(p,before)
    t.train_step(loss,j)
    with pytest.raises(ValueError):t.load_state_dict(before,j)
    original=p.read_bytes()
    import routeset.observed_qwen_continuation as core
    monkeypatch.setattr(core.os,'replace',lambda *a:(_ for _ in ()).throw(OSError('rename fault')))
    with pytest.raises(OSError):atomic_torch_save(p,t.state_dict(j,0.))
    assert p.read_bytes()==original
    with pytest.raises(ValueError):t.load_state_dict(torch.load(p,weights_only=False),j)
