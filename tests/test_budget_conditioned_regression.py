import copy
import ctypes
import json
from pathlib import Path
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from scripts import train_budget_conditioned_regression as driver


ROOT = Path(__file__).resolve().parents[1]


def registered():
    return json.loads((ROOT/'configs/budget_conditioned_regression_v1.json').read_text())


def test_actual_four_budget_cycle_and_total():
    config = driver.validate_config(registered())
    assert [driver.k_for_step(i) for i in range(1, 9)] == [1, 2, 4, 8]*2
    counts = driver.budget_counts(3000, 64)
    assert counts == dict(requests=192000, slots=720000, requests_by_k={str(k):48000 for k in driver.BUDGETS})
    assert 6*128*sum(driver.BUDGETS) == config['selection_output_slots'] == 11520
    assert (2+10)*sum(driver.BUDGETS)*2 == 360
    assert 720000+11520+360 == config['maximum_total_output_slots'] == 731880
    assert driver.budget_counts(3, 64)['slots'] == 448


@pytest.mark.parametrize('key,value', [('steps', 4000), ('batch_size',32), ('budgets',[1,2,4]), ('lr',1e-3), ('objective','positive'), ('eval_every',250), ('dataset_sha256','bad')])
def test_policy_changes_rejected(key, value):
    config = registered(); config[key] = value
    with pytest.raises(ValueError, match='registered config mismatch'):
        driver.validate_config(config)


def test_selection_all_k_equal_weight_not_k8_scale():
    metrics = {str(k):dict(unique_valid=k/2, valid_rate=1.) for k in driver.BUDGETS}
    assert driver.selection_score(metrics) == pytest.approx(.55)
    metrics.pop('8')
    with pytest.raises(ValueError):
        driver.selection_score(metrics)


def test_parent_and_k_stream_is_exact_and_order_sensitive():
    s = driver.new_stream()
    for step in range(1, 5):
        s = driver.advance_stream(s, step, driver.k_for_step(step), [3, 1])
    assert s['slots'] == 30 and s['requests'] == 8
    alternate = driver.new_stream()
    for step in range(1, 5):
        alternate = driver.advance_stream(alternate, step, driver.k_for_step(step), [1, 3])
    assert alternate['chain'] != s['chain']
    with pytest.raises(ValueError):
        driver.advance_stream(s, 5, 8, [1])


def test_reserved_archive_rejected_before_missing_payload_access(tmp_path):
    p = tmp_path/'data.npz'
    np.savez(p, splits=['TRAIN','TEST_LOCKED'], parent_ids=['t','s'], scene_ids=['t0','s0'])
    with pytest.raises(ValueError, match='physically separated'):
        driver.load_development(p, driver.digest(p))


def test_ledger_refuses_uncheckpointed_issued_call(tmp_path):
    ledger = driver.Ledger(tmp_path/'calls.jsonl')
    ledger.add('train_issued', step=1, k=1, slots=2, requests=2)
    ledger.add('train_completed', step=1, k=1, slots=2, requests=2)
    checkpoint = dict(ledger_events=len(ledger.events), ledger_sha256=ledger.sha())
    driver.Ledger(ledger.path).verify_checkpoint(checkpoint)
    ledger.add('selection_issued', step=1, k=1, slots=128, requests=128)
    with pytest.raises(ValueError, match='no automatic replay'):
        driver.Ledger(ledger.path).verify_checkpoint(checkpoint)


def test_sealed_pool_reuse_and_tampering(tmp_path):
    (tmp_path/'predictions.npz').write_bytes(b'fixed')
    driver.atomic_json(tmp_path/'receipt.json',dict(files_sha256={'predictions.npz':driver.digest(tmp_path/'predictions.npz')}, step=500))
    assert driver.sealed(tmp_path)['step'] == 500
    (tmp_path/'predictions.npz').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        driver.sealed(tmp_path)


def test_unsealed_pool_rejected(tmp_path):
    with pytest.raises(ValueError, match='refuse extra forward'):
        driver.sealed(tmp_path)


def test_numpy_metric_json_is_explicitly_supported(tmp_path):
    driver.atomic_json(tmp_path/'metrics.json',dict(valid=np.array([True,False]), count=np.int64(2)))
    assert json.loads((tmp_path/'metrics.json').read_text()) == dict(valid=[True,False], count=2)


def test_actual_cuda_uuid_and_allocator_cap(monkeypatch):
    cfg=registered();calls=[]
    monkeypatch.setattr(driver,'cuda_driver_identity',lambda config:dict(actual_uuid=config['gpu_uuid']))
    # Torch 2.4.1 actual properties have no uuid attribute: this must still work.
    props=SimpleNamespace(name='test GPU',total_memory=24*2**30)
    cuda=SimpleNamespace(device_count=lambda:1,get_device_properties=lambda i:props,
                         set_per_process_memory_fraction=lambda f,device:calls.append((f,device)),reset_peak_memory_stats=lambda i:None)
    result=driver.verify_cuda_device(SimpleNamespace(cuda=cuda),cfg)
    assert result['actual_uuid']==cfg['gpu_uuid'] and calls==[(.35,0)]


def fake_cuda_library(actual_uuid, count=1, failure=None):
    calls=[]
    class Function:
        def __init__(self,name):self.name=name
        def __call__(self,*args):
            calls.append(self.name)
            if self.name==failure:return 100
            if self.name=='cuInit':assert args==(0,)
            elif self.name=='cuDeviceGetCount':args[0]._obj.value=count
            elif self.name=='cuDeviceGet':
                assert args[1]==0;args[0]._obj.value=7
            elif self.name=='cuDeviceGetUuid':
                assert args[1]==7
                args[0]._obj.bytes[:]=uuid.UUID(actual_uuid[4:]).bytes
            return 0
    library=SimpleNamespace(**{n:Function(n) for n in ('cuInit','cuDeviceGetCount','cuDeviceGet','cuDeviceGetUuid')})
    return library,calls


def authorize_uuid(monkeypatch):
    cfg=registered()
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES',cfg['gpu_uuid']);monkeypatch.setenv('RESEARCH_GPU_UUID',cfg['gpu_uuid'])
    return cfg


def test_cuda_driver_public_abi_and_exact_raw_uuid(monkeypatch):
    cfg=authorize_uuid(monkeypatch);lib,calls=fake_cuda_library(cfg['gpu_uuid'])
    result=driver.cuda_driver_identity(cfg,library=lib)
    assert result['actual_uuid']==cfg['gpu_uuid']
    assert calls==['cuInit','cuDeviceGetCount','cuDeviceGet','cuDeviceGetUuid']
    assert ctypes.sizeof(driver._CUuuid)==16
    assert lib.cuInit.argtypes==[ctypes.c_uint] and lib.cuInit.restype==ctypes.c_int
    assert lib.cuDeviceGetUuid.argtypes==[ctypes.POINTER(driver._CUuuid),ctypes.c_int]
    assert result['explicit_context_create_calls']==result['model_forwards']==result['kernel_launch_calls']==0


def test_cuda_driver_actual_wrong_uuid_rejected_despite_correct_env(monkeypatch):
    cfg=authorize_uuid(monkeypatch);lib,_=fake_cuda_library('GPU-00000000-0000-0000-0000-000000000001')
    with pytest.raises(ValueError,match='actual CUDA Driver UUID mismatch'):
        driver.cuda_driver_identity(cfg,library=lib)


@pytest.mark.parametrize('failed_api',['cuInit','cuDeviceGetCount','cuDeviceGet','cuDeviceGetUuid'])
def test_cuda_driver_error_codes_fail_closed(monkeypatch,failed_api):
    cfg=authorize_uuid(monkeypatch);lib,calls=fake_cuda_library(cfg['gpu_uuid'],failure=failed_api)
    with pytest.raises(RuntimeError,match=failed_api):
        driver.cuda_driver_identity(cfg,library=lib)
    assert calls[-1]==failed_api


def test_cuda_ordinal_env_rejected_before_driver_init(monkeypatch):
    cfg=authorize_uuid(monkeypatch);lib,calls=fake_cuda_library(cfg['gpu_uuid'])
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    with pytest.raises(ValueError,match='full authorized GPU UUID'):
        driver.cuda_driver_identity(cfg,library=lib)
    assert not calls


def test_cuda_multiple_visible_devices_reject_before_handle_or_uuid(monkeypatch):
    cfg=authorize_uuid(monkeypatch);lib,calls=fake_cuda_library(cfg['gpu_uuid'],count=2)
    with pytest.raises(ValueError,match='exactly one'):
        driver.cuda_driver_identity(cfg,library=lib)
    assert calls==['cuInit','cuDeviceGetCount']


def test_zero_model_metadata_probe_records_success_and_failure(tmp_path,monkeypatch):
    config=ROOT/'configs/budget_conditioned_regression_v1.json'
    monkeypatch.setattr(driver,'cuda_driver_identity',lambda c:dict(actual_uuid=c['gpu_uuid']))
    def forbidden(*args,**kwargs):raise AssertionError('metadata must not read dataset or call training')
    monkeypatch.setattr(driver,'run',forbidden);monkeypatch.setattr(driver,'load_development',forbidden)
    result=driver.probe_cuda_metadata(config,tmp_path/'pass.json')
    assert result['status']=='completed' and result['model_forwards']==result['dataset_payload_reads']==0
    assert result['configured_memory_fraction_applied'] is False
    def failed(*args,**kwargs):raise RuntimeError('CUDA Driver cuInit returned CUresult 100')
    monkeypatch.setattr(driver,'cuda_driver_identity',failed)
    with pytest.raises(RuntimeError):driver.probe_cuda_metadata(config,tmp_path/'failed.json')
    saved=json.loads((tmp_path/'failed.json').read_text())
    assert saved['status']=='failed' and saved['exit_code']==1 and saved['model_forwards']==0


def torch_modules():
    torch = pytest.importorskip('torch')
    from routeset.budget_conditioned_regression import BudgetConditionedRegressor
    return torch, BudgetConditionedRegressor


def test_torch_only_k_tokens_and_explicit_condition():
    torch, Model = torch_modules()
    torch.manual_seed(0)
    model = Model(width=16, depth=1, heads=4)
    geometry = torch.randn(2,34)
    seen = []
    hook = model.blocks[0].register_forward_pre_hook(lambda module,args:seen.append(tuple(args[0].shape)))
    conditions=[]
    chook=model.condition_encoder.register_forward_pre_hook(lambda module,args:conditions.append(args[0].detach().clone()))
    for k in driver.BUDGETS:
        assert model(geometry,k).shape == (2,k,22,3)
    hook.remove(); chook.remove()
    assert seen == [(2,k,16) for k in driver.BUDGETS]
    assert [float(c[0,-1]) for c in conditions] == pytest.approx([0,1/3,2/3,1])
    for c in conditions:
        assert torch.equal(c[:,:34],geometry)
    with pytest.raises(ValueError):
        model(geometry,3)


def tiny_setup():
    torch, Model = torch_modules()
    torch.set_num_threads(1); torch.manual_seed(0)
    model=Model(width=16,depth=1,heads=4)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    rng=np.random.default_rng(5)
    data=dict(scenes=rng.normal(size=(5,34)).astype(np.float32), paths=rng.normal(size=(5,4,24,3)).astype(np.float32),
              path_mask=np.ones((5,4),dtype=bool))
    state=dict(step=0,stream=driver.new_stream(),reference_pool_access=0,loss_window=[])
    return torch,model,optimizer,scheduler,data,state,np.random.default_rng(0),np.random.default_rng(100000)


def assert_nested_equal(left,right):
    if isinstance(left,dict):
        assert set(left)==set(right)
        for k in left:assert_nested_equal(left[k],right[k])
    elif isinstance(left,(list,tuple)):
        assert len(left)==len(right)
        for x,y in zip(left,right):assert_nested_equal(x,y)
    elif isinstance(left,np.ndarray):
        np.testing.assert_array_equal(left,right)
    elif hasattr(left,'detach'):
        assert bool((left==right).all())
    else:assert left==right


def test_torch_shared_step_resume_exact_all_optim_rng_stream(tmp_path):
    from routeset.common import seed_all
    torch=pytest.importorskip('torch')
    from routeset.train_v2 import rng_state,restore_rng
    cfg=dict(batch_size=2,gradient_clip_norm=1.)
    seed_all(0)
    t,m,o,s,d,state,rng,sampler=tiny_setup()
    initial=driver.state_hash(m.state_dict())
    for _ in range(8):driver.model_step(m,o,s,d,np.arange(5),cfg,state,rng,sampler,'cpu')
    full=dict(model=copy.deepcopy(m.state_dict()),optimizer=copy.deepcopy(o.state_dict()),scheduler=s.state_dict(),state=copy.deepcopy(state),rng=rng_state(rng),sampler=copy.deepcopy(sampler.bit_generator.state))
    seed_all(0)
    t,m,o,s,d,state,rng,sampler=tiny_setup()
    assert driver.state_hash(m.state_dict())==initial
    for _ in range(3):driver.model_step(m,o,s,d,np.arange(5),cfg,state,rng,sampler,'cpu')
    saved=dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),state=state,rng=rng_state(rng),sampler=sampler.bit_generator.state)
    torch.save(saved,tmp_path/'resume.pt')
    t,m,o,s,d,_,rng,sampler=tiny_setup()
    ck=torch.load(tmp_path/'resume.pt',map_location='cpu',weights_only=False)
    m.load_state_dict(ck['model']);o.load_state_dict(ck['optimizer']);s.load_state_dict(ck['scheduler']);state=ck['state']
    restore_rng(ck['rng'],rng);sampler.bit_generator.state=ck['sampler']
    for _ in range(5):driver.model_step(m,o,s,d,np.arange(5),cfg,state,rng,sampler,'cpu')
    actual=dict(model=m.state_dict(),optimizer=o.state_dict(),scheduler=s.state_dict(),state=state,rng=rng_state(rng),sampler=sampler.bit_generator.state)
    assert_nested_equal(full,actual)
    assert state['stream']['slots']==60 and state['stream']['requests']==16


def test_torch_every_query_real_gradient_at_k8():
    torch,Model=torch_modules()
    torch.manual_seed(0)
    model=Model(width=16,depth=1,heads=4)
    result=model(torch.randn(2,34),8)
    result.square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert (model.queries.grad.abs().sum(-1)>0).all()


def test_torch_complete_driver_pause_resume_final_pool_reuse(tmp_path,monkeypatch):
    torch,Model=torch_modules()
    from routeset.multigate import _sample_scene,route_from_openings,unpack_scene
    import routeset.budget_conditioned_regression as model_module
    cfg=registered()
    cfg.update(steps=8,batch_size=2,width=16,depth=1,eval_every=4,checkpoint_every=2,
               train_requests=16,train_output_slots=60,selection_requests=16,selection_output_slots=60,
               selection_opportunities=2,latency_warmups=0,latency_timed=1,maximum_timing_output_slots=30,
               maximum_total_output_slots=150)
    config_path=tmp_path/'config.json';driver.atomic_json(config_path,cfg)
    scene=_sample_scene(np.random.default_rng(11)).astype(np.float32)
    _,_,walls=unpack_scene(scene)
    modes=[4*a+b for a in np.flatnonzero(walls[0,10:14]>.5) for b in np.flatnonzero(walls[1,10:14]>.5)]
    refs=np.zeros((16,24,3),dtype=np.float32);mask=np.zeros(16,dtype=bool);labels=np.full(16,-1)
    for i,mode in enumerate(modes):refs[i]=route_from_openings(scene,mode//4,mode%4,24);mask[i]=True;labels[i]=mode
    data=dict(scenes=np.repeat(scene[None],5,axis=0),paths=np.repeat(refs[None],5,axis=0),
              path_mask=np.repeat(mask[None],5,axis=0),modes=np.repeat(labels[None],5,axis=0),
              splits=np.array(['TRAIN']*3+['DEV_MODEL']*2),parent_ids=np.array(['p%d'%i for i in range(5)]),
              scene_ids=np.array(['s%d'%i for i in range(5)]))
    # Only test fixture identity and shortened budget are injected; same run loop.
    monkeypatch.setattr(driver,'validate_config',lambda c:c)
    monkeypatch.setattr(driver,'load_development',lambda p:data)
    monkeypatch.setenv('CODE_COMMIT','a'*40)
    full=driver.run(config_path,tmp_path/'full','cpu')
    driver.run(config_path,tmp_path/'resumed','cpu',stop_after=3)
    part=driver.run(config_path,tmp_path/'resumed','cpu',resume=True)
    for name in ('model','optimizer','scheduler','rng','sampler_state'):
        left=torch.load(tmp_path/'full/last.pt',map_location='cpu',weights_only=False)[name]
        right=torch.load(tmp_path/'resumed/last.pt',map_location='cpu',weights_only=False)[name]
        assert_nested_equal(left,right)
    assert full['stream']==part['stream']
    assert full['actual_budget']['train_issued']['slots']==60
    assert full['actual_budget']['selection_issued']['slots']==60
    assert full['actual_budget']['latency_issued']['slots'] in (15,30)
    for label in ('best','last'):
        for k in driver.BUDGETS:
            l=full['final'][label]['pools'][str(k)];r=part['final'][label]['pools'][str(k)]
            np.testing.assert_array_equal(np.load(tmp_path/'full'/l['path']/'predictions.npz')['paths'],
                                          np.load(tmp_path/'resumed'/r['path']/'predictions.npz')['paths'])
    before=driver.digest(tmp_path/'resumed/calls.jsonl')
    def forbidden(*args,**kwargs):raise AssertionError('completed reuse must not construct a model')
    monkeypatch.setattr(model_module,'BudgetConditionedRegressor',forbidden)
    again=driver.run(config_path,tmp_path/'resumed','cpu',resume=True)
    assert driver.digest(tmp_path/'resumed/calls.jsonl')==before
    assert again['stream']==part['stream']
