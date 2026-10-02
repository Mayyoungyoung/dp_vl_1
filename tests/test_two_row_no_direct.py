import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import train_observed_two_row_no_direct as driver

ROOT = Path(__file__).resolve().parents[1]


def policy():
    def unique(pairs):
        assert len(dict(pairs)) == len(pairs)
        return dict(pairs)
    return json.loads((ROOT/'configs/observed_two_row_no_direct_v1.json').read_text(),object_pairs_hook=unique)


def metadata(total=200):
    values={key:'fixture' for key in driver.POLICY_FIELDS}
    values.update(no_direct_protocol=driver.PROTOCOL,no_direct_model_protocol=driver.MODEL_PROTOCOL,no_direct_total_steps=total)
    return values


def test_registered_budget_and_capacity_are_explicit():
    p=driver.validate_policy(policy())
    assert p['total_steps']*p['batch_size']*p['candidates']==1536000
    assert p['total_steps']//p['eval_every']==48 and p['fixed_dev_inputs']==36
    assert p['original_parameters']-p['removed_parameters']==p['remaining_parameters']==682845
    assert not p['geometry_frozen'] and not p['endpoint_clamping']


@pytest.mark.parametrize('key,value',[('total_steps',6000),('seed',1),('lr',1e-4),('geometry_frozen',True),
    ('endpoint_clamping',True),('fixed_dev_inputs',48),('remaining_parameters',1231965),('data','new32DEV')])
def test_policy_refuses_additional_trials_information_or_budget(key,value):
    p=policy();p[key]=value
    with pytest.raises(ValueError):driver.validate_policy(p)


def checkpoint():
    return dict(config=dict(metadata(),steps=200,lr=.0003),step=100,model={'geometry.x':1},
        scheduler={'base_lrs':[.0003],'_last_lr':[.0003],'last_epoch':100},
        optimizer={'param_groups':[{'lr':.0003}]})


def test_resume_requires_new_model_protocol_same_source_and_constant_schedule():
    current=dict(metadata(),steps=200,lr=.0003);driver.validate_resume(current,checkpoint(),200)
    for key in driver.POLICY_FIELDS:
        changed=checkpoint();changed['config'][key]='changed'
        with pytest.raises(ValueError):driver.validate_resume(current,changed,200)
    for changed in (dict(checkpoint(),model={'head.feature_encoder.0.weight':1}),
                    dict(checkpoint(),scheduler={'base_lrs':[.0003],'_last_lr':[0.],'last_epoch':100})):
        with pytest.raises(ValueError):driver.validate_resume(current,changed,200)


def input_fixture(torch,feature_dim=16):
    generator=torch.Generator().manual_seed(41)
    return dict(features=torch.randn(2,feature_dim,generator=generator),
        current=torch.cat([torch.randn(2,7,generator=generator),torch.ones(2,1)],-1),
        world_xyz=torch.randn(2,11,3,generator=generator),rgb=torch.rand(2,11,3,generator=generator),
        uv=torch.rand(2,11,2,generator=generator),depth=torch.rand(2,11,generator=generator)+.1,
        valid_mask=torch.tensor([[True]*10+[False],[True]*11]))


def test_model_shared_initialization_rng_old_forward_and_information_boundary():
    torch=pytest.importorskip('torch')
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from routeset.observed_geometry_no_direct import ObservedGeometryNoDirectHead,REMOVED_PREFIX
    options=dict(feature_dim=16,horizon=6,max_candidates=4,width=16,depth=1,point_width=8,anchor_mode='straight_through_peak')
    torch.manual_seed(0);old=ObservedGeometryRouteHead(**options);old_rng=torch.get_rng_state().clone()
    torch.manual_seed(0);new=ObservedGeometryNoDirectHead(**options)
    assert torch.equal(old_rng,torch.get_rng_state())
    assert set(new.state_dict())=={k for k in old.state_dict() if not k.startswith(REMOVED_PREFIX)}
    for key,value in new.state_dict().items():assert torch.equal(value,old.state_dict()[key])
    inputs=input_fixture(torch)
    a=old(**inputs);b=new(**inputs)
    assert not torch.equal(a[0],b[0])
    for key in ('anchor_xyz','context','attention'):assert torch.equal(a[2][key],b[2][key])
    assert torch.equal(b[0][:,:,0],inputs['current'][:,None,:3].expand(-1,4,-1))
    assert torch.all((b[0][:,:,-1]-b[2]['anchor_xyz'][:,None]).abs()<=.050001)
    assert not torch.equal(b[0][:,:,-1],a[0][:,:,-1]) # no normal endpoint clamp
    assert not list(new.head.feature_encoder.parameters())
    # Direct summand ignores even nonfinite feature values; geometry still uses language.
    assert torch.equal(new.head.feature_encoder(inputs['features']),new.head.feature_encoder(inputs['features']*float('nan')))
    with pytest.raises(TypeError):new(**inputs,semantic_targets=torch.zeros(2,3))
    old_again=old(**inputs)
    assert torch.equal(a[0],old_again[0]) and torch.equal(a[1],old_again[1])
    assert ObservedGeometryRouteHead.forward is ObservedGeometryNoDirectHead.forward


def test_actual_production_constructor_matches_frozen_old_initialization_and_count():
    torch=pytest.importorskip('torch')
    from routeset.observed_geometry_no_direct import ObservedGeometryNoDirectHead
    from routeset.train_v2 import seed_all
    seed_all(0);model=ObservedGeometryNoDirectHead(4096,anchor_mode='straight_through_peak')
    proof=model.initialization_receipt;p=policy();stream=p['expected_sample_stream_audit']
    assert proof['original_model_sha256']==stream['initial_model_sha256']
    assert proof['post_construction_torch_rng_sha256']==stream['initial_torch_cpu_rng_sha256']
    assert (proof['original_parameters'],proof['removed_parameters'],proof['remaining_parameters'])==(1231965,549120,682845)


def test_all_remaining_modules_receive_gradient_and_actual_updates():
    torch=pytest.importorskip('torch')
    from routeset.observed_geometry_no_direct import ObservedGeometryNoDirectHead
    from scripts.train_observed_geometry import positive_endpoint_attention_loss
    torch.manual_seed(3)
    model=ObservedGeometryNoDirectHead(16,horizon=6,width=16,depth=1,point_width=8,anchor_mode='straight_through_peak')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=1e-4)
    before={k:v.clone() for k,v in model.state_dict().items()};inputs=input_fixture(torch)
    xyz,opened,details=model(**inputs)
    target=inputs['world_xyz'][:,2:3]
    loss=xyz.square().mean()+opened.square().mean()+.02*positive_endpoint_attention_loss(
        details['attention'],inputs['world_xyz'],inputs['valid_mask'],target,torch.ones(2,1,dtype=torch.bool),.025)
    loss.backward()
    for name,p in model.named_parameters():
        assert p.requires_grad and p.grad is not None and bool(torch.isfinite(p.grad).all()),name
    optimizer.step()
    for prefix in ('geometry.task_query','geometry.point_encoder','geometry.fusion','head.state_encoder','head.queries','head.blocks','head.output'):
        assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items() if k.startswith(prefix)),prefix


def test_scoped_factory_restores_old_entry_on_exception():
    pytest.importorskip('torch')
    from routeset.observed_geometry import ObservedGeometryRouteHead
    def fail(args):raise RuntimeError('interrupted')
    audit=lambda *args:{}
    ordinary=SimpleNamespace(base=SimpleNamespace(train=fail,ObservedGeometryRouteHead=ObservedGeometryRouteHead,new_stream_audit=audit))
    with pytest.raises(RuntimeError):
        with driver.model_adapter(ordinary,metadata(),{},total_steps=200):
            ordinary.base.train(SimpleNamespace(steps=1500,output='unused',resume=False))
    assert ordinary.base.train is fail and ordinary.base.ObservedGeometryRouteHead is ObservedGeometryRouteHead
    assert ordinary.base.new_stream_audit is audit


def test_actual_unchanged_loop_complete_resume_and_index_chain_are_exact(tmp_path,monkeypatch):
    torch=pytest.importorskip('torch')
    from scripts import train_observed_two_row as ordinary
    from test_two_row_observation_training import fixture
    data,geometry,_,sources,labels=fixture(tmp_path,monkeypatch)
    ids=np.array(['t_target0','t_target1','t_target2'])
    data={key:(np.concatenate([value,value]) if isinstance(value,np.ndarray) else value*2) for key,value in data.items()}
    data['scene_ids'][:3]=ids;data['parent_ids'][:3]='t';data['splits']=np.array(['TRAIN']*3+['DEV_MODEL']*3)
    data.update(tasks=['reach']*6,source_hashes={},cache_config={},skipped=[],unreferenced=[],evaluation_protocol='observation_eval_v2')
    train_labels=[dict(row,id=str(ids[i]),parent_id='t',split='TRAIN') for i,row in enumerate(labels)]
    sources[1].write_text(''.join(json.dumps(r)+'\n' for r in train_labels+labels))
    geometry.update(index=np.zeros(6,dtype=int),fingerprint='fixture',metadata={'source_hashes':{}})
    monkeypatch.setattr(ordinary.base,'load_observed_dataset',lambda *a,**kw:data)
    monkeypatch.setattr(ordinary.base,'load_geometry',lambda *a,**kw:geometry)
    monkeypatch.setattr(ordinary.base,'measure_latency',lambda *a,**kw:{})
    values=dict(observations=str(sources[0]),supervision=str(sources[1]),cache_dir='fixture',
        steps=1500,batch_size=2,candidates=2,horizon=2,width=16,depth=1,pooling='both',geometry_pooling='spatial',
        anchor_mode='straight_through_peak',endpoint_mode='surface_anchor',sampling_mode='uniform',
        metric_aggregation='instruction',refinement_mode='none',refinement_sigma=None,refinement_prefix_fraction=None,
        refinement_bound=None,checkpoint_selection='tip_unique_valid',point_width=8,pixel_stride=2,
        endpoint_residual_bound=.05,grounding_weight=.02,grounding_sigma=.025,grounding_target='endpoint',
        event_scale=.2,lr=3e-4,seed=0,eval_every=100,threads=1,device='cpu',sample_stream_audit=True)
    values.update({key:'fixture' for key in ('two_row_driver_sha256','two_row_export_sha256','two_row_quality_audit_sha256','two_row_selection_sha256')})
    def run(name,resume=False,stop=None,no_direct=True):
        args=SimpleNamespace(**dict(values,output=str(tmp_path/name),resume=resume,stop_after=stop))
        with ordinary.evaluation_adapter(sources):
            if no_direct:
                with driver.model_adapter(ordinary,metadata(),{},total_steps=200):ordinary.base.train(args)
            else:
                args.steps=200;ordinary.base.train(args)
        return torch.load(tmp_path/name/'last.pt',weights_only=False)
    full=run('full');run('split',stop=100);resumed=run('split',resume=True)
    for key in driver.shared.STATE_FIELDS:
        assert not driver.shared.state_differences(full[key],resumed[key],key),key
    old=run('old',no_direct=False)
    for key in full['sample_stream_audit']:
        if key!='initial_model_sha256':assert full['sample_stream_audit'][key]==old['sample_stream_audit'][key],key
    assert not driver.shared.state_differences(full['rng'],old['rng'])
    gradients=json.loads((tmp_path/'full/first_backward_after_step0.json').read_text())
    assert gradients['all_remaining_parameters_received_gradient'] and gradients['all_finite']
    bad=copy.deepcopy(resumed);bad['config']['no_direct_model_protocol']='old_model'
    torch.save(bad,tmp_path/'split/last.pt')
    with pytest.raises(ValueError,match='resume protocol'):run('split',resume=True)
