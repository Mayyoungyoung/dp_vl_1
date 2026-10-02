"""Matched one-update controls preserve historical and information contracts."""
import io
import numpy as np
import pytest
import torch

from routeset.observed_geometry import ObservedGeometryRouteHead
from routeset.observed_path_refinement import ObservedPathRefiner,validate_refinement_resume
from test_observed_geometry import inputs


def model(mode='none'):
    options={} if mode=='none' else dict(refinement_mode=mode,refinement_sigma=.1,
        refinement_prefix_fraction=.5,refinement_bound=.1)
    return ObservedGeometryRouteHead(16,horizon=8,width=16,point_width=8,
                                    anchor_mode='straight_through_peak',**options)


def test_default_and_zero_update_match_original_peak_exactly():
    torch.set_num_threads(1)
    torch.manual_seed(71);old=model()
    torch.manual_seed(71);local=model('local')
    torch.manual_seed(71);global_pool=model('global')
    assert not any(name.startswith('refiner.') for name in old.state_dict())
    assert local.active_parameter_count()==global_pool.active_parameter_count()
    for name,value in old.state_dict().items():
        assert torch.equal(value,local.state_dict()[name])
        assert torch.equal(value,global_pool.state_dict()[name])
    assert all(torch.equal(value,global_pool.state_dict()[name]) for name,value in local.state_dict().items())
    batch=inputs();reference=old(**batch)
    assert set(reference[2])=={'context','anchor_xyz','attention'}
    for candidate in (local,global_pool):
        xyz,events,details=candidate(**batch)
        assert torch.equal(xyz,reference[0]) and torch.equal(events,reference[1])
        assert torch.equal(details['draft_paths'],reference[0])
        assert torch.equal(details['attention'],reference[2]['attention'])


def test_nonzero_update_leaves_start_endpoint_events_and_late_prefix_unchanged():
    net=model('local');batch=inputs()
    torch.nn.init.constant_(net.refiner.update[-1].bias,.4)
    xyz,events,details=net(**batch)
    draft=details['draft_paths'];eligible=details['refinement_eligible']
    assert eligible.any()
    assert torch.equal(xyz[:,:,0],draft[:,:,0]) and torch.equal(xyz[:,:,-1],draft[:,:,-1])
    assert torch.equal(xyz[:,:,1:-1][~eligible],draft[:,:,1:-1][~eligible])
    assert (xyz-draft).abs().max() <= .100001
    original=net.refiner;net.refiner=None
    coarse,coarse_events,_=net(**batch);net.refiner=original
    assert torch.equal(coarse,draft) and torch.equal(events,coarse_events)


def test_observed_mask_and_point_permutation_no_labels():
    net=model('local');batch=inputs()
    torch.nn.init.normal_(net.refiner.update[-1].weight,std=.02)
    reference=net(**batch)[0]
    changed={key:value.clone() for key,value in batch.items()}
    for key in ('world_xyz','rgb','uv','depth'):
        changed[key][~changed['valid_mask']]=float('nan')
    assert torch.allclose(net(**changed)[0],reference,atol=1e-6,rtol=0)
    order=torch.tensor([6,2,1,3,0,5,4])
    permuted={key:value if key in ('features','current') else value[:,order] for key,value in batch.items()}
    assert torch.allclose(net(**permuted)[0],reference,atol=1e-6,rtol=0)
    with pytest.raises(TypeError):
        net(**batch,reference_path=reference)
    with pytest.raises(TypeError):
        net(**batch,obstacle_centers=torch.zeros(1,3))
    changed['valid_mask'][0]=False
    with pytest.raises(ValueError,match='at least one valid'):
        net(**changed)


def test_local_and_uniform_pooling_differ_after_training_but_share_parameters():
    local=ObservedPathRefiner(2,'local',.1,.5,.1,hidden_width=8)
    uniform=ObservedPathRefiner(2,'global',.1,.5,.1,hidden_width=8)
    torch.nn.init.normal_(local.update[-1].weight,std=.1);uniform.load_state_dict(local.state_dict())
    draft=torch.zeros(1,1,6,3);draft[0,0,:,0]=torch.linspace(0,1,6)
    points=torch.tensor([[[.1,0,0],[.2,0,0],[1.,0,0]]])
    features=torch.tensor([[[1.,0.],[1.,0.],[0.,1.]]]);mask=torch.ones(1,3,dtype=torch.bool)
    current=torch.zeros(1,8)
    a,details=local(draft,features,points,mask,current);b,_=uniform(draft,features,points,mask,current)
    assert not torch.equal(a,b)
    assert details['refinement_eligible'].tolist()==[[[True,True,False,False]]]
    assert torch.equal(a[:,:,3:],draft[:,:,3:])


def test_full_optimizer_rng_sampler_resume_matches_uninterrupted_stream():
    torch.set_num_threads(1);torch.manual_seed(19)
    net=model('local');batch=inputs()
    optimizer=torch.optim.AdamW(net.parameters(),lr=.001)
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda _:1.)
    sampler=np.random.default_rng(100019)
    seen=[]
    def step():
        ids=sampler.choice(2,2,replace=True);seen.append(ids.tolist())
        sub={key:value[ids] for key,value in batch.items()}
        xyz,events,_=net(**sub)
        optimizer.zero_grad(set_to_none=True)
        (xyz.square().mean()+events.square().mean()).backward()
        optimizer.step();scheduler.step()
    step();step()
    assert net.refiner.update[0].weight.grad.abs().sum()>0
    assert net.geometry.point_encoder[0].weight.grad.abs().sum()>0
    buffer=io.BytesIO()
    torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    torch_rng=torch.get_rng_state(),sampler=sampler.bit_generator.state),buffer)
    step();step();expected={name:value.clone() for name,value in net.state_dict().items()};expected_ids=seen[2:]
    buffer.seek(0);saved=torch.load(buffer,weights_only=False)
    net.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
    torch.set_rng_state(saved['torch_rng']);sampler.bit_generator.state=saved['sampler'];seen=[]
    step();step()
    assert seen==expected_ids
    assert all(torch.equal(value,net.state_dict()[name]) for name,value in expected.items())
    validate_refinement_resume({}, {'refinement_mode':'none'})
    config=dict(refinement_mode='local',refinement_sigma=.1,refinement_prefix_fraction=.5,refinement_bound=.1)
    validate_refinement_resume(config,config)
    for key,value in [('refinement_mode','global'),('refinement_sigma',.2),('refinement_prefix_fraction',.25),('refinement_bound',.2)]:
        changed=dict(config);changed[key]=value
        with pytest.raises(ValueError,match='refinement'):
            validate_refinement_resume(changed,config)


def test_explicit_prespecified_scale_required_and_exact_parameter_budget():
    with pytest.raises(ValueError,match='explicit positive'):
        ObservedPathRefiner(64,'local',None,.5,.1)
    with pytest.raises(ValueError,match='enabled refinement'):
        ObservedGeometryRouteHead(feature_dim=16,refinement_sigma=.1)
    for mode in ('local','global'):
        refiner=ObservedPathRefiner(64,mode,.1,.5,.1)
        assert sum(parameter.numel() for parameter in refiner.parameters())==4995


def test_evaluator_saves_complete_drafts_and_counts_both_path_states(tmp_path):
    from test_observed_selection import fixture,FixedObservedModel
    from scripts.train_observed_geometry import evaluate
    paths,data,points,sources=fixture(tmp_path)
    class WithDraft(FixedObservedModel):
        def __call__(self,**inputs):
            xyz,events,details=super().__call__(**inputs)
            details['draft_paths']=xyz.clone()
            return xyz,events,details
    result=evaluate(WithDraft(paths),data,points,np.arange(2),'cpu',tmp_path/'out',
                    selection_metric='tip_unique_valid',evaluation_sources=sources)
    budget=result['generation_budget']
    assert budget['final_candidates']==2 and budget['draft_complete_paths']==2
    assert budget['total_complete_path_states']==4 and budget['updates']==1
    with np.load(tmp_path/'out/predictions.npz') as archive:
        assert np.array_equal(archive['draft_paths'],archive['paths'])
