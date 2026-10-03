"""Synthetic math/input tests only: no model, dataset, checkpoint or renderer."""
import copy
import itertools
import json
from pathlib import Path
import numpy as np
import pytest
from routeset import observed_ordered_relation_loss as loss


def torch_module():
    torch=pytest.importorskip('torch');torch.set_num_threads(1);return torch


def enumerated_soft_dtw(cost,gamma):
    n,m=cost.shape;totals=[]
    def walk(i,j,total):
        total+=cost[i,j]
        if (i,j)==(n-1,m-1):totals.append(total);return
        for di,dj in ((1,0),(0,1),(1,1)):
            if i+di<n and j+dj<m:walk(i+di,j+dj,total)
    walk(0,0,0.)
    v=-np.asarray(totals)/gamma;maximum=v.max()
    return -gamma*(maximum+np.log(np.exp(v-maximum).sum()))


def test_single_setting_and_original_four_coordinate_weighting():
    p=loss.load_policy()
    assert p['gamma']==.1 and p['xyz_scale_m']==.1
    assert p['xyz_weight']==3/4 and p['event_weight']==1/4 and p['event_scale']==.2
    assert p['extra_generation_calls']==0 and p['generation_model_changed'] is False
    assert p['proposed_training_target_slots_per_arm']==3000*32*4
    m=p['microbenchmark']
    assert (m['batch_size'],m['candidates'],m['references_max'],m['horizon'],m['observed_points'])==(32,4,9,24,12544)


def test_wavefronts_cover_each_cell_once_with_full_endpoints():
    for n,m in itertools.product(range(1,6),repeat=2):
        cells=[(i,d-i) for d,(lo,hi) in enumerate(loss._diagonals(n,m)) for i in range(lo,hi+1)]
        assert cells[0]==(0,0) and cells[-1]==(n-1,m-1)
        assert set(cells)==set(itertools.product(range(n),range(m))) and len(cells)==n*m


def test_changed_policy_refuses_gamma_or_scale_sweep(tmp_path):
    p=loss.load_policy()
    for name,value in [('gamma',.2),('xyz_scale_m',.2),('support_radius_m',.3)]:
        changed=copy.deepcopy(p);changed[name]=value
        path=tmp_path/'policy.json';path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):loss.load_policy(path)
        with pytest.raises(ValueError):loss.validate_policy(changed)


@pytest.mark.parametrize('shape',[(1,1),(1,4),(3,1),(2,3),(3,3)])
def test_real_torch_dp_matches_all_complete_monotone_paths(shape):
    t=torch_module();a=np.random.default_rng(7).uniform(.02,.2,size=shape)
    c=t.tensor(np.stack([a,a*.7]),dtype=t.float64,requires_grad=True)
    actual=loss.soft_dtw(c,.1)
    expected=t.tensor([enumerated_soft_dtw(a,.1),enumerated_soft_dtw(a*.7,.1)],dtype=t.float64)
    assert t.allclose(actual,expected,atol=1e-12,rtol=1e-12)
    gradient=t.autograd.grad(actual.sum(),c)[0]
    assert t.allclose(gradient[:,0,0],t.ones(2,dtype=t.float64))
    assert t.allclose(gradient[:,-1,-1],t.ones(2,dtype=t.float64))
    assert bool((gradient>=0).all())


def test_real_torch_divergence_identity_self_gradient_symmetry_and_gradcheck():
    t=torch_module();x=t.tensor([[[.1,.2],[.3,-.1],[.5,.4]]],dtype=t.float64,requires_grad=True)
    y=t.tensor([[[.05,.2],[.3,.1],[.4,.4]]],dtype=t.float64)
    identity=loss.soft_dtw_divergence(x,x.detach(),.1)
    assert t.allclose(identity,t.zeros_like(identity),atol=1e-13,rtol=0)
    assert float(t.autograd.grad(identity.sum(),x)[0].abs().max())<1e-11
    assert t.allclose(loss.soft_dtw_divergence(x,y),loss.soft_dtw_divergence(y,x),atol=1e-12)
    assert t.autograd.gradcheck(lambda q:loss.soft_dtw_divergence(q,y),(x,),eps=1e-6,atol=1e-5,rtol=1e-4)


def fixture(h=24):
    t=torch_module();rng=np.random.default_rng(17)
    x=t.tensor(rng.normal(size=(1,4,h,3))*.025,dtype=t.float64,requires_grad=True)
    y=t.tensor(rng.normal(size=(1,2,h,3))*.025,dtype=t.float64)
    opened=t.ones((1,4,h),dtype=t.float64);ref_open=t.ones((1,2,h),dtype=t.float64)
    obs={'world_xyz':t.tensor([[[.007,.011,.053],[.083,-.057,.027],[-.089,.043,-.047]]],dtype=t.float64),
         'valid_mask':t.ones((1,3),dtype=t.bool)}
    return t,x,opened,y,ref_open,t.ones((1,2),dtype=t.bool),obs


def test_real_torch_descriptor_numerical_gradient_and_strict_observation():
    t,x,_,_,_,_,obs=fixture(3);x=x[:,:1].detach().requires_grad_()
    assert t.autograd.gradcheck(lambda q:loss.observed_descriptors(q,obs)['values'],(x,),eps=1e-6,atol=1e-5,rtol=1e-4)
    for name in ('target_xyz','routes','obstacle_centers','route_types','guide'):
        with pytest.raises(ValueError):loss.observed_descriptors(x,dict(obs,**{name:'forbidden'}))


def test_real_torch_unknown_support_retains_xyz_and_event_original_weights():
    t,x,opened,y,ref_open,mask,obs=fixture();obs['valid_mask'].zero_()
    b,_=loss.ordered_relation_costs(x,opened,y,ref_open,mask,'xyz_divergence')
    c,meta=loss.ordered_relation_costs(x,opened,y,ref_open,mask,'observed_divergence',obs)
    assert t.equal(b,c) and meta['prediction_support_fraction']==0
    assert bool(t.isfinite(t.autograd.grad(c.sum(),x)[0]).all())
    changed=ref_open.clone();changed[0,0,7]=0
    d,_=loss.ordered_relation_costs(x,opened,y,changed,mask,'xyz_divergence')
    assert t.allclose(d[0,:,0]-b[0,:,0],t.full((4,),.25*.2**2/23,dtype=t.float64),atol=1e-14)
    assert t.equal(d[0,:,1],b[0,:,1])
    changed[:,0,0]=0
    e,_=loss.ordered_relation_costs(x,opened,y,changed,mask,'xyz_divergence')
    assert t.equal(d,e) # original known initial event is not supervised again


def test_real_torch_prediction_self_term_full_gradient_and_cache_equivalence():
    t,x,opened,y,ref_open,mask,obs=fixture()
    cache=loss.prepare_reference_descriptors(y,mask,obs)
    direct,_=loss.ordered_relation_costs(x,opened,y,ref_open,mask,'observed_divergence',obs)
    cached,_=loss.ordered_relation_costs(x,opened,y,ref_open,mask,'observed_divergence',obs,cache)
    assert t.equal(direct,cached)
    analytical=t.autograd.grad(cached.sum(),x)[0]
    for index in [(0,0,2,0),(0,2,9,1),(0,3,20,2)]:
        plus=x.detach().clone();minus=x.detach().clone();plus[index]+=1e-6;minus[index]-=1e-6
        a,_=loss.ordered_relation_costs(plus,opened,y,ref_open,mask,'observed_divergence',obs,cache)
        b,_=loss.ordered_relation_costs(minus,opened,y,ref_open,mask,'observed_divergence',obs,cache)
        assert float(analytical[index])==pytest.approx(float((a-b).sum()/2e-6),abs=1e-6,rel=1e-4)
    altered=copy.deepcopy(cache);altered['description']['values'][0,0,0,0]+=.01
    with pytest.raises(ValueError):loss.ordered_relation_costs(x,opened,y,ref_open,mask,'observed_divergence',obs,altered)
    changed_obs=dict(obs,world_xyz=obs['world_xyz']+.001)
    with pytest.raises(ValueError):loss.ordered_relation_costs(x,opened,y,ref_open,mask,'observed_divergence',changed_obs,cache)


def test_real_torch_whole_reference_no_segmentwise_splicing_or_unknown_type_label():
    t,x,opened,y,ref_open,mask,obs=fixture()
    # A candidate assembled from two distinct references is not allowed to take
    # the best reference independently at every timestep. Both intact costs stay.
    y.zero_();y[0,1,:,1]=.2
    mixed=y[0,0].clone();mixed[12:,1]=.2
    x=mixed[None,None].expand(1,4,24,3).clone().requires_grad_()
    costs,_=loss.ordered_relation_costs(x,opened,y,ref_open,mask,'xyz_divergence')
    assert costs.shape==(1,4,2) and float(costs.min())>.001
    changed=y.clone();changed[:,1,:,2]=.1
    other,_=loss.ordered_relation_costs(x,opened,changed,ref_open,mask,'xyz_divergence')
    assert t.equal(costs[:,:,0],other[:,:,0])
    # Type names are not part of the loss API; known positives are not filtered.
    assert 'route_types' not in loss.ordered_relation_costs.__code__.co_varnames


def test_real_torch_no_supervision_gradient_and_padded_reference_preserved():
    t,x,opened,y,ref_open,mask,obs=fixture();y.requires_grad_();ref_open.requires_grad_()
    cost,_=loss.ordered_relation_costs(x,opened,y,ref_open,mask,'observed_divergence',obs)
    cost.sum().backward();assert y.grad is None and ref_open.grad is None and x.grad is not None
    mask[0,1]=False;bad=y.detach().clone();bad[:,1]=float('nan')
    q,_=loss.ordered_relation_costs(x,opened,bad,ref_open,mask,'xyz_divergence')
    assert bool(t.isinf(q[:,:,1]).all()) and bool(t.isfinite(q[:,:,0]).all())
    with pytest.raises(ValueError):loss.ordered_relation_costs(x,opened,y,ref_open,t.zeros_like(mask),'xyz_divergence')


def test_real_torch_C_identity_and_support_unknown_is_not_free_certificate():
    t,x,opened,y,ref_open,mask,obs=fixture()
    reference=x.detach()[:,:1];reference_open=opened[:,:1];one=t.ones((1,1),dtype=t.bool)
    a,_=loss.ordered_relation_costs(x,opened,reference,reference_open,one,'observed_divergence',obs)
    assert abs(float(a[0,0,0]))<1e-12
    g=t.autograd.grad(a[0,0,0],x)[0]
    assert float(g.abs().max())<1e-10
    desc=loss.observed_descriptors(x.detach()+10,obs)
    assert not bool(desc['known'].any())
    assert set(desc)=={'values','known'} and 'free' not in desc


def test_real_torch_selected_nearest_gradient_equals_full_dense_min():
    t,x,_,_,_,_,obs=fixture(3);x=x[:,:1]
    p=loss.load_policy();centers=t.cat([(x[:,:,:-1]+x[:,:,1:])*.5,x[:,:,-1:]],2)
    queries=centers[...,None,:]+x.new_tensor(p['descriptor_offsets'])*p['descriptor_offset_m']
    dense=((queries[...,None,:]-obs['world_xyz'][:,None,None,None]).square().sum(-1)).min(-1).values
    full=t.exp(-dense/(2*p['descriptor_sigma_m']**2))
    actual=loss.observed_descriptors(x,obs)['values']
    assert t.allclose(actual,full,atol=1e-14,rtol=1e-13)
    a=t.autograd.grad(actual.sum(),x,retain_graph=True)[0];b=t.autograd.grad(full.sum(),x)[0]
    assert t.allclose(a,b,atol=1e-12,rtol=1e-11)


def test_real_torch_per_input_reference_cache_batches_without_recomputation(monkeypatch):
    t,x,opened,y,ref_open,mask,obs=fixture();cache=loss.prepare_reference_descriptors(y,mask,obs)
    y2=t.cat([y,y+.001]);mask2=t.cat([mask,mask]);obs2={k:t.cat([v,v]) for k,v in obs.items()}
    cache2=loss.prepare_reference_descriptors(y+.001,mask,obs)
    def forbidden(*args,**kwargs):raise AssertionError('reference was recomputed')
    with monkeypatch.context() as m:
        m.setattr(loss,'observed_descriptors',forbidden)
        batched=loss.batch_reference_descriptors([cache,cache2],y2,mask2,obs2)
        with pytest.raises(ValueError):loss.batch_reference_descriptors([cache2,cache],y2,mask2,obs2)
    direct=loss.prepare_reference_descriptors(y2,mask2,obs2)
    assert batched['identity']==direct['identity']
    assert all(t.equal(batched['description'][k],direct['description'][k]) for k in direct['description'])


@pytest.mark.parametrize('references',[2,6])
def test_real_torch_saturation_cost_adapter_exact_original_and_negative_preserved(references):
    t=torch_module()
    from routeset.train_v2 import positive_assignment_loss
    generator=t.Generator().manual_seed(9)
    x=t.randn((2,4,23,4),generator=generator,dtype=t.float64,requires_grad=True)
    y=t.randn((2,references,23,4),generator=generator,dtype=t.float64)
    mask=t.ones((2,references),dtype=t.bool)
    costs=(x[:,:,None]-y[:,None]).square().mean((-1,-2))
    old=positive_assignment_loss(x,y,mask.numpy(),'saturation',np.random.default_rng(0))
    new=loss.saturation_loss_from_costs(costs,mask)
    assert t.equal(old,new)
    assert t.equal(t.autograd.grad(old,x,retain_graph=True)[0],t.autograd.grad(new,x)[0])
    negative=loss.saturation_loss_from_costs(t.full((2,4,references),-1.),mask)
    assert float(negative)==-1. # No assumption that self-correction is nonnegative


def test_real_torch_diagonal_coordinate_and_event_scales_equal_old_four_dimensional_MSE():
    t,x,opened,y,ref_open,mask,obs=fixture();x=x.detach();x[:,:,0]=0;y[:,:,0]=0
    s=loss.load_policy()['xyz_scale_m']
    diagonal=((x[:,:,None]/s-y[:,None]/s).square().mean(-1)).sum(-1)
    b=.75*s*s/23*diagonal+.25*((opened[:,:,None,1:]-ref_open[:,None,:,1:])*.2).square().mean(-1)
    pred=t.cat([x[:,:,1:],opened[:,:,1:,None]*.2],-1)
    target=t.cat([y[:,:,1:],ref_open[:,:,1:,None]*.2],-1)
    original=(pred[:,:,None]-target[:,None]).square().mean((-1,-2))
    assert t.allclose(b,original,atol=1e-15,rtol=1e-13)
