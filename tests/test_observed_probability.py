import numpy as np
import pytest

torch = pytest.importorskip('torch')
from routeset.observed_probability import (RouteValidityHead,route_observation_features,
    reference_cluster_weights,balanced_assignment_loss,selection_metrics)
from scripts.run_observed_probability import role_for_index


def test_export_containment_python38(tmp_path):
    from scripts.run_observed_probability import contained
    assert contained(tmp_path/'parent'/'observation.npz',tmp_path/'parent')
    assert not contained(tmp_path/'parent'/'..'/'other'/'secret',tmp_path/'parent')


def test_roles_exclude_historical_and_sealed():
    assert role_for_index(32)=='SCORE_TRAIN'
    assert role_for_index(96)=='DEV_SCORE'
    assert role_for_index(128)=='CALIBRATION'
    assert role_for_index(160)=='FUTURE_GENERATOR_TRAIN'
    for i in (0,31,256,287):
        with pytest.raises(ValueError):role_for_index(i)


def fixture():
    torch.manual_seed(2)
    paths=torch.randn(2,4,24,3)
    args=dict(paths=paths,events=torch.ones(2,4,24),current=torch.zeros(2,8),
              world_xyz=torch.randn(2,12,3),rgb=torch.rand(2,12,3),valid_mask=torch.ones(2,12,dtype=torch.bool),
              point_features=torch.randn(2,12,64),context=torch.randn(2,128),anchor=torch.randn(2,3))
    return args


def test_actual_route_features_and_candidate_equivariance():
    args=fixture();x,c=route_observation_features(**args)
    assert x.shape==(2,4,47,80) and c.shape==(2,4,131)
    net=RouteValidityHead();score=net(x,c)
    order=[3,0,1,2]
    changed=dict(args,paths=args['paths'][:,order],events=args['events'][:,order])
    xx,cc=route_observation_features(**changed)
    torch.testing.assert_close(net(xx,cc),score[:,order])
    changed['paths']=changed['paths']+0.3
    xx,cc=route_observation_features(**changed)
    assert not torch.allclose(net(xx,cc),score[:,order])


def test_masked_depth_does_not_change_features():
    args=fixture();args['valid_mask'][:,-1]=False
    x,c=route_observation_features(**args)
    args['world_xyz'][:,-1]=float('nan');args['rgb'][:,-1]=float('nan')
    xx,cc=route_observation_features(**args)
    torch.testing.assert_close(x,xx);torch.testing.assert_close(c,cc)


def test_empty_observation_fails():
    args=fixture();args['valid_mask'][:]=False
    with pytest.raises(ValueError):route_observation_features(**args)


def test_balanced_reference_mass_retains_unnamed_routes():
    p=np.zeros((1,3,24,3));p[0,1,:,0]=.001;p[0,2,:,1]=.3
    w=reference_cluster_weights(p,np.ones((1,3),bool))
    np.testing.assert_allclose(w,[[.25,.25,.5]])
    order=[2,0,1]
    np.testing.assert_allclose(reference_cluster_weights(p[:,order],np.ones((1,3),bool)),w[:,order])


def test_mode_assignment_does_not_average_separated_targets():
    p=torch.tensor([[[[0.]],[[1.]]]],requires_grad=True)
    t=torch.tensor([[[[0.]],[[0.]],[[1.]]]])
    loss,mass=balanced_assignment_loss(p,t,torch.tensor([[.25,.25,.5]]))
    assert loss.item()==0
    torch.testing.assert_close(mass,torch.tensor([[.5,.5]]))
    loss.backward();assert torch.isfinite(p.grad).all()


def test_duplicate_queries_share_probability_mass():
    p=torch.zeros(1,4,2,3,requires_grad=True);t=torch.zeros(1,1,2,3)
    loss,mass=balanced_assignment_loss(p,t,torch.ones(1,1))
    torch.testing.assert_close(mass,torch.full((1,4),.25))


def test_probability_and_validity_have_distinct_normalizations():
    q=torch.tensor([[3.,3.]])
    assert q.sigmoid().sum()>1
    assert q.softmax(-1).sum()==1


def test_temperature_preserves_selection_and_parent_bootstrap():
    logits=np.array([[0.,2.],[3.,0.],[1.,0.],[0.,1.]])
    y=np.array([[0,1],[1,0],[0,1],[1,0]])
    paths=np.zeros((4,2,24,3));parents=np.array(['a','a','b','b'])
    a=selection_metrics(logits,y,paths,parents)
    b=selection_metrics(logits,y,paths,parents,2)
    assert a['selected_valid']==b['selected_valid']==.5
    assert a['parent_count']==2 and len(a['per_parent'])==2
    assert a['brier']!=b['brier']


def test_q_checkpoint_optimizer_rng_resume_equivalence(tmp_path):
    import copy
    torch.manual_seed(71)
    model=RouteValidityHead();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
    x=torch.randn(3,4,47,80);c=torch.randn(3,4,131);y=torch.rand(3,4).round()
    def step(m,o):
        o.zero_grad();torch.nn.functional.binary_cross_entropy_with_logits(m(x,c),y).backward();o.step()
    step(model,optimizer)
    saved=copy.deepcopy(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state()))
    step(model,optimizer)
    restored=RouteValidityHead();restored.load_state_dict(saved['model'])
    opt=torch.optim.AdamW(restored.parameters(),lr=.001);opt.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['rng'])
    step(restored,opt)
    for a,b in zip(model.parameters(),restored.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
