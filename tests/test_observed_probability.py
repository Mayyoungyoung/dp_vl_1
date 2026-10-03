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


def test_probabilistic_generator_preserves_baseline_paths():
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    torch.manual_seed(12)
    baseline=ObservedGeometryRouteHead(feature_dim=16,max_candidates=8)
    model=ProbabilisticGeometryRouteHead(feature_dim=16,max_candidates=8)
    missing,unexpected=model.load_state_dict(baseline.state_dict(),strict=False)
    assert set(missing)=={'mode_mass.weight','mode_mass.bias'} and not unexpected
    args=dict(features=torch.randn(2,16),current=torch.randn(2,8),world_xyz=torch.randn(2,10,3),
              rgb=torch.rand(2,10,3),uv=torch.rand(2,10,2),depth=torch.ones(2,10),valid_mask=torch.ones(2,10,dtype=torch.bool))
    a,b,_=baseline(**args);x,y,d=model(**args)
    torch.testing.assert_close(a,x,rtol=0,atol=0);torch.testing.assert_close(b,y,rtol=0,atol=0)
    torch.testing.assert_close(d['pi'],torch.full((2,8),1/8))
    with pytest.raises(ValueError):model(**args,k=4)


def test_deployment_output_k_does_not_change_internal_mass_or_paths():
    from routeset.observed_probability import ProbabilisticGeometryRouteHead,ScoredRoutePlanner
    generator=ProbabilisticGeometryRouteHead(feature_dim=16,max_candidates=8)
    norm=dict(nodes_mean=np.zeros(80,dtype='float32'),nodes_std=np.ones(80,dtype='float32'),
              context_mean=np.zeros(131,dtype='float32'),context_std=np.ones(131,dtype='float32'))
    planner=ScoredRoutePlanner(generator,RouteValidityHead(),norm,temperature=1.5).eval()
    args=dict(features=torch.randn(1,16),current=torch.randn(1,8),world_xyz=torch.randn(1,12,3),
              rgb=torch.rand(1,12,3),uv=torch.rand(1,12,2),depth=torch.ones(1,12),valid_mask=torch.ones(1,12,dtype=torch.bool))
    with torch.no_grad():
        a=planner(**args,return_k=1);b=planner(**args,return_k=4)
    assert a['selected_indices'].shape==(1,1) and b['selected_indices'].shape==(1,4)
    for key in ('paths','pi','q'):torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    assert a['internal_candidates']==8
    assert ((a['q']>0)&(a['q']<1)).all()


def test_q_first_diverse_selection_does_not_use_oracle_labels():
    from routeset.observed_probability import select_route_indices
    paths=torch.zeros(1,4,24,3);paths[:,2,:,1]=.1;paths[:,3,:,1]=.2
    scores=torch.tensor([[.9,.8,.7,.1]])
    assert select_route_indices(paths,scores,2).tolist()==[[0,2]]
    assert select_route_indices(paths,scores,4).tolist()==[[0,2,1,3]]


def test_composite_evaluation_uses_matching_verifier_and_restores(monkeypatch):
    from scripts import train_observed_two_row as ordinary
    from scripts.export_two_row_composite_observations import verify_export
    from scripts.train_observed_probability_set import evaluate_composite
    original=ordinary.verify_export
    def fake(*args,**kwargs):
        assert ordinary.verify_export is verify_export
        return {'ok':True}
    monkeypatch.setattr(ordinary,'evaluate',fake)
    assert evaluate_composite()=={'ok':True}
    assert ordinary.verify_export is original


@pytest.mark.parametrize('role',['SCORE_TRAIN','DEV_SCORE','CALIBRATION','TEST_LOCKED','DEV_MODEL'])
def test_generator_expansion_rejects_every_held_role(role):
    from scripts.train_observed_probability_set import append_generator_data
    with pytest.raises(ValueError):append_generator_data({}, {}, {'splits':np.array([role])}, {})


def test_generator_expansion_rejects_repeated_parent():
    from scripts.train_observed_probability_set import append_generator_data
    with pytest.raises(ValueError):append_generator_data({'parent_ids':['a']},{},
        {'splits':np.array(['FUTURE_GENERATOR_TRAIN']),'parent_ids':['a']},{})
