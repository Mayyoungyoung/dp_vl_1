import copy
import numpy as np
import pytest
torch=pytest.importorskip('torch')
from routeset.paired_modes import row_crossings,relation_cost,within_scene_loss,partial_pair_loss,class_weights
from scripts.paired_modes_data import registration
from scripts.observed_layout_variation import crossing_signature
from scripts.train_paired_modes import sample_batch
from routeset.geometric_modes import portal_word


def example():
    plan=registration()['parent_plan'][0];cfg=plan['config']
    good=[s for s in plan['guide_plans'][0] if s['collection_allowed']]
    paths=torch.tensor([np.vstack([cfg['entry_xyz'],s['waypoints_supervision_only']]).tolist() for s in good],dtype=torch.float64)
    modes=[tuple(s['intent_supervision_only']) for s in good]
    return paths,cfg,modes


def test_witnesses_satisfy_geometric_relations_and_permutation():
    x,c,m=example();cost=relation_cost(x,c,m)
    assert torch.max(cost.diag())<1e-12
    assert float(within_scene_loss(x[None],[c],[m]))<1e-12
    pair,_=partial_pair_loss(torch.stack([x,x.flip(0)]),[c,c],[m,m],[(0,1)])
    assert float(pair)<1e-12


def test_partial_correspondence_ignores_unmatched_and_allows_translation():
    x,c,m=example();d=copy.deepcopy(c);d['post_y']=[[v+.02 for v in row] for row in c['post_y']]
    shifted=x.clone();shifted[...,1]+=.02
    loss,n=partial_pair_loss(torch.stack([x,shifted]),[c,d],[m,m[:-1]],[(0,1)])
    assert n==len(m)-1 and float(loss)<1e-12
    loss,n=partial_pair_loss(torch.stack([x,shifted]),[c,d],[[m[0]],[m[-1]]],[(0,1)])
    assert n==0 and float(loss)==0


def test_shared_exterior_modes_can_deform_when_gap_closes():
    plans=registration()['parent_plan'][:2];paths=[];modes=[]
    for plan in plans:
        slots=[s for s in plan['guide_plans'][0] if s['collection_allowed']]
        x=torch.tensor([np.vstack([plan['config']['entry_xyz'],s['waypoints_supervision_only']]).tolist() for s in slots],dtype=torch.float64)
        paths.append(x);modes.append([tuple(s['intent_supervision_only']) for s in slots])
    loss,n=partial_pair_loss(paths,[p['config'] for p in plans],modes,[(0,1)])
    assert n>0 and float(loss)<1e-12
    # The routes really deform: shared outer passage y coordinates differ.
    assert not torch.equal(paths[0][0],paths[1][0])


def test_relation_loss_has_correct_nonzero_finite_difference_gradient():
    x,c,m=example();x=x[:1].clone();x[...,1]+=.11;x.requires_grad_(True)
    f=lambda v: relation_cost(v,c,[m[0]]).sum()
    grad=torch.autograd.grad(f(x),x)[0]
    assert grad.abs().sum()>0 and torch.isfinite(grad).all()
    index=np.unravel_index(int(grad.abs().argmax()),grad.shape)
    a=x.detach().clone();b=a.clone();a[index]+=1e-6;b[index]-=1e-6
    assert abs(float((f(a)-f(b))/2e-6-grad[index]))<1e-6


def test_reference_weights_keep_unknown_and_balance_classes():
    x,c,m=example();a=x.numpy();a=np.concatenate([a,a[:1],np.zeros_like(a[:1])])
    w,known=class_weights(a[None],np.ones((1,len(a)),bool),[c],crossing_signature)
    assert np.isclose(w.sum(),1) and w[0,-1]>0
    assert np.isclose(w[0,0],w[0,-2]) and len(known[0])==len(m)


def test_sampler_resume_preserves_pairing_and_order():
    groups=[[i*3+j for j in range(3)] for i in range(6)];old=np.arange(100,130)
    rng=np.random.default_rng(11);sample_batch(rng,groups,old);state=copy.deepcopy(rng.bit_generator.state)
    a,p=sample_batch(rng,groups,old);other=np.random.default_rng();other.bit_generator.state=state
    b,q=sample_batch(other,groups,old)
    assert np.array_equal(a,b) and p==q
    assert all(a[i]//3==a[j]//3 and a[i]!=a[j] for i,j in p)
    assert np.all(a[16:]>=100)


def test_variable_height_portal_and_subdivision():
    x,c,m=example()
    for path in x.numpy():
        mid=(path[:-1]+path[1:])/2
        divided=np.empty((2*len(path)-1,3));divided[::2]=path;divided[1::2]=mid
        assert portal_word(path,c)==portal_word(divided,c)
