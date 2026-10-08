import copy
import numpy as np
import torch
from scripts.research_v3_frequency import match_loss, probabilities, pad_targets, record_history


def test_pause_does_not_add_history_or_change_resume_schedule():
    uninterrupted=[s for s in range(1,151) if record_history(s,150)]
    resumed=[s for part in (range(1,51),range(51,151)) for s in part if record_history(s,150)]
    assert uninterrupted == resumed == [100,150]


def test_frequency_mass_is_about_demonstrations_not_validity():
    tags=np.array(['gap0|gap0']*4+['gap1|gap1']*4+['over|over']*4)
    p=probabilities(tags,'empirical_90')
    assert p=={'gap0|gap0':.9,'gap1|gap1':(1-.9)/2,'over|over':(1-.9)/2}
    assert all(p==probabilities(tags,'empirical_90') for _ in range(3))
    assert np.isclose(sum(probabilities(tags,'balanced').values()),1.)


def test_ordinary_injective_group_matching_does_not_reuse_one_candidate():
    pred=torch.tensor([[[[0.]],[[1.]]]],requires_grad=True)
    target=torch.tensor([[[[0.]],[[0.1]],[[2.]],[[2.1]]]])
    tags=np.array([['left','left','right','right']])
    loss=match_loss(pred,target,tags,np.random.default_rng(1))
    assert abs(float(loss)-.5)<1e-6
    loss.backward();assert pred.grad[0,1,0,0]<0
    perm=match_loss(pred.detach().flip(1),target,tags,np.random.default_rng(1))
    assert float(perm)==float(loss)


def test_matching_handles_more_groups_than_budget_without_impossible_full_cover():
    pred=torch.zeros(1,2,1,1,requires_grad=True)
    target=torch.arange(4.).reshape(1,4,1,1)
    loss=match_loss(pred,target,np.array([['a','b','c','d']]),np.random.default_rng(3))
    assert torch.isfinite(loss);loss.backward();assert torch.isfinite(pred.grad).all()


def test_reference_padding_preserves_h24_geometry_events_and_mask():
    a,b=np.ones((8,24,3),np.float32),np.ones((31,24,3),np.float32)*2
    x,e,m=pad_targets([a,b],[np.ones((8,24)),np.zeros((31,24))])
    assert x.shape==(2,31,24,3) and e.shape==(2,31,24)
    assert m.sum(1).tolist()==[8,31]
    np.testing.assert_array_equal(x[0,:8],a);np.testing.assert_array_equal(x[1],b)
    assert not x[0,8:].any() and not m[0,8:].any()
