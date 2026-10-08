import copy
import numpy as np
import torch
from scripts.research_v3_frequency import match_loss, probabilities


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
