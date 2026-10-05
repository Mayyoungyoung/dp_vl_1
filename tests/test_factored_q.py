import numpy as np
import pytest
import torch
from routeset.factored_q import (FactorRouteScorer, factor_labels, training_loss,
                                apply_calibration, joint_nll)


def test_conditional_gradient_excludes_task_failures():
    z=torch.tensor([[[.2,-.3],[.1,.7]]],requires_grad=True)
    t=torch.tensor([[0.,1.]]);f=torch.tensor([[1.,0.]])
    loss=training_loss(z,t,f,'conditional');loss.backward()
    assert z.grad[0,0,1]==0 and z.grad[0,1,1]!=0 and z.grad[0,0,0]!=0
    zz=z.detach().clone().requires_grad_()
    loss2=training_loss(zz,t,torch.tensor([[0.,0.]]),'conditional')
    torch.testing.assert_close(loss,loss2)


def test_empty_task_batch_and_extreme_logits_are_finite():
    z=torch.tensor([[[80.,-80.],[-80.,80.]]],requires_grad=True)
    loss=training_loss(z,torch.zeros(1,2),torch.ones(1,2),'conditional')
    loss.backward();assert torch.isfinite(loss) and torch.isfinite(z.grad).all()
    assert torch.count_nonzero(z.grad[...,1])==0
    for arm in ('joint','marginal','conditional'):
        assert torch.isfinite(joint_nll(z,torch.ones(1,2),arm))


def test_same_initial_parameters_and_candidate_permutation():
    x=torch.randn(2,8,47,80);c=torch.randn(2,8,131);models=[]
    for arm in ('joint','marginal','conditional'):
        torch.manual_seed(3);models.append(FactorRouteScorer(arm))
    for key,v in models[0].state_dict().items():
        for m in models[1:]:torch.testing.assert_close(v,m.state_dict()[key],rtol=0,atol=0)
    order=[7,2,3,0,1,5,4,6]
    torch.testing.assert_close(models[2](x[:,order],c[:,order]),models[2](x,c)[:,order])


def test_product_identity_no_competition():
    q,f=apply_calibration(torch.ones(1,8,2)*3,'conditional',{'temperatures':[1.,2.]})
    torch.testing.assert_close(q,f[...,0]*f[...,1],rtol=0,atol=0)
    assert q.sum()>1 and torch.all(q<=f[...,0]) and torch.all(q<=f[...,1])


def test_labels_preserve_checker_conjunction_and_events():
    c=dict(semantic_goal_correct=True,finite_event_values=True,event_state_sequence_correct=True,
           finite_xyz=True,starts_at_current_state=True,tip_segments_clear=True,TipValid=True)
    changed=dict(c,event_state_sequence_correct=False,TipValid=False)
    t,f=factor_labels([c,changed]);np.testing.assert_array_equal(t,[True,False]);assert f.all()
    with pytest.raises(ValueError):factor_labels([dict(c,TipValid=False)])


def test_platt_float32_inputs_do_not_stall_at_identity():
    from scripts.run_factored_q import fit_joint_platt
    z=np.linspace(-5,5,101,dtype=np.float32)
    target=1/(1+np.exp(-(.4*z.astype(float)+.7)))
    fit=fit_joint_platt(z,target)
    assert abs(fit['slope']-.4)<.01 and abs(fit['intercept']-.7)<.01
