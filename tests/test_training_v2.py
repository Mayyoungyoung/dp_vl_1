import numpy as np
import torch

from routeset.train_v2 import positive_assignment_loss, rng_state, restore_rng


def test_positive_assignment_permutation_and_gradient():
    torch.manual_seed(9)
    pred = torch.randn(2, 4, 3, 3, requires_grad=True)
    target = torch.randn(2, 7, 3, 3)
    mask = np.array([[1,1,1,1,1,0,0],[1,1,0,0,0,0,0]], dtype=bool)
    a = positive_assignment_loss(pred, target, mask, 'positive', np.random.default_rng(7))
    b = positive_assignment_loss(pred[:,[2,0,3,1]], target, mask, 'positive', np.random.default_rng(7))
    torch.testing.assert_close(a,b)
    a.backward()
    assert torch.isfinite(pred.grad).all() and pred.grad.abs().sum() > 0


def test_padding_not_a_negative_or_training_target():
    pred = torch.zeros(1,2,2,3,requires_grad=True)
    target = torch.ones(1,5,2,3)
    mask = np.array([[1,0,0,0,0]],dtype=bool)
    loss = positive_assignment_loss(pred,target,mask,'positive',np.random.default_rng(0))
    target[:,1:] = -1000
    changed = positive_assignment_loss(pred,target,mask,'positive',np.random.default_rng(0))
    torch.testing.assert_close(loss,changed)
    assert loss.item() == 1.


def test_rectangular_assignment_chooses_valid_positive_subset():
    pred = torch.tensor([[[[0.,0.,0.]],[[10.,10.,10.]]]])
    target = torch.tensor([[[[0.,0.,0.]],[[5.,5.,5.]],[[10.,10.,10.]]]])
    loss = positive_assignment_loss(pred,target,np.ones((1,3),bool),'positive',np.random.default_rng(0))
    assert loss.item() == 0.


def test_rng_restore_continues_sampling():
    rng = np.random.default_rng(3)
    state = rng_state(rng)
    a, b = rng.integers(10000,size=30), torch.randn(10)
    restore_rng(state,rng)
    np.testing.assert_array_equal(a,rng.integers(10000,size=30))
    torch.testing.assert_close(b,torch.randn(10),atol=0,rtol=0)


def test_saturation_matches_exhaustive_valid_multiset_optimum():
    import itertools
    pred = torch.tensor([[[[-1.,0.,0.]], [[-0.8,0.,0.]], [[1.1,0.,0.]], [[1.2,0.,0.]]]],requires_grad=True)
    target = torch.tensor([[[[-1.,0.,0.]], [[1.,0.,0.]]]])
    loss = positive_assignment_loss(pred,target,np.ones((1,2),bool),'saturation',np.random.default_rng(0))
    options = [(pred[0]-target[0,list(assignment)]).square().mean()
               for assignment in itertools.product(range(2),repeat=4) if len(set(assignment))==2]
    torch.testing.assert_close(loss,torch.stack(options).min())
    loss.backward()
    assert torch.isfinite(pred.grad).all()
