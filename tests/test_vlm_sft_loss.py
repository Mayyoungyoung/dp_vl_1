import pytest
import torch
from torch import nn
from torch.nn import functional as F

from routeset.vlm_sft_loss import causal_chunked_loss


def test_chunked_causal_loss_and_input_gradient_equal_dense_with_masked_prompt():
    torch.set_num_threads(1);torch.manual_seed(4)
    head=nn.Linear(13,79,bias=False).requires_grad_(False)
    first=torch.randn(2,17,13,requires_grad=True);second=first.detach().clone().requires_grad_(True)
    labels=torch.randint(0,79,(2,17));labels[:,:5]=-100;labels[1,15:]=-100
    dense=F.cross_entropy(head(first)[:,:-1].reshape(-1,79),labels[:,1:].reshape(-1))
    chunked=causal_chunked_loss(second,labels,head,chunk_size=3)
    dense.backward();chunked.backward()
    torch.testing.assert_close(chunked,dense,atol=5e-7,rtol=5e-7)
    torch.testing.assert_close(second.grad,first.grad,atol=1e-7,rtol=1e-6)
    assert not second.grad[:,:4].any() and not second.grad[:,-1].any()
    assert head.weight.grad is None
    with torch.no_grad():torch.testing.assert_close(causal_chunked_loss(second,labels,head,2),dense.detach())


def test_chunked_loss_refuses_empty_supervision_or_trainable_vocabulary():
    head=nn.Linear(3,7).requires_grad_(False);hidden=torch.randn(1,4,3)
    with pytest.raises(ValueError,match='No supervised'):causal_chunked_loss(hidden,torch.full((1,4),-100),head)
    head.requires_grad_(True)
    with pytest.raises(ValueError,match='frozen vocabulary'):causal_chunked_loss(hidden,torch.ones((1,4),dtype=torch.long),head)
