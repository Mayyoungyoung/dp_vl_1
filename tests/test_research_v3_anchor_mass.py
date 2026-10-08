import torch
from scripts.research_v3_anchor_mass import mass_anchor


def test_mass_can_reject_isolated_point_peak_and_is_chunk_invariant():
    p=torch.tensor([[0.,0.,0.],[1.,0.,0.],[1.01,0.,0.],[.99,0.,0.]])
    w=torch.tensor([.4,.2,.2,.2])
    a,s=mass_anchor(p,w,chunk=1)
    b,t=mass_anchor(p,w,chunk=9)
    assert torch.equal(a,p[1]) and torch.equal(a,b)
    assert torch.allclose(s,t) and .5<s<.6


def test_exact_kernel_matches_direct_float64_calculation():
    p=torch.tensor([[0.,0.,0.],[.02,0.,0.],[.1,0.,0.]],dtype=torch.float64)
    w=torch.tensor([.2,.5,.3],dtype=torch.float64)
    scores=torch.exp(-torch.cdist(p,p).square()/(2*.025**2))@w
    a,s=mass_anchor(p,w,chunk=2)
    assert torch.equal(a,p[scores.argmax()]) and torch.allclose(s,scores.max(),atol=1e-14,rtol=0)
