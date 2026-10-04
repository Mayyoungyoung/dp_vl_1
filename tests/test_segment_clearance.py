import numpy as np
import torch
from routeset.segment_clearance import segment_box_signed_clearance, segment_clearance_loss
from routeset.geometry import segment_aabb_intersection


def test_crossing_with_clear_vertices_and_penetration_gradient():
    a=torch.tensor([[-.2,.013,0.]],dtype=torch.double,requires_grad=True)
    b=torch.tensor([[.2,.013,0.]],dtype=torch.double,requires_grad=True)
    c=torch.zeros(3,dtype=torch.double);h=torch.tensor([.03,.04,.05],dtype=torch.double)
    d=segment_box_signed_clearance(a,b,c,h)
    assert abs(d.item()+.027)<1e-10
    (.02-d).square().sum().backward()
    assert a.grad.abs().sum()+b.grad.abs().sum()>0


def test_exact_slab_equivalence_and_subdivision():
    rng=np.random.default_rng(928)
    a=rng.uniform(-.4,.4,(1000,3));b=rng.uniform(-.4,.4,(1000,3))
    c=rng.uniform(-.1,.1,(1000,3));h=rng.uniform(.01,.12,(1000,3))
    t=[torch.tensor(x,dtype=torch.double) for x in (a,b,c,h)]
    d=segment_box_signed_clearance(*t)
    actual=segment_aabb_intersection(a,b,c-h-.02,c+h+.02)
    assert np.array_equal(d.numpy()<=.02,actual)
    mid=(t[0]+t[1])/2
    split=torch.minimum(segment_box_signed_clearance(t[0],mid,*t[2:]),segment_box_signed_clearance(mid,t[1],*t[2:]))
    torch.testing.assert_close(d,split,atol=1e-12,rtol=0)


def test_gradcheck_non_degenerate():
    a=torch.tensor([[-.19,.013,.004]],dtype=torch.double,requires_grad=True)
    b=torch.tensor([[.23,.021,.006]],dtype=torch.double,requires_grad=True)
    c=torch.zeros(3,dtype=torch.double);h=torch.tensor([.03,.04,.05],dtype=torch.double)
    assert torch.autograd.gradcheck(lambda x,y:segment_box_signed_clearance(x,y,c,h),(a,b))


def test_zero_length_inside_outside_and_all_queries():
    points=torch.tensor([[0.,.013,0.],[.2,0.,0.]],dtype=torch.double)
    h=torch.tensor([.03,.04,.05],dtype=torch.double)
    d=segment_box_signed_clearance(points,points,torch.zeros(3),h)
    torch.testing.assert_close(d,torch.tensor([-.027,.17],dtype=torch.double))
    p=torch.zeros(1,8,24,3,dtype=torch.double);p[...,1]=.013;p.requires_grad_()
    loss=segment_clearance_loss(p,torch.zeros(1,1,3),h[None,None]);loss.backward()
    assert (p.grad.abs().sum((2,3))>0).all()
