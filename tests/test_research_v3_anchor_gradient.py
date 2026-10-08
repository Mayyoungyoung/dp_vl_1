import torch
from routeset.observed_geometry import ObservedGeometryEncoder


def test_hard_peak_keeps_forward_and_context_gradient_but_removes_surrogate():
    torch.manual_seed(81)
    m=ObservedGeometryEncoder(feature_dim=8,width=8,point_width=8,anchor_mode='straight_through_peak').double().eval()
    args=dict(features=torch.randn(1,8,dtype=torch.double),current=torch.zeros(1,8,dtype=torch.double),
        world_xyz=torch.tensor([[[-.2,0.,.8],[.3,.1,.9],[float('nan')]*3]],dtype=torch.double),
        rgb=torch.tensor([[[1.,0.,0.],[0.,1.,0.],[0.,0.,0.]]],dtype=torch.double),
        uv=torch.zeros(1,3,2,dtype=torch.double),depth=torch.ones(1,3,dtype=torch.double),valid_mask=torch.tensor([[True,True,False]]))
    a=m(**args)
    surrogate=torch.autograd.grad(a['anchor_xyz'].sum(),m.log_attention_scale)[0]
    assert abs(surrogate.item())>1e-6
    m.anchor_mode='hard_peak';b=m(**args)
    for key in ('anchor_xyz','attention','context'):assert torch.equal(a[key],b[key])
    assert not b['anchor_xyz'].requires_grad
    derivative=torch.autograd.grad(b['context'].square().sum(),m.log_attention_scale)[0]
    original=m.log_attention_scale.detach().clone();values=[]
    with torch.no_grad():
        for shift in (-1e-5,1e-5):
            m.log_attention_scale.copy_(original+shift);v=m(**args)
            assert torch.equal(v['anchor_xyz'],b['anchor_xyz']);values.append(v['context'].square().sum())
        m.log_attention_scale.copy_(original)
    torch.testing.assert_close(derivative,(values[1]-values[0])/2e-5,atol=1e-9,rtol=1e-5)
    assert abs(derivative.item())>1e-8
