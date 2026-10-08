import torch
from routeset.attention_mass import mass_anchor


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


def test_geometry_intervention_preserves_weights_attention_and_excludes_masked_points():
    from routeset.observed_geometry import ObservedGeometryEncoder
    torch.manual_seed(0)
    model=ObservedGeometryEncoder(feature_dim=8,width=8,point_width=8,
                                  anchor_mode='straight_through_peak').eval()
    args=dict(features=torch.randn(1,8),current=torch.zeros(1,8),
        world_xyz=torch.tensor([[[0.,0.,0.],[.01,0.,0.],[float('nan')]*3]]),
        rgb=torch.zeros(1,3,3),uv=torch.zeros(1,3,2),depth=torch.ones(1,3),
        valid_mask=torch.tensor([[True,True,False]]))
    with torch.inference_mode():
        original=model(**args)
        model.anchor_mode='local_mass_peak'
        modified=model(**args)
    assert torch.equal(original['attention'],modified['attention'])
    assert torch.isfinite(modified['context']).all()
    expected,_=mass_anchor(args['world_xyz'][0,:2],original['attention'][0,:2])
    assert torch.equal(expected,modified['anchor_xyz'][0])
