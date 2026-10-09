import unittest
import numpy as np
from research_feasible_space_v1.geometry import segment_radii,node_radii_numpy
from routeset.verified_route_set import signed_clearances

class CertificateTests(unittest.TestCase):
    def test_ray_evidence_preserves_unknown_occlusion(self):
        from research_feasible_space_v1.observation_evidence import classify_probes
        depth=np.ones((3,3));k=np.array([[-1,0,1],[0,-1,1],[0,0,1.]])
        p=np.array([[0,0,.5],[0,0,1],[0,0,2],[0,0,-1],[10,0,.5]])
        s=classify_probes(p,depth,k,np.eye(4))
        self.assertEqual(s.tolist(),['free_at_probe','observed_surface','unknown','unknown','unknown'])
    def test_continuous_collision_between_clear_endpoints(self):
        p=np.linspace([-.2,0,.2],[.2,0,.2],24)[None]
        r,slack=segment_radii(p,[[0,0,.2]],[[.03,.03,.03]],0)
        self.assertLess(slack.min(),0);self.assertTrue((r[slack<=0]==0).all())
    def test_box_certificate_protects_extreme_shared_nodes(self):
        p=np.linspace([-.2,.09,.2],[.2,.09,.2],24)[None]
        r,slack=segment_radii(p,[[0,0,.2]],[[.03,.03,.03]],0)
        n=node_radii_numpy(r);self.assertGreater(r.min(),0)
        # Every coordinate independently reaches an extreme, including towards box.
        offset=np.zeros_like(p);offset[...,1]=-n;offset[...,2]=-n
        moved=p+offset
        self.assertGreater(signed_clearances(moved,[[0,0,.2]],[[.05,.05,.05]]).min(),0)
        self.assertTrue((moved[...,2]>=0).all())
    def test_final_segment_certificate_includes_shared_connectors(self):
        r=np.array([[.03,.002]+[.04]*21]);n=node_radii_numpy(r)
        self.assertEqual(n[0,0],0);self.assertEqual(n[0,-1],0)
        self.assertEqual(n[0,1],.002);self.assertEqual(n[0,2],.002)

try:
    import torch
    from research_feasible_space_v1.mapping import construct,node_radii
except ImportError:torch=None

@unittest.skipIf(torch is None,'Torch tests run in unchanged authorized server environment')
class MappingTests(unittest.TestCase):
    def test_bound_gradient_and_endpoints(self):
        p=torch.randn(3,8,24,3);r=torch.rand(3,8,23)*.06
        z=torch.randn_like(p,requires_grad=True);q=construct(p,r,z)
        self.assertTrue(((q-p).abs()<=node_radii(r)[...,None]+1e-6).all())
        torch.testing.assert_close(q[...,[0,-1],:],p[...,[0,-1],:])
        q.square().mean().backward();self.assertGreater(z.grad[...,1:-1,:].abs().sum().item(),0)
    def test_same_info_projection(self):
        p=torch.zeros(1,8,24,3);r=torch.full((1,8,23),.02);z=torch.full_like(p,10)
        a=construct(p,r,z,'xyz');b=construct(p,r,z,'projection')
        self.assertGreater(a.abs().max().item(),.02);self.assertLessEqual(b.abs().max().item(),.020001)
    def test_tapered_exactness_and_fixed_end_floor(self):
        from research_feasible_space_v1.tapered import tapered_segment_clearance,tapered_clearance_numpy
        p=np.linspace([-.2,.09,.2],[.2,.09,.02],24)[None];n=np.full((1,24),.005);n[:,[0,-1]]=0
        cs=np.array([[0,0,.2]]);hs=np.array([[.05,.05,.05]])
        expected=tapered_clearance_numpy(p,n,cs,hs,.02)
        actual=tapered_segment_clearance(torch.tensor(p[:,:-1,None]),torch.tensor(p[:,1:,None]),torch.tensor(n[:,:-1,None]),torch.tensor(n[:,1:,None]),torch.tensor(cs),torch.tensor(hs)).min(-1).values.numpy()
        floor=np.minimum(p[:,:-1,2]-n[:,:-1],p[:,1:,2]-n[:,1:])-.02
        np.testing.assert_allclose(expected,np.minimum(actual,floor),atol=1e-10)
        self.assertGreaterEqual(expected.min(),0)
    def test_companion_independent_boundary_and_nonzero_joint_gradients(self):
        from routeset.mode_geometry import ModeGeometryHead
        from research_feasible_space_v1.model import FeasibleSpaceHead
        base=ModeGeometryHead(feature_dim=16,width=32,max_candidates=8)
        model=FeasibleSpaceHead(base);ctx=torch.randn(2,32);anchor=torch.randn(2,3);current=torch.randn(2,8)
        m=torch.arange(8)[None].repeat(2,1);v=torch.zeros_like(m)
        p,e,a=model.decode(ctx,anchor,current,m,v);mm=m.clone();mm[:,0]=15
        _,_,b=model.decode(ctx,anchor,current,mm,v)
        torch.testing.assert_close(a['centers'][:,1:],b['centers'][:,1:],rtol=0,atol=0)
        torch.testing.assert_close(a['radii'][:,1:],b['radii'][:,1:],rtol=0,atol=0)
        p.square().mean().backward()
        self.assertGreater(model.relative_output.weight.grad.abs().sum().item(),0)
        self.assertGreater(sum(p.grad.abs().sum().item() for p in model.corridor_output.parameters() if p.grad is not None),0)
        self.assertTrue(all(x.grad is None for x in model.base.parameters()))

if __name__=='__main__':unittest.main()
