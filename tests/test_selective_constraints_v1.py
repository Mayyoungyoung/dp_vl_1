import unittest
import numpy as np
import torch
from research_selective_repair_v1.constraints import ConstraintHead,summaries
from research_selective_repair_v1.constraint_operator import repair,crossings
from research_selective_repair_v1.core import VOCAB

class ConstraintsTests(unittest.TestCase):
    def test_empty_surface_is_explicit(self):
        priors=np.array([[.15,-.1,.9],[.15,.1,.9],[.35,-.1,.9],[.35,.1,.9]])
        x,a=summaries(np.zeros((1,8,3)),np.zeros((1,8),bool),priors)
        self.assertEqual(x.shape,(1,4,23));self.assertTrue((x[:,:,-1]==0).all());np.testing.assert_array_equal(a,priors[None].astype(np.float32))
        model=ConstraintHead();torch.testing.assert_close(model(torch.zeros(1,128),torch.tensor(x),torch.tensor(a)),torch.tensor(a))
    def test_protected_valid_representative_is_exact(self):
        p=torch.tensor(np.stack([np.linspace(0,.5,24),np.full(24,-.3),np.full(24,.82)],-1),dtype=torch.float32)[None,None]
        c=torch.tensor([[[.15,-.1,.805],[.15,.1,.805],[.35,-.1,.805],[.35,.1,.805]]]);h=torch.tensor([[[.0175,.0175,.05]]*4])
        m=torch.tensor([[VOCAB.index('gap0|gap0')]])
        q,info=repair(p,m,p[:,:,-1][:,0],torch.ones(1),c,h,dict(post_base=.755,floor=.775),steps=2)
        torch.testing.assert_close(q,p,rtol=0,atol=0);self.assertEqual(info['steps'],0)
    def test_crossing_interpolation_has_finite_gradient(self):
        p=torch.tensor(np.stack([np.linspace(0,.5,24),np.full(24,-.3),np.full(24,.82)],-1),dtype=torch.float32)[None,None].requires_grad_()
        point,ix,has=crossings(p,torch.tensor([[.15,.35]]));self.assertTrue(has.all());point.square().sum().backward();self.assertTrue(torch.isfinite(p.grad).all())

if __name__=='__main__':unittest.main()
