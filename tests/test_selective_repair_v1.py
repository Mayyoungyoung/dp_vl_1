import unittest
import numpy as np
import torch
from research_selective_repair_v1.model import RepairHead
from research_selective_repair_v1.core import masks
from research_selective_repair_v1.local import features

class RepairTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(4);torch.manual_seed(2)
    def test_contiguous_budget_and_identity(self):
        scores=torch.zeros(2,8,24);scores[0,0,2:21]=.9
        m=masks(scores,.5)
        self.assertEqual(int(m[0,0].sum()),12)
        self.assertEqual(int(m[1].sum()),0)
        self.assertEqual(int(m[...,0].sum()+m[...,-1].sum()),0)
        self.assertEqual(int(torch.diff(torch.nonzero(m[0,0]).flatten()).min()),1)
    def test_support_exterior_identity_and_companion_independence(self):
        model=RepairHead(64).eval();c=torch.randn(2,64);m=torch.arange(8)[None].expand(2,-1)
        d=torch.randn(2,8,24,3);local=torch.randn(2,8,24,29)
        with torch.no_grad():
            model.output.weight.normal_(std=.01);model.output.bias[3]=1
            p,info=model(c,m,d,local,hard=True)
            self.assertTrue(torch.equal(p[info['support']==0],d[info['support']==0]))
            mm=m.clone();mm[:,0]=15;ll=local.clone();ll[:,0]+=1
            pp,_=model(c,mm,d,ll,hard=True)
            self.assertTrue(torch.equal(p[:,1:],pp[:,1:]))
            zero,_=model(c,m,d,local,hard=True,threshold=1.01)
            self.assertTrue(torch.equal(zero,d))
    def test_unknown_feature_and_translation(self):
        d=np.tile(np.array([[0.,0.,2.]]),(8,24,1));points=np.array([[.1,0.,2.]])
        obs=dict(depth=np.ones((5,5)),camera_intrinsics=np.eye(3),camera_extrinsics=np.eye(4))
        a=features(d,points,obs);self.assertTrue((a[:,:,18]==1).all())
        shift=np.array([.2,.3,.4]);obs['camera_extrinsics'][:3,3]=shift
        b=features(d+shift,points+shift,obs)
        np.testing.assert_allclose(a,b,atol=1e-6)

if __name__=='__main__':unittest.main()
