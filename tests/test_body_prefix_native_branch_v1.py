import unittest
import numpy as np
import torch
from research_selective_repair_v1.body_native_branch import NativeBranchHead,likelihood

ROBOT=dict(lower=[-2.9,-1.76,-2.9,-3.07,-2.9,-.0175,-2.9],upper=[2.9,1.76,2.9,-.0698,2.9,3.75,2.9],q0=[1.4,-.33,-1.0,-3.04,1.8,3.52,-.7])

class NativeBranchTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(17)
        self.x=torch.randn(2,24,26);self.c=torch.randn(2,128)
        self.paths=torch.zeros(2,24,3);self.paths[:,:,0]=torch.linspace(0,1,24)
        self.posts=torch.tensor([[[.3,-.1,.5],[.3,.1,.5],[.6,-.1,.5],[.6,.1,.5]]]*2)
    def forward(self,model,x=None):return model(self.x if x is None else x,self.c,self.paths,self.posts)
    def test_bounds_and_causal_prefix(self):
        model=NativeBranchHead(ROBOT);p=self.forward(model)
        q=p['joint_mean']*(model.upper-model.lower)/2+(model.upper+model.lower)/2
        self.assertTrue(bool(((q>=model.lower-1e-6)&(q<=model.upper+1e-6)).all()))
        changed=self.x.clone();changed[:,9:]+=20
        other=self.forward(model,changed)
        torch.testing.assert_close(p['joint_mean'][:,:,:8],other['joint_mean'][:,:,:8],rtol=0,atol=0)
        torch.testing.assert_close(p['hazard'][:,:,:8],other['hazard'][:,:,:8],rtol=0,atol=0)
    def test_feedback_only_difference(self):
        branch=NativeBranchHead(ROBOT,True);aux=NativeBranchHead(ROBOT,False);aux.load_state_dict(branch.state_dict())
        self.assertEqual(sum(p.numel() for p in branch.parameters()),sum(p.numel() for p in aux.parameters()))
        p=self.forward(branch);q=self.forward(aux)
        a=torch.autograd.grad(p['hazard'].sum(),branch.joint.weight,allow_unused=True)[0]
        b=torch.autograd.grad(q['hazard'].sum(),aux.joint.weight,allow_unused=True)[0]
        self.assertGreater(float(a.abs().sum()),0);self.assertIsNone(b)
    def test_unknown_joint_suffix_is_masked(self):
        model=NativeBranchHead(ROBOT);p=self.forward(model)
        d=dict(prefix_hazard=torch.zeros(2,23),prefix_observed=torch.ones(2,23,dtype=torch.bool),
            event_tip=torch.zeros(2,23,2,3),event_present=torch.zeros(2,23,2,dtype=torch.bool),
            prefix_valid=torch.zeros(2,23,dtype=torch.bool),labels=torch.zeros(2,dtype=torch.long),
            prefix_q=torch.zeros(2,23,4,7),joint_label_valid=torch.zeros(2,23,dtype=torch.bool))
        original=likelihood(p,d,model.lower,model.upper);d['prefix_q']+=100
        altered=likelihood(p,d,model.lower,model.upper)
        torch.testing.assert_close(original,altered,rtol=0,atol=0)
        self.assertTrue(bool(torch.isfinite(original)))

if __name__=='__main__':unittest.main()
