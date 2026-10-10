import unittest
import numpy as np
import torch
from research_selective_repair_v1.body_prefix_forecast import PrefixHead,tip_forward,likelihood,prefix_features
from research_selective_repair_v1.public_kinematics import forward

class StateTransport(unittest.TestCase):
    def setUp(self):
        base=np.eye(4);relative=np.repeat(np.eye(4)[None],7,0);relative[:,0,3]=.1
        self.robot=(np.zeros(7),base,relative)
    def test_fk_gradient_and_reference(self):
        q=torch.zeros((2,7),requires_grad=True);qref,b,r=[torch.tensor(v,dtype=torch.float32) for v in self.robot]
        actual=tip_forward(q,qref,b,r)
        np.testing.assert_allclose(actual.detach(),forward(q.detach().numpy(),*self.robot),atol=1e-6)
        actual[:,1].sum().backward();self.assertGreater(float(q.grad[:,0].abs().sum()),.1)
    def test_free_forward_and_unknown_suffix_loss(self):
        model=PrefixHead(self.robot);x=torch.randn(2,24,26);context=torch.randn(2,128);q0=torch.zeros(2,7)
        out=model(x,context,q0);self.assertEqual(out['q'].shape,(2,4,23,4,7))
        valid=torch.zeros(2,23,dtype=torch.bool);valid[:,0]=True
        obs=valid.clone();target=torch.zeros(2,23,4,7);tip=torch.zeros(2,23,4,3);hazard=torch.zeros(2,23)
        event=dict(q=torch.zeros(2,23,2,7),tip=torch.zeros(2,23,2,3),present=torch.zeros(2,23,2,dtype=torch.bool),observed=torch.zeros(2,23,2,dtype=torch.bool))
        event['present'][:,0,0]=True;event['observed'][:,0]=True
        a=likelihood(out,target,tip,valid,hazard,obs,event)
        # Keep tensors saved by the first autograd graph immutable.
        other=target.clone();other[:,1:]=10000
        other_tip=tip.clone();other_tip[:,1:]=10000
        other_hazard=hazard.clone();other_hazard[:,1:]=1
        b=likelihood(out,other,other_tip,valid,other_hazard,obs,event)
        self.assertAlmostEqual(float(a),float(b),places=5)
        a.backward();self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_unseen_future_nodes_preserve_the_prefix(self):
        model=PrefixHead(self.robot).eval();x=torch.randn(1,24,26);context=torch.randn(1,128);q0=torch.zeros(1,7)
        with torch.no_grad():
            a=model(x,context,q0);x[:,10:]+=100;b=model(x,context,q0)
        torch.testing.assert_close(a['q'][:,:,:9],b['q'][:,:,:9])
        torch.testing.assert_close(a['hazard'][:,:,:9],b['hazard'][:,:,:9])
        paths=np.zeros((1,24,3),np.float32);completed=np.zeros((1,4,3),np.float32)
        before=prefix_features(paths,completed);paths[:,10:]=1
        after=prefix_features(paths,completed)
        np.testing.assert_array_equal(before[:,:10],after[:,:10])

if __name__=='__main__':unittest.main()
