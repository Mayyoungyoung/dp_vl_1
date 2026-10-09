import copy
import unittest
import torch
from routeset.mode_geometry import ModeGeometryHead,mode_loss,pair_displacement_loss


class ModeGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.model=ModeGeometryHead(feature_dim=12,width=16,point_width=8,horizon=24,max_candidates=8)
        self.context=torch.randn(2,16)
        self.anchor=torch.randn(2,3)
        self.current=torch.randn(2,8)
        self.modes=torch.arange(8).repeat(2,1)

    def test_modes_and_context_affect_actual_coordinates(self):
        p,e,_=self.model.decode(self.context,self.anchor,self.current,self.modes)
        changed,_,_=self.model.decode(self.context,self.anchor,self.current,self.modes+8)
        other,_,_=self.model.decode(self.context+.3,self.anchor,self.current,self.modes)
        self.assertEqual(p.shape,(2,8,24,3));self.assertEqual(e.shape,(2,8,24))
        self.assertGreater(float((p-changed).abs().max()),1e-5)
        self.assertGreater(float((p-other).abs().max()),1e-5)
        torch.testing.assert_close(p[:,:,0],self.current[:,None,:3].expand(-1,8,-1))

    def test_word_is_not_slot_identity(self):
        order=torch.tensor([7,2,1,6,3,4,5,0])
        p,_,_=self.model.decode(self.context,self.anchor,self.current,self.modes)
        q,_,_=self.model.decode(self.context,self.anchor,self.current,self.modes[:,order])
        torch.testing.assert_close(q,p[:,order],atol=2e-6,rtol=2e-6)

    def test_pair_gradients_and_nonzero_target_displacement(self):
        p=torch.zeros(2,8,24,3,requires_grad=True);target=torch.zeros_like(p)
        target[1,:,1:,0]=.2
        loss=pair_displacement_loss(p,target,torch.ones(1,8));loss.backward()
        self.assertGreater(float(loss),0)
        self.assertLess(float(p.grad[1,:,:,0].sum()),0)
        self.assertGreater(float(p.grad[0,:,:,0].sum()),0)
        self.assertEqual(float(pair_displacement_loss(target,target,torch.ones(1,8))),0)
        self.assertEqual(float(pair_displacement_loss(p,target,torch.zeros(1,8))),0)

    def test_unknown_is_not_binary_negative(self):
        evidence=torch.full((2,16),-1.);evidence[:,0]=1;evidence[:,1]=0
        logits=torch.zeros(2,16,requires_grad=True)
        mode_loss(logits,evidence).backward()
        self.assertGreater(float(logits.grad[0,1]),float(logits.grad[0,2]))
        self.assertLess(float(logits.grad[0,0]),0)

    def test_gradient_and_reload(self):
        self.model.freeze_encoders()
        p,_,_=self.model.decode(self.context,self.anchor,self.current,self.modes)
        p.square().mean().backward()
        self.assertGreater(float(self.model.mode_embedding.weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is None for p in self.model.geometry.parameters()))
        other=copy.deepcopy(self.model);other.load_state_dict(self.model.state_dict())
        q,_,_=other.decode(self.context,self.anchor,self.current,self.modes)
        torch.testing.assert_close(p,q,rtol=0,atol=0)

    def test_duplicate_modes_have_geometry_variants(self):
        p,_,_=self.model.decode(self.context,self.anchor,self.current,torch.zeros(2,8,dtype=torch.long))
        self.assertGreater(float((p[:,0]-p[:,1]).abs().max()),1e-6)

    def test_observation_only_allocation_both_samplers(self):
        for sampler in ('ordinary','balanced'):
            p,e,d=self.model.decode(self.context,self.anchor,self.current,sampling=sampler)
            self.assertEqual(p.shape,(2,8,24,3));self.assertEqual(d['mode_ids'].shape,(2,8))


if __name__=='__main__':unittest.main()
