import unittest
import torch
from research_selective_repair_v1.body_crossing_measure import CrossingHead,likelihood,compose_tensor,anchors

class CrossingTests(unittest.TestCase):
    def test_unknown_crossing_and_suffix_do_not_train_unobserved_targets(self):
        model=CrossingHead();paths=torch.zeros(2,24,3);posts=torch.zeros(2,4,3)
        p=model(torch.zeros(2,24,26),torch.zeros(2,128),paths,posts)
        p['mean'].retain_grad();p['hazard'].retain_grad()
        d=dict(prefix_hazard=torch.zeros(2,23),prefix_observed=torch.zeros(2,23,dtype=torch.bool),
            event_tip=torch.zeros(2,23,2,3),event_present=torch.zeros(2,23,2,dtype=torch.bool),
            prefix_valid=torch.zeros(2,23,dtype=torch.bool),labels=torch.zeros(2,dtype=torch.long))
        d['prefix_observed'][:,0]=True;d['event_present'][:,0,0]=True
        a=likelihood(p,d);changed={k:v.clone() for k,v in d.items()};changed['event_tip'][:,:,1]=1000
        self.assertTrue(torch.equal(a,likelihood(p,changed)))
        a.backward();self.assertTrue(torch.equal(p['mean'].grad[:,:,1],torch.zeros_like(p['mean'].grad[:,:,1])))
        self.assertTrue(torch.equal(p['hazard'].grad[:,:,1:],torch.zeros_like(p['hazard'].grad[:,:,1:])))

    def test_anchor_interpolation_and_joint_component_mass(self):
        paths=torch.zeros(1,24,3);paths[0,:,0]=torch.linspace(-1,1,24);paths[0,:,1]=2*paths[0,:,0]
        posts=torch.zeros(1,4,3);posts[0,:,0]=torch.tensor([-.5,-.5,.5,.5])
        point,valid=anchors(paths,posts);torch.testing.assert_close(point[0,:,0],torch.tensor([-1.,1.]));self.assertTrue(valid.all())
        p=dict(mean=torch.zeros(1,2,2,2),scale=torch.full((1,2,2,2),.0001),hazard=torch.full((1,2,23),-40.),clear=torch.full((1,2),40.),log_weights=torch.log(torch.tensor([[.5,.5]])),has_requested_crossings=valid)
        p['mean'][:,0,:,0]=-.3;p['mean'][:,1,:,0]=.3;p['mean'][...,1]=.1
        cfg=dict(post_y=[[-.1,.1],[-.1,.1]],post_heights=[.3,.3],post_base_z=0.,tip_clearance_m=.01)
        mass=compose_tensor(p,cfg);self.assertAlmostEqual(float(mass[0,1]),.5,places=6);self.assertAlmostEqual(float(mass[0,11]),.5,places=6);self.assertAlmostEqual(float(mass[0,3]),0,places=6)
        p['mean'].requires_grad_(True);compose_tensor(p,cfg)[:,1:].sum().backward();self.assertTrue(torch.isfinite(p['mean'].grad).all())

    def test_categorical_control_receives_auxiliary_gradients(self):
        model=CrossingHead(readout='categorical_aux');p=model(torch.zeros(2,24,26),torch.zeros(2,128),torch.zeros(2,24,3),torch.zeros(2,4,3))
        d=dict(prefix_hazard=torch.zeros(2,23),prefix_observed=torch.ones(2,23,dtype=torch.bool),event_tip=torch.zeros(2,23,2,3),event_present=torch.ones(2,23,2,dtype=torch.bool),prefix_valid=torch.ones(2,23,dtype=torch.bool),labels=torch.ones(2,dtype=torch.long))
        likelihood(p,d).backward()
        for layer in (model.positions[-1],model.hazard,model.category[-1]):self.assertGreater(float(layer.weight.grad.abs().sum()),0)

if __name__=='__main__':unittest.main()
