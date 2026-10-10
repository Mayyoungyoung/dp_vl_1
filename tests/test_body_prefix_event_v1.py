import unittest
import numpy as np
import torch
from research_selective_repair_v1.body_event_forecast import EventHead,likelihood,compose

class EventTests(unittest.TestCase):
    def test_unknown_suffix_has_zero_likelihood_gradient(self):
        model=EventHead();paths=torch.zeros(2,24,3)
        pred=model(torch.zeros(2,24,26),torch.zeros(2,128),paths)
        for key in ('hazard','events','mean'):pred[key].retain_grad()
        data=dict(prefix_hazard=torch.zeros(2,23),prefix_observed=torch.zeros(2,23,dtype=torch.bool),
            event_tip=torch.zeros(2,23,2,3),event_present=torch.zeros(2,23,2,dtype=torch.bool),
            event_observed=torch.zeros(2,23,2,dtype=torch.bool),prefix_valid=torch.zeros(2,23,dtype=torch.bool),labels=torch.zeros(2,dtype=torch.long))
        data['prefix_observed'][:,0]=True;data['event_observed'][:,0]=True
        likelihood(pred,data).backward()
        for key in ('hazard','events','mean'):
            self.assertTrue(torch.equal(pred[key].grad[:,:,1:],torch.zeros_like(pred[key].grad[:,:,1:])))
        self.assertTrue(torch.isfinite(pred['hazard'].grad).all())
        self.assertGreater(float(pred['hazard'].grad[:,:,:1].abs().sum()),0)

    def test_future_nodes_cannot_change_past_event(self):
        torch.manual_seed(102);model=EventHead().eval();x=torch.randn(2,24,26);paths=torch.randn(2,24,3);ctx=torch.randn(2,128)
        a=model(x,ctx,paths);changed=x.clone();changed[:,10:]+=100;newpaths=paths.clone();newpaths[:,10:]+=100
        b=model(changed,ctx,newpaths)
        for key in ('hazard','events','mean','scale'):
            self.assertTrue(torch.equal(a[key][:,:,:9],b[key][:,:,:9]),key)
        self.assertTrue(torch.equal(a['log_weights'],b['log_weights']))

    def test_integrated_mixture_preserves_joint_row_coupling(self):
        pred=dict(mean=np.zeros((1,2,23,2,2)),scale=np.full((1,2,23,2,2),.0001),
            events=np.full((1,2,23,2),-40.),hazard=np.full((1,2,23),-40.),clear=np.full((1,2),40.),log_weights=np.log(np.array([[.5,.5]])))
        pred['events'][:,:,0]=40;pred['mean'][:,0,...,0]=-.3;pred['mean'][:,1,...,0]=.3
        pred['mean'][...,1]=.1
        cfg=dict(row_x=[0.,1.],post_y=[[-.1,.1],[-.1,.1]],post_heights=[.3,.3],post_base_z=0.,tip_clearance_m=.01)
        prob,_=compose(pred,np.zeros((1,24,3)),cfg)
        self.assertAlmostEqual(float(prob[0,1]),.5,places=6)
        self.assertAlmostEqual(float(prob[0,11]),.5,places=6)
        self.assertAlmostEqual(float(prob[0,3]),0,places=6)
        self.assertAlmostEqual(float(prob.sum()),1,places=6)

    def test_censored_survival_matches_product(self):
        p=dict(hazard=torch.tensor([[[.3,-.7,9.]]],requires_grad=True),events=torch.zeros(1,1,3,2),
            mean=torch.zeros(1,1,3,2,2),scale=torch.ones(1,1,3,2,2),log_weights=torch.zeros(1,1),clear=torch.zeros(1,1))
        d=dict(prefix_hazard=torch.tensor([[0.,1.,0.]]),prefix_observed=torch.tensor([[True,True,False]]),
            event_tip=torch.zeros(1,3,2,3),event_present=torch.zeros(1,3,2,dtype=torch.bool),event_observed=torch.zeros(1,3,2,dtype=torch.bool),prefix_valid=torch.zeros(1,3,dtype=torch.bool),labels=torch.zeros(1,dtype=torch.long))
        loss=likelihood(p,d);expected=-torch.log(torch.sigmoid(-p['hazard'][0,0,0])*torch.sigmoid(p['hazard'][0,0,1]))
        self.assertTrue(torch.allclose(loss,expected));loss.backward();self.assertEqual(float(p['hazard'].grad[0,0,2]),0)

if __name__=='__main__':unittest.main()
