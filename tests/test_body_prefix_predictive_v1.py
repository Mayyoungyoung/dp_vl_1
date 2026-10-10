import unittest
import numpy as np
import torch
from research_selective_repair_v1.body_event_forecast import compose
from research_selective_repair_v1.body_forecast import features
from research_selective_repair_v1.body_predictive_repair import compose_tensor,tensor_features,amplitude_caps
from research_selective_repair_v1.execution_semantics import word

class PredictiveTests(unittest.TestCase):
    def test_tensor_feature_contract_and_gradient(self):
        rng=np.random.default_rng(9);path=rng.normal(size=(8,24,3)).astype('float32');c=rng.normal(size=(8,4,3)).astype('float32')
        p=torch.tensor(path,requires_grad=True);t=tensor_features(p,torch.tensor(c),prefix=False)
        np.testing.assert_allclose(t.detach().numpy(),features(path,c),rtol=2e-6,atol=2e-6)
        t.square().sum().backward();self.assertTrue(torch.isfinite(p.grad).all())

    def test_gaussian_composition_matches_numpy_and_differentiates(self):
        rng=np.random.default_rng(19)
        p=dict(mean=rng.normal(0,.1,size=(8,4,23,2,2)),scale=np.full((8,4,23,2,2),.05),events=rng.normal(-2,1,size=(8,4,23,2)),
            hazard=np.full((8,4,23),-4.),clear=np.full((8,4),3.),log_weights=np.log(np.full((8,4),.25)))
        cfg=dict(row_x=[0.,1.],post_y=[[-.1,.1],[-.1,.1]],post_heights=[.3,.3],post_base_z=0.,tip_clearance_m=.01)
        expected,_=compose(p,np.zeros((8,24,3)),cfg)
        pt={k:torch.tensor(v,requires_grad=True) for k,v in p.items()};actual=compose_tensor(pt,cfg)
        np.testing.assert_allclose(actual.detach().numpy(),expected,rtol=1e-6,atol=1e-7)
        (actual[:,1:]*torch.arange(16,dtype=actual.dtype)).sum().backward()
        for key in ('mean','scale','events','hazard','clear','log_weights'):self.assertTrue(torch.isfinite(pt[key].grad).all(),key)

    def test_caps_keep_first_cross_word_and_endpoints(self):
        path=np.zeros((8,24,3),np.float32);path[...,0]=np.linspace(-.3,.3,24);path[...,2]=.28
        c=np.array([[-.1,-.1,.3],[-.1,.1,.3],[.1,-.1,.3],[.1,.1,.3]])
        cfg=dict(row_x=[-.1,.1],post_y=[[-.1,.1],[-.1,.1]],post_heights=[.3,.3],post_base_z=0.,tip_clearance_m=.01)
        basis,cap,_=amplitude_caps(path,c,cfg)
        for alpha,beta in ((0,0),(1,0),(0,1),(.4,.6)):
            a=np.minimum(alpha,cap);b=np.minimum(beta,1-a)
            p=path+a[:,None,None]*basis[:,0]+b[:,None,None]*basis[:,1]
            self.assertEqual([word(v,cfg) for v in p],[word(v,cfg) for v in path])
            self.assertTrue(np.array_equal(p[:,[0,-1]],path[:,[0,-1]]))

if __name__=='__main__':unittest.main()
