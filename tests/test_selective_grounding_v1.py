import unittest
import numpy as np
from scripts.observation_prototype_grounding import feature_colors
from research_selective_repair_v1.grounding import predict

class GroundingTests(unittest.TestCase):
    def test_observed_prior_rejects_remote_component_and_can_abstain(self):
        rgb=np.zeros((30,40,3));rgb[5:12,5:12]=[1.,0.,0.];rgb[15:22,25:32]=[1.,0.,0.]
        xy=np.stack(np.meshgrid(np.arange(40)*.002,np.arange(30)*.002),-1)
        xyz=np.concatenate([xy,np.ones((30,40,1))],-1);xyz[15:22,25:32,2]=2
        model=dict(prototypes={'reach red':dict(center=feature_colors([1.,0.,0.]).tolist(),scale=[.025]*6,threshold=1.,extent_m=.017)},spatial_prior=dict(lower=[0,0,.9],upper=[.1,.1,1.1]),surface_offsets={'reach red':[.005,0,0]})
        point,info=predict(rgb,xyz,np.ones((30,40),bool),'reach red',model)
        np.testing.assert_allclose(point,[.021,.016,1.],atol=1e-8)
        valid=np.zeros((30,40),bool);valid[15:22,25:32]=True
        point,info=predict(rgb,xyz,valid,'reach red',model)
        self.assertIsNone(point)
        self.assertEqual(info['status'],'no_supported_component')

if __name__=='__main__':unittest.main()
