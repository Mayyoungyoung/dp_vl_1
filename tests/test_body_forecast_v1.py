import unittest
import numpy as np
import torch
from research_selective_repair_v1.body_forecast import features,OutcomeHead

class ObservedExecutionForecast(unittest.TestCase):
    def test_joint_translation_invariance_and_sequence_gradient(self):
        rng=np.random.default_rng(42);p=rng.normal(size=(3,24,3)).astype(np.float32);c=rng.normal(size=(3,4,3)).astype(np.float32)
        x=features(p,c);shift=np.array([.14,-.05,.02],np.float32)
        np.testing.assert_allclose(x,features(p+shift,c+shift),atol=2e-5,rtol=1e-5)
        model=OutcomeHead();tx=torch.tensor(x,requires_grad=True);context=torch.zeros(3,128)
        value=model(tx,context);self.assertEqual(value.shape,(3,17));value.square().mean().backward()
        self.assertTrue(torch.isfinite(tx.grad).all());self.assertGreater(float(tx.grad[:,:6].abs().sum()),0)

if __name__=='__main__':unittest.main()
