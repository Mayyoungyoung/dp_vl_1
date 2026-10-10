import unittest
import numpy as np
from research_selective_repair_v1.prefix_conditioning import proposals

class PrefixConditioningTest(unittest.TestCase):
    def test_common_budget_fallback_and_fixed_suffix(self):
        p=np.zeros((8,24,3),np.float32);p[...,0]=np.linspace(0,.8,24);p[...,2]=.35
        completed=np.array([[.2,-.1,.25],[.2,.1,.25],[.5,-.1,.25],[.5,.1,.25]],np.float32)
        out,ok,reasons=proposals(p,completed,.15)
        self.assertFalse(ok.any()) # node3's original displacement exceeds8cm.
        np.testing.assert_array_equal(out,np.broadcast_to(p[:,None],out.shape))
    def test_loop_returns_public_root_without_changing_passages(self):
        p=np.zeros((8,24,3),np.float32);p[...,0]=np.r_[0,.01,.02,.03,np.linspace(.08,.8,20)];p[...,2]=.35
        completed=np.array([[.2,-.1,.25],[.2,.1,.25],[.5,-.1,.25],[.5,.1,.25]],np.float32)
        out,ok,_=proposals(p,completed,.15)
        self.assertTrue(ok.all());np.testing.assert_array_equal(out[:,2,3],p[:,0]);np.testing.assert_array_equal(out[:,:,4:],np.broadcast_to(p[:,None,4:],out[:,:,4:].shape))
        self.assertLessEqual(np.linalg.norm(out-p[:,None],axis=-1).max(),.08)

if __name__=='__main__':unittest.main()
