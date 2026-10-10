import unittest
import numpy as np
from research_selective_repair_v1.local_loop_support import proposals

class LocalSupportTests(unittest.TestCase):
    def test_window_changes_only_three_interior_nodes(self):
        p=np.zeros((8,24,3),np.float32);p[:,:,0]=np.linspace(0,.45,24);p[:,:,2]=.55
        c=np.array([[.6,-.1,.4],[.6,.1,.4],[.8,-.1,.4],[.8,.1,.4]],np.float32)
        result,ok,_=proposals(p,c,.2,np.array([4]*8))
        # Missing forward crossings make the proposal invalid and both arms
        # must retain exact identity, even though its local displacement fits.
        self.assertFalse(ok.any());np.testing.assert_array_equal(result[:,2],p)
    def test_endpoint_failure_is_unknown_not_fabricated_repair(self):
        p=np.zeros((8,24,3),np.float32);p[:,:,0]=np.linspace(0,1,24);p[:,:,2]=.55
        c=np.array([[.6,-.1,.4],[.6,.1,.4],[.8,-.1,.4],[.8,.1,.4]],np.float32)
        result,ok,_=proposals(p,c,.2,np.array([22]*8))
        self.assertFalse(ok.any());np.testing.assert_array_equal(result[:,2],p)
        p[:,:6,0]=np.linspace(0,.075,6);p[:,6:,0]=np.linspace(.1,1,18)
        result,ok,_=proposals(p,c,.2,np.array([2]*8))
        self.assertTrue(ok.all())
        np.testing.assert_array_equal(result[:,2,:3],p[:,:3]);np.testing.assert_array_equal(result[:,2,6:],p[:,6:])
        self.assertLessEqual(float(np.linalg.norm(result[:,2]-p,axis=-1).max()),.0800001)

if __name__=='__main__':unittest.main()
