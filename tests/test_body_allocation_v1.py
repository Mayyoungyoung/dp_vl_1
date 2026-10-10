import unittest
import numpy as np
from research_selective_repair_v1.body_allocation import allocate,coverage
from research_selective_repair_v1.body_options import options

class FiniteBodyCorrection(unittest.TestCase):
    def test_word_filter_blocks_only_unsafe_action_and_is_shared(self):
        p=np.zeros((8,3,17));p[:,:,0]=1;p[:,1,0]=0;p[:,1,16]=1
        allowed=np.ones((8,3),bool);allowed[::2,1]=False
        for kind in ('lift','success','actual','coordinate'):
            choice,_=allocate(p,kind,protected=False,allowed=allowed)
            self.assertTrue(allowed[np.arange(8),choice].all())
    def test_success_only_loses_diversity_actual_forecast_recovers(self):
        p=np.zeros((8,3,17));p[:,:,0]=1
        # Original slots0--3 preserve four different successful modes.
        for slot in range(4):p[slot,0,0]=.1;p[slot,0,slot+1]=.9
        # Uniform lift succeeds, but all eight execute the same mode.
        p[:,1,0]=0;p[:,1,16]=1
        # Failed original slots can each repair to an additional distinct mode.
        for slot in range(4,8):p[slot,2,0]=.05;p[slot,2,slot+1]=.95
        simple,_=allocate(p,'success');full,queries=allocate(p,'actual')
        self.assertLess(coverage(p[np.arange(8),simple]).sum(),2)
        self.assertGreater(coverage(p[np.arange(8),full]).sum(),7)
        np.testing.assert_array_equal(full[:4],0)
        self.assertEqual(queries,6561)

    def test_zero_opportunity_stays_identity_and_endpoints_are_exact(self):
        p=np.zeros((8,3,17));p[:,:,0]=1
        choice,_=allocate(p);np.testing.assert_array_equal(choice,0)
        x=np.linspace(0,.5,24);path=np.broadcast_to(np.c_[x,x*0,np.full(24,.82)],(8,24,3)).copy()
        completed=np.array([[.15,-.1,.88],[.15,.1,.88],[.35,-.1,.88],[.35,.1,.88]])
        v=options(path,completed)
        np.testing.assert_array_equal(v[:,:,0],np.broadcast_to(path[:,None,0],(8,3,3)))
        np.testing.assert_array_equal(v[:,:,-1],np.broadcast_to(path[:,None,-1],(8,3,3)))
        # A real crossing window is kept, not a requested-mode substitution.
        for row in (.15,.35):
            q=int(np.flatnonzero((x[:-1]<row)&(x[1:]>=row))[0])
            np.testing.assert_array_equal(v[:,2,q-1:q+3],path[:,q-1:q+3])
        self.assertLessEqual(float(np.abs(v-path[:,None]).max()),.08000001)

if __name__=='__main__':unittest.main()
