import unittest
import numpy as np
from research_selective_repair_v1.body_prefix_data import prefix,crossing_events

class PrefixLabels(unittest.TestCase):
    def test_failed_suffix_is_unknown(self):
        q=np.arange(7*7).reshape(7,7);p=q[:,:3]
        record={'planning_segments':[dict(segment=0,simulated_steps=4,simulation_status='success'),dict(segment=1,simulated_steps=2,simulation_status='failed')]}
        a,b,valid,hazard,observed=prefix(record,q,p)
        self.assertEqual(valid.sum(),1);self.assertEqual(observed.sum(),2)
        self.assertEqual(hazard[1],1);self.assertEqual(hazard[2:].sum(),0)
        np.testing.assert_array_equal(a[0],q[1:5]);self.assertEqual(a[1:].sum(),0)

    def test_exact_crossing_inside_failed_prefix(self):
        tips=np.array([[0,0,0],[.2,1,2],[.4,2,4],[.6,3,6]],float);q=np.repeat(tips[:,:1],7,axis=1)
        record={'planning_segments':[dict(segment=0,simulated_steps=2),dict(segment=1,simulated_steps=1),dict(segment=2,simulated_steps=0)]}
        a,b,present,observed=crossing_events(record,q,tips,[.3,.5])
        self.assertTrue(present[0,0]);self.assertTrue(present[1,1])
        np.testing.assert_allclose(b[0,0],[.3,1.5,3]);np.testing.assert_allclose(a[1,1],.5)
        np.testing.assert_array_equal(observed[:,0],[True]+[False]*22)
        np.testing.assert_array_equal(observed[:,1],[True,True]+[False]*21)

if __name__=='__main__':unittest.main()
