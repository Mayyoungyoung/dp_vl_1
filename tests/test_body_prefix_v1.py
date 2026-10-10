import unittest
import numpy as np
from research_selective_repair_v1.body_prefix_data import prefix

class PrefixLabels(unittest.TestCase):
    def test_failed_suffix_is_unknown(self):
        q=np.arange(7*7).reshape(7,7);p=q[:,:3]
        record={'planning_segments':[dict(segment=0,simulated_steps=4,simulation_status='success'),dict(segment=1,simulated_steps=2,simulation_status='failed')]}
        a,b,valid,hazard,observed=prefix(record,q,p)
        self.assertEqual(valid.sum(),1);self.assertEqual(observed.sum(),2)
        self.assertEqual(hazard[1],1);self.assertEqual(hazard[2:].sum(),0)
        np.testing.assert_array_equal(a[0],q[1:5]);self.assertEqual(a[1:].sum(),0)

if __name__=='__main__':unittest.main()
