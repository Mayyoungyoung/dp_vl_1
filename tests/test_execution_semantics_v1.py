import unittest
import numpy as np
from research_selective_repair_v1.execution_semantics import tip_clear

class ActualTraceClearance(unittest.TestCase):
    def test_full_four_post_trace_and_floor(self):
        cfg=dict(row_x=[.15,.35],post_y=[[-.1,.1],[-.1,.1]],post_heights=[.1,.1],post_base_z=.755,tip_clearance_m=.02)
        x=np.linspace(0,.5,40)
        self.assertTrue(tip_clear(np.c_[x,np.full(40,-.3),np.full(40,.82)],cfg))
        self.assertFalse(tip_clear(np.c_[x,np.full(40,-.1),np.full(40,.82)],cfg))
        self.assertFalse(tip_clear(np.c_[x,np.full(40,-.3),np.full(40,.76)],cfg))

if __name__=='__main__':unittest.main()
