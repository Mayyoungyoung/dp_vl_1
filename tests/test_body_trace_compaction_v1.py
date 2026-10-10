import unittest
import numpy as np
from research_selective_repair_v1.execution_trace_compaction import compact
from research_selective_repair_v1.execution_semantics import word,tip_clear

class CompactionTests(unittest.TestCase):
    def test_exact_firstcross_anchors_preserve_word_and_endpoints(self):
        cfg=dict(row_x=[0.,1.],post_y=[[-.1,.1],[-.1,.1]],post_heights=[.3,.3],post_base_z=0.,tip_clearance_m=.02)
        x=np.linspace(-.5,1.5,1001);path=np.stack([x,np.zeros_like(x),.2+.25/(1+np.exp(-20*(x-.5)))],-1)
        result,report=compact(path,cfg,path[0],path[-1])
        self.assertTrue(tip_clear(path,cfg));self.assertTrue(report['tip_clear']);self.assertEqual(word(result,cfg),word(path,cfg))
        self.assertEqual(report['mandatory_firstcross_anchors'],2);np.testing.assert_array_equal(result[[0,-1]],path[[0,-1]])
        self.assertEqual(result.shape,(24,3))

    def test_known_loop_cannot_relabel_first_crossing_as_later_over(self):
        cfg=dict(row_x=[0.,1.],post_y=[[-.1,.1],[-.1,.1]],post_heights=[.3,.3],post_base_z=0.,tip_clearance_m=.02)
        nodes=np.array([[-.5,0,.2],[.25,0,.2],[-.25,0,.45],[1.5,0,.45]])
        path=np.concatenate([a+np.linspace(0,1,100)[:,None]*(b-a) for a,b in zip(nodes[:-1],nodes[1:])])
        result,report=compact(path,cfg,path[0],path[-1])
        self.assertEqual(word(path,cfg),'gap1|over');self.assertEqual(word(result,cfg),'gap1|over');self.assertTrue(report['tip_clear'])

if __name__=='__main__':unittest.main()
