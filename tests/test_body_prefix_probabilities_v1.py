import unittest
import numpy as np
from research_selective_repair_v1.body_prefix_probabilities import compose

class ExactEventComposition(unittest.TestCase):
    def test_event_signature_survives_uniform_mesh_aliasing(self):
        path=np.zeros((1,24,3));path[0,:,0]=np.linspace(0,.6,24);path[0,:,2]=.7
        tip=np.zeros((1,1,23,4,3));tip[0,0]=np.broadcast_to(path[0,1:,None],(23,4,3))
        event=np.zeros((1,1,23,2,3));event[0,0,1,0]=[.2,-.3,.3];event[0,0,2,1]=[.4,0,.3]
        logits=np.full((1,1,23,2),-40.);logits[0,0,1,0]=40;logits[0,0,2,1]=40
        prediction=dict(tip=tip,event_tip=event,events=logits,hazard=np.full((1,1,23),-40.),log_weights=np.zeros((1,1)))
        cfg=dict(row_x=[.2,.4],post_y=[[-.1,.1],[-.1,.1]],post_heights=[.4,.4],post_base_z=0,tip_clearance_m=.02)
        probability,count=compose(prediction,path,cfg)
        self.assertGreater(probability[0,2],.9999) # gap0|gap1,not sparse over|over
        self.assertAlmostEqual(float(probability.sum()),1,places=6)
        self.assertEqual(count['actual_controller_queries'],0)

if __name__=='__main__':unittest.main()
