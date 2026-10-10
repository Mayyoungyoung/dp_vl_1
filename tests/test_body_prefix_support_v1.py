import unittest
import torch
from research_selective_repair_v1.body_spatial_repair import support_from_stop

class SupportTests(unittest.TestCase):
    def test_support_covers_likely_first_stop_and_excludes_endpoints(self):
        hazard=torch.full((2,4,23),-30.);hazard[0,:,4]=30.;hazard[1,:,19]=30.
        mask,stop=support_from_stop(dict(hazard=hazard,log_weights=torch.full((2,4),-torch.log(torch.tensor(4.)))))
        self.assertEqual(mask.sum(-1).tolist(),[12.,12.]);self.assertFalse(mask[:,[0,-1]].any())
        self.assertTrue(mask[0,4] or mask[0,5]);self.assertTrue(mask[1,19] or mask[1,20]);self.assertEqual(stop.argmax(-1).tolist(),[4,19])

    def test_first_stop_survival_excludes_later_high_hazard(self):
        hazard=torch.full((1,1,23),-30.);hazard[:,:,2]=30.;hazard[:,:,15]=30.
        _,stop=support_from_stop(dict(hazard=hazard,log_weights=torch.zeros(1,1)))
        self.assertGreater(float(stop[0,2]),.9999);self.assertLess(float(stop[0,15]),1e-6)

if __name__=='__main__':unittest.main()
