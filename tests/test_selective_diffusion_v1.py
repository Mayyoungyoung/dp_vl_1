import unittest
import torch
from research_selective_repair_v1.patch_diffusion import LocalDiffusion

class DiffusionControlTests(unittest.TestCase):
    def test_no_current_trigger_gives_identity_and_skips_diffusion(self):
        torch.set_num_threads(4);model=LocalDiffusion(64).eval();c=torch.zeros(1,64);m=torch.arange(8)[None];d=torch.randn(1,8,24,3);local=torch.zeros(1,8,24,33);local[...,12:16]=2
        p,info=model(c,m,d,local,threshold=.025)
        self.assertTrue(torch.equal(p,d));self.assertEqual(model.last_sampling_steps,0)
    def test_local_mask_preserves_start_and_exterior(self):
        torch.set_num_threads(4);model=LocalDiffusion(64).eval();c=torch.zeros(1,64);m=torch.arange(8)[None];d=torch.randn(1,8,24,3);local=torch.zeros(1,8,24,33);local[...,12:16]=2;local[:,0,5:8,12:16]=.3
        with torch.no_grad():p,info=model(c,m,d,local,threshold=.025)
        self.assertEqual(model.last_sampling_steps,12);self.assertTrue(torch.equal(p[:,:,0],d[:,:,0]));self.assertTrue(torch.equal(p[info['support']==0],d[info['support']==0]));self.assertLessEqual(float((p[:,:,:18]-d[:,:,:18]).abs().max()),.080001)

if __name__=='__main__':unittest.main()
