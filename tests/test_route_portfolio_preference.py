import unittest
import torch
from routeset.mode_geometry import ModeGeometryHead
from research_route_portfolio_v1.planner import allocate_preference


class PreferenceContractTests(unittest.TestCase):
    def test_all_allowed_reproduces_original_decode(self):
        torch.manual_seed(40)
        model=ModeGeometryHead(feature_dim=12,width=16,point_width=8,horizon=24,max_candidates=8).eval()
        context=torch.randn(3,16);anchor=torch.randn(3,3);current=torch.randn(3,8)
        with torch.no_grad():
            logits=model.mode_predictor(context)
            modes=allocate_preference(logits,torch.ones_like(logits,dtype=torch.bool))
            a,ae,info=model.decode(context,anchor,current,sampling='adaptive')
            b,be,_=model.decode(context,anchor,current,modes)
        torch.testing.assert_close(modes,info['mode_ids'],rtol=0,atol=0)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        torch.testing.assert_close(ae,be,rtol=0,atol=0)

    def test_mask_excludes_high_score_and_keeps_negative_legal_fallback(self):
        logits=torch.arange(16,dtype=torch.float32)[None]
        allowed=torch.zeros_like(logits,dtype=torch.bool);allowed[:,1]=True;allowed[:,2]=True
        modes=allocate_preference(logits,allowed)
        self.assertEqual(set(modes[0].tolist()),{1,2})
        modes=allocate_preference(logits-30,allowed)
        self.assertTrue(torch.all(modes==2))

    def test_one_requested_mode_uses_all_eight_variants(self):
        logits=torch.zeros(1,16);allowed=torch.zeros_like(logits,dtype=torch.bool);allowed[:,4]=True
        modes=allocate_preference(logits,allowed)
        counts=torch.stack([(modes[:,:j]==modes[:,j:j+1]).sum(-1) for j in range(8)],1)
        self.assertEqual(counts.tolist(),[list(range(8))])

    def test_invalid_preferences_rejected(self):
        logits=torch.zeros(1,16)
        for allowed in (torch.zeros_like(logits,dtype=torch.bool),torch.ones(16,dtype=torch.bool),torch.ones_like(logits)):
            with self.assertRaises(ValueError):allocate_preference(logits,allowed)


if __name__=='__main__':unittest.main()
