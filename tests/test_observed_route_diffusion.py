"""Information/output, shared initialization, and actual DDIM call contracts."""
import json
from pathlib import Path
import unittest

try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from routeset.observed_route_diffusion import (ObservedRouteDiffusion,ObservedX0Schedule,
        targets_to_state,decode_state,ORDINARY_INITIAL_SHA,ORDINARY_RNG_SHA)
    from routeset.diffusion import DiffusionSchedule


class PolicyTests(unittest.TestCase):
    def test_single_fixed_policy_budget(self):
        p = json.loads((Path(__file__).parents[1]/'configs/observed_two_row_diffusion_v1.json').read_text())
        self.assertEqual(p['steps']*p['batch_size']*p['candidates'],p['training_path_states'])
        self.assertEqual(p['steps']//p['eval_every'],48)
        self.assertEqual((p['parameterization'],p['sampling_steps'],p['eval_noise_seeds']),('x0',40,[300000,300001,300002]))
        self.assertFalse(p['extension_dev_raw_allowed'])


@unittest.skipIf(torch is None,'requires actual Torch; no mock substitutes')
class DiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.independent = ObservedRouteDiffusion(False)
        cls.joint = ObservedRouteDiffusion(True)
        g = torch.Generator().manual_seed(123)
        cls.inputs = dict(features=torch.randn((2,4096),generator=g),current=torch.randn((2,8),generator=g),
            world_xyz=torch.randn((2,11,3),generator=g),rgb=torch.rand((2,11,3),generator=g),
            uv=torch.randn((2,11,2),generator=g),depth=torch.rand((2,11),generator=g),
            valid_mask=torch.ones(2,11,dtype=torch.bool))
        cls.inputs['current'][:,7] = 1
        cls.noise = torch.randn((2,4,23,4),generator=g)
        cls.times = torch.tensor([0,99],dtype=torch.long)

    def test_historical_and_pair_initialization(self):
        audit = self.independent.initialization_audit
        self.assertEqual(audit['ordinary_initial_model_sha256'],ORDINARY_INITIAL_SHA)
        self.assertEqual(audit['ordinary_initial_torch_cpu_rng_sha256'],ORDINARY_RNG_SHA)
        self.assertEqual(audit['shared_tensor_sha256'],self.joint.initialization_audit['shared_tensor_sha256'])
        self.assertGreater(audit['shared_tensor_count'],0)
        self.assertGreater(audit['inactive_parameters'],0)
        self.assertEqual(self.joint.initialization_audit['inactive_parameters'],0)
        for key,value in self.independent.state_dict().items():
            self.assertTrue(torch.equal(value,self.joint.state_dict()[key]),key)

    def test_construction_preserves_external_rng(self):
        before = torch.get_rng_state().clone()
        ObservedRouteDiffusion(False)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        with self.assertRaises(ValueError):ObservedRouteDiffusion(True,k=8)

    def test_strict_observation_whitelist(self):
        for forbidden in ('target_xyz','paths','modes','geometry','semantic_targets'):
            with self.assertRaises(ValueError):
                self.independent.encode_observation(dict(self.inputs,**{forbidden:torch.zeros(2,3)}))
        missing = dict(self.inputs);missing.pop('depth')
        with self.assertRaises(ValueError):self.independent.encode_observation(missing)

    def test_independent_isolation_and_set_permutation(self):
        with torch.no_grad():
            encoded = self.independent.encode_observation(self.inputs)
            first = self.independent.forward_x0(self.noise,self.times,encoded)
            changed = self.noise.clone();changed[:,0] += 5
            second = self.independent.forward_x0(changed,self.times,encoded)
            self.assertTrue(torch.equal(first[:,1:],second[:,1:]))
            self.assertFalse(torch.equal(first[:,0],second[:,0]))
            encoded = self.joint.encode_observation(self.inputs)
            first = self.joint.forward_x0(self.noise,self.times,encoded)
            permutation = [3,0,2,1]
            second = self.joint.forward_x0(self.noise[:,permutation],self.times,encoded)
            self.assertTrue(torch.allclose(first[:,permutation],second,atol=2e-6,rtol=2e-6))

    def test_start_endpoint_events_and_noisy_terminal_are_used(self):
        encoded = self.joint.encode_observation(self.inputs)
        clean = self.joint.forward_x0(self.noise,self.times,encoded)
        paths,opened = decode_state(clean,self.inputs['current'])
        self.assertTrue(torch.equal(paths[:,:,0],self.inputs['current'][:,None,:3].expand(-1,4,-1)))
        self.assertTrue(torch.equal(opened[:,:,0],self.inputs['current'][:,None,7].expand(-1,4)))
        self.assertLessEqual(float((paths[:,:,-1]-encoded['anchor_xyz'][:,None]).abs().max()),.050001)
        self.assertTrue(bool(((opened>=0)&(opened<=1)).all()))
        for channel in (0,3):
            changed = self.noise.clone();changed[:,:,-1,channel] += 10
            self.assertFalse(torch.equal(clean,self.joint.forward_x0(changed,self.times,encoded)))

    def test_all_active_modules_have_gradient(self):
        for model in (self.independent,self.joint):
            model.zero_grad(set_to_none=True)
            clean = model(self.noise,self.times,model.encode_observation(self.inputs))
            (clean-self.noise).square().mean().backward()
            for name,parameter in model.named_parameters():
                if parameter.requires_grad:self.assertIsNotNone(parameter.grad,name)
                else:self.assertIsNone(parameter.grad,name)

    def test_target_conversion_keeps_terminal_and_event(self):
        paths = torch.randn(2,4,24,3)
        events = torch.rand(2,4,24)
        paths[:,:,0] = self.inputs['current'][:,None,:3]
        events[:,:,0] = self.inputs['current'][:,None,7]
        clean = targets_to_state(paths,events,self.inputs['current'])
        restored,event = decode_state(clean,self.inputs['current'])
        self.assertTrue(torch.allclose(restored,paths,atol=3e-7,rtol=1e-6))
        self.assertTrue(torch.allclose(event,events,atol=1e-7,rtol=1e-6))
        changed = paths.clone();changed[:,:,-1,0]+=1
        self.assertFalse(torch.equal(clean,targets_to_state(changed,events,self.inputs['current'])))

    def test_schedule_matches_historical_numeric_betas_and_oracle(self):
        schedule = ObservedX0Schedule()
        historical = DiffusionSchedule(100)
        self.assertTrue(torch.equal(schedule.betas,historical.betas))
        self.assertTrue(torch.equal(schedule.alpha_bars,historical.alpha_bars))
        x0 = self.noise*.1
        noised = schedule.q_sample(x0,self.times,self.noise)
        clean = schedule.ddim_step(noised,x0,self.times,torch.tensor([-1,-1]))
        self.assertTrue(torch.equal(clean,x0))
        t = torch.tensor([99,45]);prev = torch.tensor([90,30])
        advanced = schedule.ddim_step(schedule.q_sample(x0,t,self.noise),x0,t,prev)
        expected = schedule.q_sample(x0,prev,self.noise)
        self.assertTrue(torch.allclose(advanced,expected,atol=2e-7,rtol=1e-6))

    def test_sampler_exactly40_and_mode_restoration(self):
        x0 = self.noise*.01
        class Oracle(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=[]
            def forward_x0(self,noisy,t,encoded):self.calls.append(t.tolist());return x0
        model = Oracle();model.train()
        schedule = ObservedX0Schedule()
        encoded = self.joint.encode_observation(self.inputs)
        issued=[]
        paths,opened,receipt = schedule.sample(model,encoded,self.noise,lambda i,t:issued.append((i,t)))
        expected_paths,expected_open = decode_state(x0,self.inputs['current'])
        self.assertTrue(torch.equal(paths,expected_paths));self.assertTrue(torch.equal(opened,expected_open))
        self.assertEqual(len(model.calls),40);self.assertEqual(len(issued),40)
        self.assertEqual(issued[-1],(39,0));self.assertEqual(receipt['denoiser_calls'],40)
        self.assertEqual(receipt['requested_candidates'],8);self.assertTrue(model.training)
        self.assertEqual(receipt['observation_encodes_inside_sampler'],0)

    def test_sampler_callback_failure_is_not_retried(self):
        class Counting(torch.nn.Module):
            def __init__(self):super().__init__();self.calls=0
            def forward_x0(self,noisy,t,encoded):self.calls+=1;return noisy*0
        model=Counting();model.train()
        def fail(index,time):
            if index==3:raise RuntimeError('journal failure')
        with self.assertRaisesRegex(RuntimeError,'journal'):
            ObservedX0Schedule().sample(model,self.joint.encode_observation(self.inputs),self.noise,fail)
        self.assertEqual(model.calls,3);self.assertTrue(model.training)


if __name__=='__main__':unittest.main()
