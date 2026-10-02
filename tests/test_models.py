"""Numerical architecture checks; run with python -m unittest discover -s tests."""

import unittest

import torch

from routeset.diffusion import DiffusionSchedule
from routeset.models import Critic, RouteDenoiser, SetRegressor


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(73)
        torch.set_num_threads(1)
        self.cond = torch.randn(2, 17)
        self.paths = torch.randn(2, 3, 10, 3)
        self.times = torch.tensor([0.13, 0.77])

    def denoiser(self, communicate=True):
        return RouteDenoiser(cond_dim=17, horizon=12, width=32, depth=2, heads=4,
                             set_attention=communicate)

    def test_set_denoiser_is_permutation_equivariant(self):
        model = self.denoiser().eval()
        permutation = torch.tensor([2, 0, 1])
        ordinary = model(self.paths, self.times, self.cond)
        permuted = model(self.paths[:, permutation], self.times, self.cond)
        torch.testing.assert_close(permuted, ordinary[:, permutation], atol=2e-6, rtol=2e-5)

    def test_independent_routes_do_not_communicate(self):
        model = self.denoiser(False).eval()
        ordinary = model(self.paths, self.times, self.cond)
        changed = self.paths.clone()
        changed[:, 1:] = torch.randn_like(changed[:, 1:]) * 50
        result = model(changed, self.times, self.cond)
        torch.testing.assert_close(result[:, 0], ordinary[:, 0], atol=0, rtol=0)
        single = model(self.paths[:, :1], self.times, self.cond)
        torch.testing.assert_close(single[:, 0], ordinary[:, 0], atol=2e-6, rtol=2e-5)

    def test_ablation_parameter_accounting_and_gradients(self):
        independent, joint = self.denoiser(False), self.denoiser(True)
        self.assertEqual(sum(p.numel() for p in independent.parameters()),
                         sum(p.numel() for p in joint.parameters()))
        self.assertLess(independent.active_parameter_count(), joint.active_parameter_count())
        independent(self.paths, self.times, self.cond).square().mean().backward()
        for block in independent.blocks:
            self.assertTrue(all(p.grad is None for p in block.attention.parameters()))
            self.assertTrue(all(p.grad is None for p in block.attention_norm.parameters()))
        joint(self.paths, self.times, self.cond).square().mean().backward()
        self.assertTrue(all(p.grad is not None for p in joint.blocks[0].attention.parameters()))

    def test_schedule_roundtrip(self):
        schedule = DiffusionSchedule(steps=100, clip_x0=None)
        times = torch.tensor([0, 90])
        noisy, epsilon = schedule.q_sample(self.paths, times)
        restored = schedule.x0_from_eps(noisy, times, epsilon)
        torch.testing.assert_close(restored, self.paths, atol=2e-6, rtol=2e-5)
        self.assertTrue(torch.all(schedule.alpha_bars[1:] < schedule.alpha_bars[:-1]))
        self.assertTrue(torch.all(schedule.betas > 0))
        self.assertTrue(torch.all(schedule.betas < 1))

    def test_ddim_reproducible_finite_and_restores_mode(self):
        model = self.denoiser().train()
        schedule = DiffusionSchedule()
        first = schedule.sample(model, self.cond, k=4, steps=8,
                                generator=torch.Generator().manual_seed(19))
        second = schedule.sample(model, self.cond, k=4, steps=8,
                                 generator=torch.Generator().manual_seed(19))
        torch.testing.assert_close(first, second, atol=0, rtol=0)
        self.assertEqual(tuple(first.shape), (2, 4, 10, 3))
        self.assertTrue(torch.isfinite(first).all())
        self.assertTrue(model.training)
        self.assertLessEqual(float(first.abs().max()), 2.0)

    def test_query_regressor_and_critic_shapes_and_gradients(self):
        regressor = SetRegressor(cond_dim=17, horizon=12, max_candidates=4,
                                 width=32, depth=2, heads=4)
        prediction = regressor(self.cond, k=3)
        self.assertEqual(tuple(prediction.shape), tuple(self.paths.shape))
        (prediction - self.paths).square().mean().backward()
        self.assertGreater(float(regressor.queries.grad[:3].norm()), 0)
        self.assertTrue(torch.isfinite(regressor.queries.grad).all())
        self.assertTrue(torch.all(regressor.queries.grad[3] == 0))
        critic = Critic(cond_dim=17, horizon=12, width=32)
        logits = critic(prediction.detach(), self.cond)
        self.assertEqual(tuple(logits.shape), (2, 3))
        torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in critic.parameters()))

    def test_invalid_model_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            self.denoiser()(self.paths[:, :, :-1], self.times, self.cond)
        with self.assertRaises(ValueError):
            self.denoiser()(self.paths, self.times, self.cond[:, :-1])
        with self.assertRaises(ValueError):
            SetRegressor(max_candidates=3)(torch.randn(2, 12), k=4)
        with self.assertRaises(ValueError):
            DiffusionSchedule(steps=1)


if __name__ == '__main__':
    unittest.main()
