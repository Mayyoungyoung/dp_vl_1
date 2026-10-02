"""Standard v identities, exact default behavior, and parameterization resume."""
import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from routeset.diffusion import DiffusionSchedule
from routeset.diffusion_parameterization import ParameterizedDiffusionSchedule
from routeset.models import RouteDenoiser
from routeset.multigate import generate_dataset
from routeset.multigate_diffusion import guided_sample


class DiffusionParameterizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_oracle_v_reconstructs_x0_and_noise_at_both_extremes(self):
        torch.manual_seed(4)
        schedule = ParameterizedDiffusionSchedule(parameterization="v")
        clean, noise = torch.randn(2, 4, 22, 3), torch.randn(2, 4, 22, 3)
        times = torch.tensor([0, 99])
        noisy, _ = schedule.q_sample(clean, times, noise)
        velocity = schedule.training_target(clean, noise, times)
        x0, epsilon = schedule.prediction_to_x0_epsilon(noisy, times, velocity)
        torch.testing.assert_close(clean, x0, atol=5e-7, rtol=1e-6)
        torch.testing.assert_close(noise, epsilon, atol=5e-7, rtol=1e-6)

    def test_default_epsilon_is_bitwise_original_sampling_and_target(self):
        torch.manual_seed(7)
        original, default = DiffusionSchedule(), ParameterizedDiffusionSchedule()
        model = RouteDenoiser(width=16, depth=1, cond_dim=34)
        condition = torch.randn(2, 34)
        for k in (1, 2, 4, 8):
            a = original.sample(model, condition, k, 4, torch.Generator().manual_seed(17))
            b = default.sample(model, condition, k, 4, torch.Generator().manual_seed(17))
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        clean, noise = torch.randn(2, 4, 22, 3), torch.randn(2, 4, 22, 3)
        self.assertIs(default.training_target(clean, noise, torch.tensor([0, 99])), noise)

    def test_v_continuous_and_resumed_are_identical_and_cross_type_rejected(self):
        from scripts.train_multigate_diffusion import train_one
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = generate_dataset(root / "data.npz", train=2, dev_model=1,
                dev_score=0, calibration=0, test_locked=0, ood_locked=0)
            common = dict(data=str(data), arm="set_diffusion", parameterization="v", steps=4,
                batch_size=2, candidates=4, width=16, depth=1, lr=3e-4, seed=8,
                eval_every=2, diffusion_steps=100, sampling_steps=2, eval_candidates=[4],
                final_repeats=1, latency_requests=0, threads=1, device="cpu", resume=False, stop_after=None)
            train_one(Namespace(**common, output=str(root / "continuous")))
            interrupted = Namespace(**common, output=str(root / "resumed"))
            interrupted.stop_after = 2
            train_one(interrupted)
            interrupted.resume, interrupted.stop_after = True, None
            rejected = copy.copy(interrupted)
            rejected.parameterization = "epsilon"
            with self.assertRaisesRegex(ValueError, "Resume config mismatch"):
                train_one(rejected)
            train_one(interrupted)
            a = torch.load(root / "continuous/last.pt", weights_only=False)
            b = torch.load(root / "resumed/last.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["stream"]["digest"], b["stream"]["digest"])
            self.assertEqual(a["loss_tail"], b["loss_tail"])

    def test_nonzero_pg_rejects_v_without_breaking_zero_strength(self):
        torch.manual_seed(6)
        schedule = ParameterizedDiffusionSchedule(parameterization="v")
        model = RouteDenoiser(width=16, depth=1, cond_dim=34)
        condition = torch.randn(2, 34)
        with self.assertRaisesRegex(ValueError, "epsilon prediction only"):
            guided_sample(schedule, model, condition, 4, steps=2, strength=.1)
        original = schedule.sample(model, condition, 4, 2, torch.Generator().manual_seed(29))
        zero = guided_sample(schedule, model, condition, 4, 2, torch.Generator().manual_seed(29), strength=0.)
        torch.testing.assert_close(original, zero, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
