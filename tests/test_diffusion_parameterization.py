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

    def test_new_output_continuation_preserves_source_and_full_training_state(self):
        from routeset.common import sha256
        from routeset.multigate_diffusion import PairedTrainingStream, load_development
        from scripts.train_multigate_diffusion import train_one
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = generate_dataset(root / "data.npz", train=2, dev_model=1,
                dev_score=0, calibration=0, test_locked=0, ood_locked=0)
            common = dict(data=str(data), arm="set_diffusion", parameterization="v",
                batch_size=2, candidates=4, width=16, depth=1, lr=3e-4, seed=8,
                eval_every=2, diffusion_steps=100, sampling_steps=2, eval_candidates=[4],
                final_repeats=1, latency_requests=0, threads=1, device="cpu", resume=False, stop_after=None)
            train_one(Namespace(**common, steps=4, output=str(root / "continuous"), continue_from=None))
            train_one(Namespace(**common, steps=2, output=str(root / "source"), continue_from=None))
            original = {name: sha256(root / "source" / name)
                        for name in ("last.pt", "best.pt", "config.json", "summary.json", "history.json")}
            continuation = Namespace(**common, steps=4, output=str(root / "continued"),
                                     continue_from=str(root / "source/last.pt"))
            with np.load(data, allow_pickle=False) as archive:
                changed = {name: archive[name].copy() for name in archive.files}
            changed["scenes"][0, 0] += .001
            np.savez_compressed(root / "changed.npz", **changed)
            for key, value in (("lr", 1e-4), ("data", str(root / "changed.npz")), ("parameterization", "epsilon")):
                rejected = copy.copy(continuation)
                rejected.output = str(root / ("rejected_" + key))
                setattr(rejected, key, value)
                with self.assertRaisesRegex(ValueError, "Continuation config mismatch"):
                    train_one(rejected)
                self.assertFalse((Path(rejected.output) / "best.pt").exists())
            # Also exercise an interruption/resumption of the new output.
            continuation.stop_after = 3
            train_one(continuation)
            continuation.resume, continuation.stop_after = True, None
            train_one(continuation)
            a = torch.load(root / "continuous/last.pt", weights_only=False)
            b = torch.load(root / "continued/last.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertEqual(a["optimizer"]["param_groups"], b["optimizer"]["param_groups"])
            for parameter, state in a["optimizer"]["state"].items():
                for key, value in state.items():
                    torch.testing.assert_close(value, b["optimizer"]["state"][parameter][key], atol=0, rtol=0)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["stream"]["digest"], b["stream"]["digest"])
            source_checkpoint = torch.load(root / "source/last.pt", weights_only=False)
            dataset, train_ids, _ = load_development(data)
            segment = PairedTrainingStream(dataset, train_ids, common["seed"])
            segment.load_state_dict(source_checkpoint["stream"])
            segment.digest = "0" * 64
            for _ in range(2):
                segment.draw(common["batch_size"], common["candidates"], common["diffusion_steps"])
            self.assertEqual(b["incremental_stream_sha256"], segment.digest)
            self.assertNotEqual(b["incremental_stream_sha256"], b["stream"]["digest"])
            self.assertEqual(a["incremental_stream_sha256"], a["stream"]["digest"])
            torch.testing.assert_close(a["stream"]["noise"], b["stream"]["noise"], atol=0, rtol=0)
            torch.testing.assert_close(a["rng"]["torch"], b["rng"]["torch"], atol=0, rtol=0)
            self.assertEqual(a["loss_tail"], b["loss_tail"])
            self.assertEqual(b["trajectory_exposures"], 32)
            self.assertEqual(b["incremental_trajectory_exposures"], 16)
            self.assertEqual(b["cost_origin"]["prior_trajectory_exposures"], 16)
            self.assertGreater(b["cumulative_elapsed_s"], b["elapsed_s"])
            self.assertEqual(b["config"]["continuation"]["source_checkpoint_sha256"], original["last.pt"])
            for name, digest in original.items():
                self.assertEqual(sha256(root / "source" / name), digest)


if __name__ == "__main__":
    unittest.main()
