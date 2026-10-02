"""Padding-safe paired supervision, analytic potential, and real resume checks."""
import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from routeset.diffusion import DiffusionSchedule
from routeset.models import RouteDenoiser
from routeset.multigate import generate_dataset, load_dataset
from routeset.multigate_diffusion import PairedTrainingStream, guided_sample, load_development, rbf_repulsion


class MultigateDiffusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = generate_dataset(Path(directory) / "data.npz", train=20, dev_model=4,
                dev_score=0, calibration=0, test_locked=0, ood_locked=0)
            cls.data = load_dataset(path)
        cls.train_ids = np.flatnonzero(cls.data["splits"] == "TRAIN")

    def test_padding_safe_targets_and_actual_stream_identity(self):
        data = {key: value.copy() for key, value in self.data.items()}
        data["paths"][~data["path_mask"]] = np.nan
        a = PairedTrainingStream(data, self.train_ids, 18)
        b = PairedTrainingStream(data, self.train_ids, 18)
        for k in (1, 2, 4, 8):
            first, second = a.draw(32, k, 100), b.draw(32, k, 100)
            for x, y in zip(first, second):
                np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
            ids, slots, paths, _, _ = first
            self.assertTrue(np.isfinite(paths).all())
            for idx, chosen in zip(ids, slots):
                self.assertTrue(data["path_mask"][idx, chosen].all())
                modes = data["modes"][idx, chosen]
                self.assertTrue((modes >= 0).all())
                known = len(np.unique(data["modes"][idx, data["path_mask"][idx]]))
                self.assertEqual(len(np.unique(modes)), min(k, known))
            self.assertEqual(a.digest, b.digest)
        restored = PairedTrainingStream(data, self.train_ids, 1)
        restored.load_state_dict(a.state_dict())
        for x, y in zip(a.draw(2, 4, 100), restored.draw(2, 4, 100)):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        self.assertEqual(a.digest, restored.digest)

    def test_locked_and_parent_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.npz"
            data = {key: value.copy() for key, value in self.data.items()}
            data["splits"] = data["splits"].astype("U20")
            data["splits"][0] = "TEST_LOCKED"
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(ValueError, "locked"):
                load_development(path)
            data["splits"][0] = "TRAIN"
            data["parent_ids"][-1] = data["parent_ids"][0]
            np.savez_compressed(path, **data)
            with self.assertRaisesRegex(ValueError, "parent split leakage"):
                load_development(path)

    def test_repulsion_matches_autograd_and_permutation(self):
        torch.manual_seed(24)
        x = torch.randn(2, 4, 3, 3, dtype=torch.float64, requires_grad=True)
        alpha = torch.tensor([.2, .9], dtype=x.dtype)
        flat = x.flatten(start_dim=2)
        delta = flat[:, :, None] - flat[:, None, :]
        scale = 9 * (alpha[:, None, None] * .2 ** 2 + 1 - alpha[:, None, None])
        kernel = torch.exp(-delta.square().sum(-1) / (2 * scale))
        energy = torch.triu(kernel, diagonal=1).sum() / 3
        expected = -torch.autograd.grad(energy, x)[0]
        actual = rbf_repulsion(x, alpha)
        torch.testing.assert_close(actual, expected, atol=1e-13, rtol=1e-12)
        permutation = [2, 0, 3, 1]
        torch.testing.assert_close(rbf_repulsion(x[:, permutation], alpha), actual[:, permutation])
        for k in (1, 2, 4, 8):
            self.assertTrue(torch.isfinite(rbf_repulsion(torch.randn(2, k, 3, 3), alpha.float())).all())
        self.assertEqual(float(rbf_repulsion(x[:, :1], alpha).abs().sum()), 0.)

    def test_zero_guidance_and_k1_are_original_ddim_with_exact_budget(self):
        torch.manual_seed(9)
        model = RouteDenoiser(cond_dim=34, horizon=6, width=16, depth=1, set_attention=False)
        diffusion = DiffusionSchedule(10)
        condition = torch.randn(2, 34)
        calls = []
        hook = model.register_forward_hook(lambda *unused: calls.append(1))
        try:
            for k in (1, 2, 4, 8):
                original = diffusion.sample(model, condition, k, 3, torch.Generator().manual_seed(12))
                calls.clear()
                guided = guided_sample(diffusion, model, condition, k, 3,
                    torch.Generator().manual_seed(12), strength=2. if k == 1 else 0.)
                torch.testing.assert_close(original, guided, atol=0, rtol=0)
                self.assertEqual(len(calls), 3)
                self.assertEqual(guided.shape[1], k)
                self.assertTrue(model.training)
        finally:
            hook.remove()

    def test_continuous_resumed_and_paired_training_streams(self):
        from scripts.train_multigate_diffusion import train_one
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = [0, int(np.flatnonzero(self.data["splits"] == "DEV_MODEL")[0])]
            small = {key: value[selected] for key, value in self.data.items()}
            np.savez_compressed(root / "data.npz", **small)
            common = dict(data=str(root / "data.npz"), arm="set_diffusion", steps=4,
                batch_size=2, candidates=4, width=16, depth=1, lr=3e-4, seed=8,
                eval_every=2, diffusion_steps=10, sampling_steps=2, eval_candidates=[1, 2, 4, 8],
                final_repeats=3, latency_requests=0, threads=1, device="cpu", resume=False, stop_after=None)
            train_one(Namespace(**common, output=str(root / "continuous")))
            interrupted = Namespace(**common, output=str(root / "resumed"))
            interrupted.stop_after = 2
            train_one(interrupted)
            interrupted.resume, interrupted.stop_after = True, None
            train_one(interrupted)
            independent = Namespace(**common, output=str(root / "independent"))
            independent.arm = "independent"
            train_one(independent)
            a = torch.load(root / "continuous" / "last.pt", weights_only=False)
            b = torch.load(root / "resumed" / "last.pt", weights_only=False)
            c = torch.load(root / "independent" / "last.pt", weights_only=False)
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["trajectory_exposures"], 32)
            self.assertEqual(a["stream"]["digest"], b["stream"]["digest"])
            self.assertEqual(a["stream"]["digest"], c["stream"]["digest"])
            np.testing.assert_array_equal(a["stream"]["noise"], b["stream"]["noise"])
            self.assertEqual(a["loss_tail"], b["loss_tail"])
            for key, value in (("lr", 1e-4), ("data", str(root / "changed.npz"))):
                changed = {name: array.copy() for name, array in small.items()}
                changed["scenes"][0, 0] += .001
                np.savez_compressed(root / "changed.npz", **changed)
                rejected = copy.copy(interrupted)
                setattr(rejected, key, value)
                with self.assertRaisesRegex(ValueError, "Resume config mismatch"):
                    train_one(rejected)


if __name__ == "__main__":
    unittest.main()
