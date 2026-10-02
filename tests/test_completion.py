"""Checks target leakage boundaries, invalid gating and exact max idempotence."""

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from routeset.completion import CompletionRegressor, build_contexts
from routeset.multigate import generate_dataset, load_dataset, route_modes


class CompletionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = generate_dataset(Path(directory) / "data.npz", train=12, dev_model=2,
                                    dev_score=0, calibration=0, test_locked=0, ood_locked=0)
            cls.data = load_dataset(path)

    def test_context_targets_remove_only_valid_covered_modes(self):
        ids = np.arange(12)
        for kind in ("empty", "single", "two_valid", "duplicate", "invalid", "mixed"):
            contexts = build_contexts(self.data, ids, np.random.default_rng(7), kind)
            self.assertTrue(contexts["remaining_mask"].any(axis=1).all())
            if kind in ("empty", "invalid"):
                self.assertFalse(contexts["valid"].any())
                np.testing.assert_array_equal(contexts["remaining_mask"], self.data["path_mask"][ids])
            if kind == "duplicate":
                np.testing.assert_array_equal(contexts["drafts"][:, 0], contexts["drafts"][:, 1])
                self.assertTrue(np.all(contexts["covered_count"] == 1))
            for row, idx in enumerate(ids):
                if not contexts["all_known_covered"][row]:
                    modes = route_modes(contexts["drafts"][row], self.data["scenes"][idx])
                    covered = set(modes[contexts["valid"][row]].tolist())
                    remaining = set(self.data["modes"][idx, contexts["remaining_mask"][row]].tolist())
                    self.assertFalse(covered & remaining)

    def test_coverage_duplicate_idempotence_and_invalid_draft_invariance(self):
        torch.manual_seed(2)
        model = CompletionRegressor(width=32, depth=1, mechanism="coverage").eval()
        scenes = torch.as_tensor(self.data["scenes"][:2])
        drafts = torch.as_tensor(self.data["paths"][:2, :2]).clone()
        one = torch.tensor([[True, False], [True, False]])
        with torch.no_grad():
            original = model(scenes, drafts, one)
            drafts[:, 1] = drafts[:, 0]
            repeated = model(scenes, drafts, torch.ones((2, 2), dtype=torch.bool))
            torch.testing.assert_close(original, repeated, atol=0., rtol=0.)
            drafts[:, 1] = float("nan")
            ignored = model(scenes, drafts, one)
            torch.testing.assert_close(original, ignored, atol=0., rtol=0.)

    def test_paired_parameter_shapes_empty_context_and_all_modules_train(self):
        torch.manual_seed(3)
        attention = CompletionRegressor(width=32, depth=1, mechanism="attention")
        torch.manual_seed(3)
        coverage = CompletionRegressor(width=32, depth=1, mechanism="coverage")
        self.assertEqual(attention.active_parameter_count(), coverage.active_parameter_count())
        for a, b in zip(attention.parameters(), coverage.parameters()):
            torch.testing.assert_close(a, b, atol=0., rtol=0.)
        scenes = torch.as_tensor(self.data["scenes"][:2])
        drafts = torch.as_tensor(self.data["paths"][:2, :2])
        for model in (attention, coverage):
            pred = model(scenes, drafts, torch.ones((2, 2), dtype=torch.bool), k=4)
            self.assertEqual(pred.shape, (2, 4, 22, 3))
            pred.square().mean().backward()
            self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
            empty = model(scenes, drafts, torch.zeros((2, 2), dtype=torch.bool), k=4)
            self.assertTrue(torch.isfinite(empty).all())

    def test_attention_also_handles_duplication_of_a_single_draft(self):
        # Do not mistake the [A,A] special case for a unique max-pool benefit.
        model = CompletionRegressor(width=32, depth=1, mechanism="attention").eval()
        scenes = torch.as_tensor(self.data["scenes"][:2])
        drafts = torch.as_tensor(self.data["paths"][:2, :2]).clone()
        single_valid = torch.tensor([[True, False], [True, False]])
        with torch.no_grad():
            single = model(scenes, drafts, single_valid)
            drafts[:, 1] = drafts[:, 0]
            duplicate = model(scenes, drafts, torch.ones((2, 2), dtype=torch.bool))
            torch.testing.assert_close(single, duplicate, atol=1e-7, rtol=1e-6)

    def test_self_draft_training_resume_preserves_rng_and_updates(self):
        from scripts.train_completion import train_one
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = generate_dataset(root / 'data.npz', train=8, dev_model=2,
                                    dev_score=0, calibration=0, test_locked=0, ood_locked=0)
            common = dict(data=str(data), mechanism='coverage', steps=8, batch_size=2,
                          width=32, depth=1, lr=3e-4, seed=11, eval_every=4,
                          threads=1, device='cpu', self_draft_prob=1., self_draft_start=1)
            train_one(Namespace(**common, output=str(root/'full'), resume=False, stop_after=None))
            train_one(Namespace(**common, output=str(root/'resumed'), resume=False, stop_after=4))
            train_one(Namespace(**common, output=str(root/'resumed'), resume=True, stop_after=None))
            full = torch.load(root/'full/last.pt', map_location='cpu', weights_only=False)
            resumed = torch.load(root/'resumed/last.pt', map_location='cpu', weights_only=False)
            self.assertEqual(full['self_draft_batches'], 6)
            self.assertEqual(full['self_draft_batches'], resumed['self_draft_batches'])
            for name, value in full['model'].items():
                torch.testing.assert_close(value, resumed['model'][name], atol=0., rtol=0.)


if __name__ == "__main__":
    unittest.main()
