"""Physical mode identity, leakage boundary, shared budgets and real resume."""
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

from routeset.constraint_update import (ConstraintUpdateRegressor, checked_context, close_opening,
    derive_changes, training_targets, update_metrics)
from routeset.multigate import generate_dataset, load_dataset, path_validity, route_modes


class ConstraintUpdateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            path = generate_dataset(Path(directory) / "data.npz", train=20, dev_model=10,
                dev_score=0, calibration=0, test_locked=0, ood_locked=0)
            data = load_dataset(path)
        # Fixtures use positive drafts only to test algorithms, not research
        # results. The actual collection CLI requires a real checkpoint.
        old = np.stack([paths[np.resize(np.flatnonzero(mask), 4)]
                        for paths, mask in zip(data["paths"], data["path_mask"])])
        cls.original, cls.old = data, old
        cls.data = derive_changes(data, old)

    def test_closed_gap_keeps_physical_ids_and_all_positives_valid(self):
        for old, new, paths, modes, mask, wall, opening in zip(self.data["old_scenes"], self.data["scenes"],
                self.data["paths"], self.data["modes"], self.data["path_mask"], self.data["closed_wall"], self.data["closed_opening"]):
            self.assertEqual(np.count_nonzero(old != new), 1)
            self.assertEqual(new[6 + 14 * wall + 10 + opening], 0)
            self.assertTrue(path_validity(paths[mask], new)["valid"].all())
            np.testing.assert_array_equal(route_modes(paths[mask], new), modes[mask])
        before = {p: s for p, s in zip(self.original["parent_ids"], self.original["splits"])}
        self.assertTrue(all(before[p] == s for p, s in zip(self.data["parent_ids"], self.data["splits"])))

    def test_locked_archive_rejected(self):
        data = {key: value.copy() for key, value in self.original.items()}
        data["splits"] = data["splits"].astype("U20")
        data["splits"][0] = "TEST_LOCKED"
        with self.assertRaises(ValueError):
            derive_changes(data, self.old)

    def test_training_labels_cannot_be_used_for_development(self):
        ids = np.flatnonzero(self.data["splits"] == "TRAIN")[:5]
        for k in (1, 2):
            a = training_targets(self.data, ids, k)
            b = training_targets(self.data, ids, k)
            for key in ("targets", "proxy", "remaining_mask", "selected_reference_ids"):
                np.testing.assert_array_equal(a[key], b[key])
            for row, idx in enumerate(ids):
                self.assertTrue(path_validity(a["targets"][row], self.data["scenes"][idx])["valid"].all())
                self.assertTrue(a["remaining_mask"][row, a["selected_reference_ids"][row]].all())
        with self.assertRaises(ValueError):
            training_targets(self.data, np.flatnonzero(self.data["splits"] == "DEV_MODEL")[:1], 1)

    def test_identical_initialization_and_no_inference_mask(self):
        ids = np.arange(2)
        torch.manual_seed(1)
        full = ConstraintUpdateRegressor(width=16, depth=1, mechanism="full").eval()
        torch.manual_seed(1)
        local = ConstraintUpdateRegressor(width=16, depth=1, mechanism="local").eval()
        self.assertEqual(full.active_parameter_count(), local.active_parameter_count())
        for a, b in zip(full.parameters(), local.parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        context = checked_context(self.data["old_paths"][ids], self.data["old_scenes"][ids], self.data["scenes"][ids])
        inputs = [torch.as_tensor(x) for x in (self.data["old_scenes"][ids], self.data["scenes"][ids], context["drafts"], context["valid"])]
        with torch.no_grad():
            # Algebraically identical; subtraction/addition may round once.
            torch.testing.assert_close(full(*inputs)[0], local(*inputs)[0], atol=1e-7, rtol=1e-6)
            local.edit_head[-1].bias.fill_(-2.)
            copied, logits = local(*inputs)
            self.assertTrue((logits < 0).all())
            torch.testing.assert_close(copied[:, :, 1:-1], inputs[2][:, :2, 1:-1], atol=0, rtol=0)
        with self.assertRaises(TypeError):
            local(*inputs, edit_mask=torch.ones(2, 2, 22))

    def test_budget_counts_failed_and_repeated_outputs(self):
        ids = np.arange(3)
        old = self.data["old_paths"][ids]
        new = old[:, :2].copy()
        new[:, 0, 10] = np.array([0., 0., 10.])
        rows = update_metrics(old, new, self.data["scenes"][ids], self.data["modes"][ids], self.data["path_mask"][ids])
        self.assertTrue(all(row["total_candidates"] == 6 and row["new_candidates"] == 2 for row in rows))
        self.assertTrue(all(row["additional_unique_valid"] == 0 for row in rows))
        empty = update_metrics(old, new[:, :0], self.data["scenes"][ids], self.data["modes"][ids], self.data["path_mask"][ids])
        self.assertTrue(all(row["total_candidates"] == 4 for row in empty))

    def test_continuous_and_resumed_training_are_identical(self):
        from scripts.train_constraint_update import train_one
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # One real parent per split, retaining all their closures.
            selected = []
            for split in ("TRAIN", "DEV_MODEL"):
                parent = self.data["parent_ids"][np.flatnonzero(self.data["splits"] == split)[0]]
                selected.extend(np.flatnonzero(self.data["parent_ids"] == parent).tolist())
            np.savez_compressed(root / "data.npz", **{k: v[selected] for k, v in self.data.items()})
            common = dict(data=str(root / "data.npz"), arm="local_paired", steps=4, batch_size=2,
                width=16, depth=1, lr=3e-4, seed=5, eval_every=2, threads=1, device="cpu",
                aux_weight=.01, proxy_threshold=.02)
            train_one(Namespace(**common, output=str(root / "full"), resume=False, stop_after=None))
            train_one(Namespace(**common, output=str(root / "resume"), resume=False, stop_after=2))
            train_one(Namespace(**common, output=str(root / "resume"), resume=True, stop_after=None))
            a = torch.load(root / "full" / "last.pt", weights_only=False)
            b = torch.load(root / "resume" / "last.pt", weights_only=False)
            self.assertEqual(a["trajectory_exposures"], 12)
            self.assertEqual(a["trajectory_exposures"], b["trajectory_exposures"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)

    def test_new_output_continuation_is_exact_and_rejects_data_or_lr_changes(self):
        import copy
        from routeset.common import sha256
        from scripts.train_constraint_update import train_one
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selected = []
            for split in ("TRAIN", "DEV_MODEL"):
                parent = self.data["parent_ids"][np.flatnonzero(self.data["splits"] == split)[0]]
                selected.extend(np.flatnonzero(self.data["parent_ids"] == parent).tolist())
            small = {k: v[selected] for k, v in self.data.items()}
            np.savez_compressed(root / "data.npz", **small)
            changed = {k: v.copy() for k, v in small.items()}
            changed["old_paths"][0, 0, 3, 1] += .001
            np.savez_compressed(root / "changed.npz", **changed)
            common = dict(data=str(root / "data.npz"), arm="local_paired", batch_size=2,
                width=16, depth=1, lr=3e-4, seed=5, eval_every=2, threads=1, device="cpu",
                aux_weight=.01, proxy_threshold=.02, resume=False, stop_after=None)
            train_one(Namespace(**common, steps=4, output=str(root / "continuous"), continue_from=None))
            train_one(Namespace(**common, steps=2, output=str(root / "source"), continue_from=None))
            source_hashes = {name: sha256(root / "source" / name) for name in ("last.pt", "best.pt", "config.json", "summary.json")}
            continuation = Namespace(**common, steps=4, output=str(root / "continued"), continue_from=str(root / "source" / "last.pt"))
            for key, value in (("lr", 1e-4), ("data", str(root / "changed.npz"))):
                rejected = copy.copy(continuation)
                rejected.output = str(root / ("rejected_" + key))
                setattr(rejected, key, value)
                with self.assertRaisesRegex(ValueError, "Continuation config mismatch"):
                    train_one(rejected)
                self.assertFalse((Path(rejected.output) / "best.pt").exists())
            train_one(continuation)
            a = torch.load(root / "continuous" / "last.pt", weights_only=False)
            b = torch.load(root / "continued" / "last.pt", weights_only=False)
            self.assertEqual(b["trajectory_exposures"], 12)
            self.assertEqual(b["incremental_trajectory_exposures"], 6)
            self.assertEqual(b["cost_origin"]["prior_trajectory_exposures"], 6)
            self.assertEqual(b["config"]["continuation"]["source_checkpoint_sha256"], source_hashes["last.pt"])
            self.assertGreater(b["cumulative_elapsed_s"], b["elapsed_s"])
            for key in a["model"]:
                torch.testing.assert_close(a["model"][key], b["model"][key], atol=0, rtol=0)
            self.assertEqual(a["scheduler"], b["scheduler"])
            self.assertEqual(a["sampler_state"], b["sampler_state"])
            for name, expected in source_hashes.items():
                self.assertEqual(sha256(root / "source" / name), expected)


if __name__ == "__main__":
    unittest.main()
