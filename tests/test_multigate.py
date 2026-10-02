"""Geometry and leakage checks for the variable-opening controlled benchmark."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from routeset.multigate import (SPLITS, generate_dataset, load_dataset, path_validity,
                               route_from_openings, route_metrics, route_modes, unpack_scene,
                               wall_boxes)


class MultiGateTests(unittest.TestCase):
    def make_data(self, directory, **kwargs):
        options = dict(train=20, dev_model=3, dev_score=2, calibration=2,
                       test_locked=3, ood_locked=5, seed=7)
        options.update(kwargs)
        return load_dataset(generate_dataset(Path(directory) / "data.npz", **options))

    def test_all_reference_pairs_valid_and_duplicates_do_not_add_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_data(directory)
        counts = data["path_mask"].sum(1)
        self.assertGreater(len(set(counts.tolist())), 4)
        self.assertTrue(np.any(counts < 4))
        self.assertTrue(np.any(counts > 8))
        for scene, paths, mask, modes in zip(data["scenes"], data["paths"], data["path_mask"], data["modes"]):
            self.assertTrue(path_validity(paths[mask], scene)["valid"].all())
            np.testing.assert_array_equal(route_modes(paths[mask], scene), modes[mask])
            _, _, walls = unpack_scene(scene)
            self.assertEqual(int(mask.sum()), int(np.prod(walls[:, 10:14].sum(1))))
            self.assertTrue(np.all(modes[~mask] == -1))
        repeated = np.repeat(data["paths"][:, :1], 8, axis=1)
        metrics = route_metrics(repeated, data["scenes"], data["modes"], data["path_mask"])
        self.assertEqual(metrics["ValidAtK"], 1.)
        self.assertEqual(metrics["UniqueValidAtK"], 1.)
        self.assertAlmostEqual(metrics["ReferenceCoverageAtK"], np.mean(1. / counts))
        self.assertNotIn("SelectedValidAtK", metrics)

    def test_segments_between_waypoints_and_nonfinite_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_data(directory)
        scene = data["scenes"][0]
        lower, upper = wall_boxes(scene)
        center = (lower[0] + upper[0]) / 2
        across = np.array([[center[0] - .2, center[1], .5], [center[0] + .2, center[1], .5]])
        self.assertTrue(path_validity(across, scene)["collision"])
        good = data["paths"][0, 0].copy()
        good[3, 1] = np.nan
        self.assertFalse(path_validity(good, scene)["valid"])
        self.assertEqual(route_modes(good, scene), -1)

    def test_variants_split_by_parent_and_locked_stream_independent_of_train_size(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_data(directory, variants_per_parent=2)
            repeat = self.make_data(directory, variants_per_parent=2)
            more_train = self.make_data(directory, train=21, variants_per_parent=2)
        for key in first:
            np.testing.assert_array_equal(first[key], repeat[key])
            np.testing.assert_array_equal(first[key][first["splits"] == "TEST_LOCKED"],
                                          more_train[key][more_train["splits"] == "TEST_LOCKED"])
        self.assertEqual(set(first["splits"]), set(SPLITS))
        for parent in set(first["parent_ids"]):
            rows = first["parent_ids"] == parent
            self.assertEqual(len(set(first["splits"][rows])), 1)
            self.assertEqual(int(rows.sum()), 2)
            np.testing.assert_array_equal(first["scenes"][rows][0, 6:], first["scenes"][rows][1, 6:])

    def test_valid_same_mode_variants_and_actual_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            data = self.make_data(directory)
        scene = data["scenes"][0]
        variants = np.stack([route_from_openings(scene, 0, 0, offsets=(offset, -offset)) for offset in [-.2, 0., .2]])
        self.assertTrue(path_validity(variants, scene)["valid"].all())
        np.testing.assert_array_equal(route_modes(variants, scene), [0, 0, 0])
        predictions = variants[None].copy()
        predictions[0, 1, -1, 0] = .7
        metrics = route_metrics(predictions, scene[None], scores=np.array([[0., 1., 0.]]))
        self.assertEqual(metrics["AnyValidAtK"], 1.)
        self.assertEqual(metrics["SelectedValidAtK"], 0.)
        self.assertEqual(metrics["UniqueValidAtK"], 1.)


if __name__ == "__main__":
    unittest.main()
