"""Meaningful geometry, grouped-data, and paired-view consistency checks."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from routeset.data import generate_dataset, load_dataset
from routeset.geometry import (camera_matrices, path_validity, project_points,
                               render_scene, reprojection_error, route_metrics,
                               route_modes, segment_aabb_distance,
                               segment_aabb_intersection, triangulate_points)


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.scene = np.array([-.85, 0., .3, .85, 0., .3,
                               0., 0., .375, .18, .25, .375])

    def test_segment_collision_checks_between_safe_waypoints(self):
        paths = np.array([[[-.85, 0., .3], [.85, 0., .3]]])
        result = path_validity(paths, self.scene)
        self.assertTrue(result["collision"][0])
        self.assertFalse(result["valid"][0])
        self.assertAlmostEqual(float(result["clearance"][0]), 0., places=12)

    def test_slabs_stationary_parallel_tangent_and_miss(self):
        lo, hi = np.array([-1., -1., -1.]), np.array([1., 1., 1.])
        starts = np.array([[0., 0., 0.], [2., 0., 0.], [-2., 1., 0.], [-2., 2., 0.]])
        ends = np.array([[0., 0., 0.], [2., 0., 0.], [2., 1., 0.], [2., 2., 0.]])
        np.testing.assert_array_equal(segment_aabb_intersection(starts, ends, lo, hi),
                                      [True, False, True, False])
        np.testing.assert_allclose(segment_aabb_distance(starts, ends, lo, hi), [0., 1., 0., 1.])

    def test_corner_distance_uses_interior_segment_minimum(self):
        distance = segment_aabb_distance(np.array([2., 0., 0.]), np.array([0., 2., 0.]),
                                         np.array([-.5, -.5, -.5]), np.array([.5, .5, .5]))
        self.assertAlmostEqual(float(distance), np.sqrt(.5))

    def test_endpoint_and_workspace_are_validity_requirements(self):
        side = np.array([[-.85, 0., .3], [-.45, -.45, .3],
                         [.45, -.45, .3], [.85, 0., .3]])
        self.assertTrue(path_validity(side, self.scene)["valid"])
        bad_endpoint = side.copy()
        bad_endpoint[-1, 0] = .7
        self.assertFalse(path_validity(bad_endpoint, self.scene)["valid"])
        out_of_bounds = side.copy()
        out_of_bounds[1:3, 1] = -1.2
        self.assertFalse(path_validity(out_of_bounds, self.scene)["in_bounds"])

    def test_nonfinite_routes_cannot_be_valid_or_classified(self):
        path = np.array([[-.85, 0., .3], [np.nan, -.5, .3], [.85, 0., .3]])
        result = path_validity(path, self.scene)
        self.assertFalse(result["finite"])
        self.assertFalse(result["valid"])
        self.assertEqual(int(route_modes(path, self.scene)), -1)

    def test_calibrated_multiview_roundtrip_and_reprojection(self):
        rng = np.random.default_rng(9)
        points = rng.uniform([-.9, -.9, .05], [.9, .9, 1.2], size=(5, 24, 3))
        matrices = camera_matrices()
        projections = project_points(points, matrices)
        self.assertEqual(projections.shape, (2, 5, 24, 2))
        reconstructed = triangulate_points(projections, matrices)
        np.testing.assert_allclose(reconstructed, points, rtol=0., atol=1e-10)
        self.assertLess(float(np.max(reprojection_error(reconstructed, projections, matrices))), 1e-10)

    def test_grouped_routes_and_seed_are_reproducible(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = generate_dataset(Path(tmp) / "a.npz", train=10, val=3, test=3, ood=6, seed=12)
            second = generate_dataset(Path(tmp) / "b.npz", train=10, val=3, test=3, ood=6, seed=12)
            a, b = load_dataset(first), load_dataset(second)
            for name in a:
                np.testing.assert_array_equal(a[name], b[name])
            self.assertEqual(a["paths"].shape, (22, 12, 24, 3))
            self.assertEqual(len(set(a["scene_ids"])), 22)
            # Stronger than unique IDs: no underlying geometry crosses splits.
            self.assertEqual(len(set(tuple(s) for s in a["scenes"])), 22)
            for paths, scene, modes in zip(a["paths"], a["scenes"], a["modes"]):
                self.assertTrue(np.all(path_validity(paths, scene)["valid"]))
                np.testing.assert_array_equal(route_modes(paths, scene), modes)
                np.testing.assert_array_equal(np.bincount(modes), [4, 4, 4])
                self.assertEqual(scene[8] - scene[11], 0.)
                straight = np.stack([scene[:3], scene[3:6]])
                self.assertTrue(path_validity(straight, scene)["collision"])
            train = load_dataset(first, "train")
            ood = load_dataset(first, "ood")
            self.assertTrue(np.all(np.abs(train["scenes"][:, 6]) <= .12))
            self.assertTrue(np.all(np.abs(ood["scenes"][:, 6]) >= .22))
            self.assertTrue(np.all(ood["scenes"][:, 10] > np.max(train["scenes"][:, 10])))
            full = route_metrics(a["paths"], a["scenes"])
            self.assertEqual(full["validity"], 1.)
            self.assertEqual(full["coverage"], 1.)
            self.assertEqual(full["unique_valid"], 3.)
            self.assertEqual(full["unclassified_valid_rate"], 0.)
            repeated = np.repeat(a["paths"][:, :1], 12, axis=1)
            collapsed = route_metrics(repeated, a["scenes"])
            self.assertEqual(collapsed["unique_valid"], 1.)
            self.assertAlmostEqual(collapsed["coverage"], 1. / 3.)

    def test_render_contains_only_geometry_and_has_paired_views(self):
        rendered = render_scene(self.scene, size=128)
        self.assertEqual(rendered.shape, (2, 128, 128, 3))
        self.assertEqual(rendered.dtype, np.uint8)
        self.assertFalse(np.array_equal(rendered[0], rendered[1]))
        self.assertTrue(np.array_equal(rendered, render_scene(self.scene, instruction="a new task")))


if __name__ == "__main__":
    unittest.main()
