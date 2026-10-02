"""Observation-only conditioning and event-preserving endpoint prediction tests."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from routeset.common import sha256
from routeset.observed_route_head import (OBSERVATION_KEYS, QWEN_REVISION, ObservedRouteHead,
                                         load_observed_dataset, resample_event_segments,
                                         semantic_endpoint_accuracy)


def make_observation_fixture(directory):
    """Synthetic unit-test inputs only; this fixture is not research evidence."""
    root = Path(directory)
    cache = root / "cache"
    cache.mkdir(exist_ok=True)
    rows, labels = [], []
    for parent_index, split in enumerate(("TRAIN", "DEV_MODEL")):
        parent = "parent_%d" % parent_index
        folder = root / parent
        folder.mkdir(exist_ok=True)
        (folder / "image.bin").write_bytes(("unit-test-image-%d" % parent_index).encode())
        current = np.array([.1, .2, .3, 0., 0., 0., 1.], dtype=np.float32)
        np.savez(folder / "observation.npz", gripper_pose=current, gripper_open=1.,
                 task_low_dim_state=np.array([99., 99., 99.]), forbidden_true_goal=np.ones(3) * 42)
        centers = [[.3, .2, .3], [.1, .4, .3]]
        for target_index in range(2):
            identifier = parent + "_target%d" % target_index
            xyz = np.linspace(current[:3], centers[target_index], 8)
            poses = np.c_[xyz, np.tile(current[3:], (8, 1))]
            filename = parent + "/route_%d.npz" % target_index
            np.savez(root / filename, gripper_pose=poses, gripper_open=np.ones(8))
            row = dict(id=identifier, parent_id=parent, split=split, image=parent + "/image.bin", instruction="Reach target %d" % target_index)
            rows.append(row)
            labels.append(dict(id=identifier, parent_id=parent, split=split, observation=parent + "/observation.npz",
                               routes=[filename], semantic_targets=dict(centers=centers, target_index=target_index, tolerance=.03)))
            key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
            # Intentional synthetic feature fixture; never used for actual runs.
            feature = np.arange(8, dtype=np.float32) + target_index
            np.savez(cache / (key + ".npz"), mean_hidden=feature, last_hidden=feature + .1,
                     id=identifier, parent_id=parent, split=split,
                     image_sha256=sha256(folder / "image.bin"), input_tokens=12)
    observation_path, supervision_path = root / "observations.jsonl", root / "supervision.jsonl"
    observation_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    supervision_path.write_text("\n".join(json.dumps(row) for row in labels) + "\n", encoding="utf-8")
    config = dict(model="Qwen/Qwen3-VL-2B-Instruct", revision=QWEN_REVISION,
                  model_trainable_parameter_count=0, manifest_sha256=sha256(observation_path), input_contract=sorted(OBSERVATION_KEYS))
    (cache / "cache_config.json").write_text(json.dumps(config), encoding="utf-8")
    return observation_path, supervision_path, cache


class ObservedRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_event_resampling_retains_grasp_and_release_boundaries(self):
        path = np.c_[np.arange(9), np.zeros((9, 2))]
        opened = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1])
        xyz, states = resample_event_segments(path, opened, 12)
        changes = np.flatnonzero(states[1:] != states[:-1]) + 1
        self.assertEqual(len(changes), 2)
        np.testing.assert_array_equal(xyz[changes - 1, 0], [2., 5.])
        np.testing.assert_array_equal(xyz[changes, 0], [3., 6.])
        with self.assertRaises(ValueError):
            resample_event_segments(path, opened, 4)

    def test_loader_uses_only_cache_and_current_state_as_inputs(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, labels, cache = make_observation_fixture(directory)
            data = load_observed_dataset(observations, labels, cache)
            self.assertEqual(data["features"].shape, (4, 16))
            self.assertEqual(data["current"].shape, (4, 8))
            self.assertFalse(np.any(data["current"] == 42))
            altered = [json.loads(line) for line in labels.read_text().splitlines()]
            for label in altered:
                label["semantic_targets"]["centers"] = [[999., 999., 999.], [-999., -999., -999.]]
            labels.write_text("\n".join(json.dumps(row) for row in altered) + "\n")
            changed = load_observed_dataset(observations, labels, cache)
            np.testing.assert_array_equal(data["features"], changed["features"])
            np.testing.assert_array_equal(data["current"], changed["current"])
            np.testing.assert_array_equal(data["paths"], changed["paths"])

    def test_true_goal_in_observation_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            observations, labels, cache = make_observation_fixture(directory)
            rows = [json.loads(line) for line in observations.read_text().splitlines()]
            rows[0]["goal_xyz"] = [1., 2., 3.]
            observations.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            with self.assertRaises(ValueError):
                load_observed_dataset(observations, labels, cache)

    def test_endpoint_is_learned_and_all_new_parameters_receive_gradients(self):
        model = ObservedRouteHead(16, width=32, depth=1, max_candidates=4)
        features = torch.randn(2, 16)
        current = torch.randn(2, 8)
        current[:, 7] = 1.
        xyz, opened = model(features, current)
        self.assertEqual(xyz.shape, (2, 4, 24, 3))
        torch.testing.assert_close(xyz[:, :, 0], current[:, None, :3].expand(-1, 4, -1))
        xyz[:, :, -1].square().mean().add(opened.mean()).backward()
        self.assertTrue(all(parameter.grad is not None for parameter in model.parameters()))
        self.assertGreater(float(model.output[-1].weight.grad.abs().sum()), 0.)

    def test_semantic_labels_require_identity_and_tolerance(self):
        spec = dict(centers=[[0., 0., 0.], [1., 0., 0.]], target_index=1, tolerance=.03)
        np.testing.assert_array_equal(semantic_endpoint_accuracy(np.array([[1., 0., 0.], [0., 0., 0.], [.9, 0., 0.]]), spec), [True, False, False])
        self.assertIsNone(semantic_endpoint_accuracy(np.zeros((2, 3)), None))


if __name__ == "__main__":
    unittest.main()
