"""Deterministic grouped 3-D route data for an initial task-layer experiment.

This is a procedurally generated, privileged-geometry benchmark, not a robotics
dataset or an evaluation of real-image VLM generalization. The unit of splitting
is a complete scene-task group with all twelve reference trajectories.
"""

import argparse
from pathlib import Path

import numpy as np

from .geometry import path_validity, render_scene, route_modes


def resample_polyline(vertices, horizon):
    """Arc-length resample a piecewise linear route with exact endpoints."""
    vertices = np.asarray(vertices, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(vertices, axis=0), axis=-1)
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if horizon < 2 or cumulative[-1] <= 0:
        raise ValueError("horizon >= 2 and a nonzero length route are required")
    coordinates = np.linspace(0.0, cumulative[-1], horizon)
    result = np.stack([np.interp(coordinates, cumulative, vertices[:, axis])
                       for axis in range(3)], axis=-1)
    result[0], result[-1] = vertices[0], vertices[-1]
    return result


def _sample_scene(rng, ood=False):
    if ood:
        # Wider obstacles and x locations outside the training support.
        height = rng.uniform(.80, .95)
        center = np.array([rng.choice([-1., 1.]) * rng.uniform(.22, .29),
                           rng.uniform(-.13, .13), height / 2.])
        halfsize = np.array([rng.uniform(.14, .19), rng.uniform(.35, .43),
                            height / 2.])
    else:
        height = rng.uniform(.65, .80)
        center = np.array([rng.uniform(-.12, .12), rng.uniform(-.13, .13),
                           height / 2.])
        halfsize = np.array([rng.uniform(.14, .19), rng.uniform(.20, .31),
                            height / 2.])
    # The obstacle rests on the workspace floor: there is no fourth under-box
    # passage. Endpoint y offsets also ensure the straight line is obstructed.
    start = np.array([-.85, center[1] + rng.uniform(-.15, .15), rng.uniform(.23, .37)])
    goal = np.array([.85, center[1] + rng.uniform(-.15, .15), rng.uniform(.23, .37)])
    return np.concatenate([start, goal, center, halfsize])


def _reference_routes(scene, rng, horizon):
    start, goal, center, halfsize = scene.reshape(4, 3)
    left_x, right_x = center[0] - halfsize[0], center[0] + halfsize[0]
    paths, modes = [], []
    for mode in range(3):
        for variant in range(4):
            # Broad geometric variants in the same useful passage; the passage
            # mode is generated explicitly, not inferred from noise samples.
            detour = .115 + .029 * variant + rng.uniform(0., .012)
            shoulder = .135 + .012 * variant
            shoulder_left, shoulder_right = left_x - shoulder, right_x + shoulder
            if mode < 2:
                sign = -1. if mode == 0 else 1.
                aisle_y = center[1] + sign * (halfsize[1] + detour)
                route_z = .5 * (start[2] + goal[2]) + rng.uniform(-.015, .015)
                # The middle bend changes lateral shape while staying outside
                # the obstacle; vertical perturbations remain in the passage.
                mid_y = aisle_y + sign * (.012 + .007 * variant)
                vertices = np.stack([start,
                    np.array([shoulder_left, aisle_y, route_z]),
                    np.array([center[0], mid_y, route_z + .012 * (variant - 1.5)]),
                    np.array([shoulder_right, aisle_y, route_z]), goal])
            else:
                top_z = center[2] + halfsize[2] + detour
                mid_y = .5 * (start[1] + goal[1]) + (variant - 1.5) * .025
                vertices = np.stack([start,
                    np.array([shoulder_left, mid_y, top_z]),
                    np.array([center[0], mid_y, top_z + .012 + .007 * variant]),
                    np.array([shoulder_right, mid_y, top_z]), goal])
            path = resample_polyline(vertices, horizon)
            # Resampling can cut a corner when one segment straddles a vertex.
            # Validate the actual sampled polyline, not the original vertices.
            check = path_validity(path, scene)
            if not bool(check["valid"]):
                raise ValueError("reference route failed geometry checks; increase horizon or margins")
            if int(route_modes(path, scene)) != mode:
                raise ValueError("reference route crossed an unintended passage")
            paths.append(path)
            modes.append(mode)
    return np.stack(paths), np.asarray(modes, dtype=np.int64)


def generate_dataset(output, train=768, val=128, test=128, ood=128,
                     seed=17, horizon=24):
    """Write one NPZ with scene-level train/val/test/OOD groups; return its Path.

    Fields are scenes [S,12], paths [S,12,H,3], modes [S,12], splits [S],
    scene_ids [S], and instructions [S]. No reference information appears in
    rendered model observations. Seeds make all arrays reproducible.
    """
    output = Path(output)
    if output.is_dir():
        output = output / "routes.npz"
    if output.suffix.lower() != ".npz":
        output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = {"train": train, "val": val, "test": test, "ood": ood}
    if any(int(n) != n or n < 0 for n in counts.values()) or sum(counts.values()) < 1:
        raise ValueError("split counts must be nonnegative integers with at least one scene")
    if horizon < 16:
        raise ValueError("horizon >= 16 is required to preserve geometric clearance")
    rng = np.random.default_rng(seed)
    scene_list, path_list, mode_list, splits, ids, instructions = [], [], [], [], [], []
    templates = [
        "Plan collision-free task routes from the red start to the green goal while avoiding the gray obstacle.",
        "Move from the red start to the green destination. Keep a safe clearance from the gray box.",
        "Find alternative safe waypoint routes connecting the red point and the green point around the obstacle.",
        "Reach the green target from the red starting point without crossing the gray box.",
    ]
    for split, count in counts.items():
        for index in range(int(count)):
            scene = _sample_scene(rng, ood=split == "ood")
            paths, modes = _reference_routes(scene, rng, horizon)
            scene_list.append(scene)
            path_list.append(paths)
            mode_list.append(modes)
            splits.append(split)
            ids.append("%s_%05d" % (split, index))
            instructions.append(templates[int(rng.integers(len(templates)))])
    np.savez_compressed(str(output), scenes=np.asarray(scene_list, dtype=np.float32),
                        paths=np.asarray(path_list, dtype=np.float32), modes=np.asarray(mode_list),
                        splits=np.asarray(splits), scene_ids=np.asarray(ids),
                        instructions=np.asarray(instructions))
    return output


def load_dataset(path, split=None):
    """Load NPZ arrays safely; optionally select an entire scene split."""
    with np.load(str(path), allow_pickle=False) as archive:
        result = {name: archive[name] for name in archive.files}
    if split is not None:
        mask = result["splits"] == split
        result = {name: values[mask] for name, values in result.items()}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/routes.npz")
    parser.add_argument("--train", type=int, default=768)
    parser.add_argument("--val", type=int, default=128)
    parser.add_argument("--test", type=int, default=128)
    parser.add_argument("--ood", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--horizon", type=int, default=24)
    args = parser.parse_args()
    output = generate_dataset(**vars(args))
    dataset = load_dataset(output)
    print("Wrote %d scene groups, %d reference routes each, to %s" %
          (len(dataset["scenes"]), dataset["paths"].shape[1], output))


if __name__ == "__main__":
    main()
