"""Versioned controlled geometry benchmark with variable opening sequences.

This is tier A: inputs expose exact wall geometry and the goal, never an
observation-task/VLM benchmark. Two full-height walls have 1--4 openings each.
Route types are ordered opening pairs, not Euclidean trajectory clusters.
Enumerating these pairs does not enumerate all continuous feasible paths.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .geometry import segment_aabb_intersection


VERSION = "multigate_v1"
SPLITS = ("TRAIN", "DEV_MODEL", "DEV_SCORE", "CALIBRATION", "TEST_LOCKED", "OOD_LOCKED")
SCENE_DIM = 34
MAX_REFERENCES = 16
CLEARANCE = .02
ENDPOINT_TOL = .06
MAX_LENGTH_RATIO = 3.5
WORKSPACE_MIN = np.array([-1., -1., 0.])
WORKSPACE_MAX = np.array([1., 1., 1.])


def unpack_scene(scene):
    """Return start [3], goal [3], wall rows [2,14] from an oracle scene."""
    scene = np.asarray(scene, dtype=np.float64)
    if scene.shape != (SCENE_DIM,) or not np.all(np.isfinite(scene)):
        raise ValueError("scene must be a finite length-34 oracle geometry vector")
    return scene[:3], scene[3:6], scene[6:].reshape(2, 14)


def wall_boxes(scene):
    """Convert the complement of open y intervals to closed obstacle AABBs."""
    _, _, walls = unpack_scene(scene)
    lower, upper = [], []
    for row in walls:
        x, halfwidth = row[:2]
        openings = row[2:10].reshape(4, 2)[row[10:14] > .5]
        bottom = -1.
        for lo, hi in openings:
            lower.append([x - halfwidth, bottom, 0.])
            upper.append([x + halfwidth, lo, 1.])
            bottom = hi
        lower.append([x - halfwidth, bottom, 0.])
        upper.append([x + halfwidth, 1., 1.])
    return np.asarray(lower), np.asarray(upper)


def _landmarks(scene):
    start, goal, walls = unpack_scene(scene)
    # Shoulders remain outside clearance-expanded walls. Keeping every corner
    # in the sampled polyline prevents resampling from cutting into a wall.
    margin = CLEARANCE + .025
    x = [start[0]]
    for row in walls:
        x.extend([row[0] - row[1] - margin, row[0], row[0] + row[1] + margin])
    x.append(goal[0])
    x = np.asarray(x)
    if np.any(np.diff(x) <= 0.):
        raise ValueError("wall shoulders must be ordered between endpoints")
    return x


def path_x(scene, horizon=24):
    """Reference x sampling [H]; free-XYZ neural outputs are not clamped to it."""
    if horizon < 8:
        raise ValueError("horizon >= 8 is required to retain all wall shoulders")
    landmarks = _landmarks(scene)
    intervals = np.full(7, (horizon - 1) // 7, dtype=int)
    intervals[:(horizon - 1) % 7] += 1
    pieces = [np.linspace(a, b, n + 1)[:-1]
              for a, b, n in zip(landmarks[:-1], landmarks[1:], intervals)]
    return np.concatenate(pieces + [landmarks[-1:]])


def route_from_openings(scene, first, second, horizon=24, offsets=(0., 0.)):
    """Construct a checked-planner reference; offsets are fractions of width.

    This helper uses privileged geometry. It must not be presented as a learned
    generator or silently used to repair only one method's predictions.
    """
    start, goal, walls = unpack_scene(scene)
    ys = []
    for row, index, offset in zip(walls, (first, second), offsets):
        if index < 0 or index >= 4 or row[10 + index] < .5:
            raise ValueError("selected opening is absent")
        lo, hi = row[2:10].reshape(4, 2)[index]
        ys.append((lo + hi) * .5 + offset * (hi - lo - 2 * CLEARANCE))
    x = path_x(scene, horizon)
    y = np.interp(x, _landmarks(scene), [start[1]] + [ys[0]] * 3 + [ys[1]] * 3 + [goal[1]])
    z = np.interp(x, [start[0], goal[0]], [start[2], goal[2]])
    return np.stack([x, y, z], axis=-1)


def path_validity(paths, scene, clearance=CLEARANCE, endpoint_tol=ENDPOINT_TOL,
                  max_length_ratio=MAX_LENGTH_RATIO):
    """Exact complete-segment AABB checking; touching the safety box collides."""
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim < 2 or paths.shape[-1] != 3 or paths.shape[-2] < 2:
        raise ValueError("expected paths [..., H>=2, 3]")
    start, goal, _ = unpack_scene(scene)
    lower, upper = wall_boxes(scene)
    finite = np.all(np.isfinite(paths), axis=(-1, -2))
    p0, p1 = paths[..., :-1, :], paths[..., 1:, :]
    # The historical slab routine allocates its output from segment shape;
    # explicitly broadcast the obstacle axis before calling it.
    a, b, lo, hi = np.broadcast_arrays(p0[..., None, :], p1[..., None, :],
                                      lower - clearance, upper + clearance)
    collision = np.any(segment_aabb_intersection(a, b, lo, hi), axis=(-1, -2))
    endpoint_error = np.maximum(np.linalg.norm(paths[..., 0, :] - start, axis=-1),
                                np.linalg.norm(paths[..., -1, :] - goal, axis=-1))
    lengths = np.sum(np.linalg.norm(p1 - p0, axis=-1), axis=-1)
    length_ratio = lengths / max(float(np.linalg.norm(goal - start)), 1e-8)
    in_bounds = np.all((paths >= WORKSPACE_MIN) & (paths <= WORKSPACE_MAX), axis=(-1, -2))
    valid = finite & ~collision & in_bounds & (endpoint_error <= endpoint_tol) & (length_ratio <= max_length_ratio)
    return dict(valid=valid, collision=collision, endpoint_error=endpoint_error,
                lengths=lengths, length_ratio=length_ratio, in_bounds=in_bounds, finite=finite)


def route_modes(paths, scene):
    """Opening-sequence IDs 4*first+second, or -1 for unclassified routes.

    All polyline crossings of each wall plane must use the same opening;
    backtracking through different openings is deliberately unclassified.
    Collision validity is evaluated separately, using the entire wall volume.
    """
    paths = np.asarray(paths, dtype=np.float64)
    if paths.ndim < 2 or paths.shape[-1] != 3 or paths.shape[-2] < 2:
        raise ValueError("expected paths [..., H>=2, 3]")
    _, _, walls = unpack_scene(scene)
    prefix = paths.shape[:-2]
    flat = paths.reshape(-1, paths.shape[-2], 3)
    all_labels = []
    for row in walls:
        x0, x1 = flat[:, :-1, 0], flat[:, 1:, 0]
        crossing = (np.minimum(x0, x1) <= row[0]) & (np.maximum(x0, x1) >= row[0])
        delta = x1 - x0
        fraction = np.zeros_like(delta)
        np.divide(row[0] - x0, delta, out=fraction, where=np.abs(delta) > 1e-12)
        y = flat[:, :-1, 1] + fraction * np.diff(flat[..., 1], axis=1)
        openings = row[2:10].reshape(4, 2)
        labels = np.full(y.shape, -1, dtype=np.int64)
        for index, (lo, hi) in enumerate(openings):
            if row[10 + index] > .5:
                labels[(y > lo) & (y < hi) & crossing] = index
        # A segment lying on the wall plane must keep both endpoints in one gap.
        parallel = crossing & (np.abs(delta) <= 1e-12)
        for index, (lo, hi) in enumerate(openings):
            bad_parallel = parallel & (labels == index) & ((flat[:, 1:, 1] <= lo) | (flat[:, 1:, 1] >= hi))
            labels[bad_parallel] = -1
        initial = labels[np.arange(len(flat)), np.argmax(crossing, axis=1)]
        consistent = np.any(crossing, axis=1) & np.all(~crossing | (labels == initial[:, None]), axis=1)
        all_labels.append(np.where(consistent, initial, -1))
    first, second = all_labels
    modes = np.where((first >= 0) & (second >= 0), 4 * first + second, -1)
    modes[~np.all(np.isfinite(flat), axis=(-1, -2))] = -1
    return modes.reshape(prefix)


def route_metrics(paths, scenes, reference_modes=None, reference_mask=None, scores=None):
    """Equal-K metrics, optionally relative to known positive reference modes.

    No absent reference labels are used as negative supervision. SelectedValid
    is only reported when independent candidate scores are supplied. AnyValid
    is explicitly the set's oracle upper bound, not scorer performance.
    """
    paths, scenes = np.asarray(paths), np.asarray(scenes)
    if paths.ndim != 4 or scenes.shape != (len(paths), SCENE_DIM) or paths.shape[1] < 1 or len(paths) < 1:
        raise ValueError("expected nonempty paths [B,K,H,3], scenes [B,34]")
    checks = [path_validity(p, s) for p, s in zip(paths, scenes)]
    valid = np.stack([c["valid"] for c in checks])
    modes = np.stack([route_modes(p, s) for p, s in zip(paths, scenes)])
    useful = np.where(valid, modes, -1)
    unique_count = np.asarray([len(set(row[row >= 0].tolist())) for row in useful])
    per_scene = dict(valid=valid, modes=useful, unique_count=unique_count,
                     lengths=np.stack([c["lengths"] for c in checks]),
                     collision=np.stack([c["collision"] for c in checks]))
    result = dict(validity=float(valid.mean()), valid_rate=float(valid.mean()),
                  unique_valid=float(unique_count.mean()), success=float(np.any(valid, axis=1).mean()),
                  collision_rate=float(per_scene["collision"].mean()),
                  unclassified_valid_rate=float((valid & (modes < 0)).mean()),
                  finite_rate=float(np.stack([c["finite"] for c in checks]).mean()),
                  candidate_count=int(paths.shape[1]))
    result.update(ValidAtK=result["validity"], AnyValidAtK=result["success"], UniqueValidAtK=result["unique_valid"])
    if reference_modes is not None:
        refs = np.asarray(reference_modes)
        mask = (refs >= 0) if reference_mask is None else np.asarray(reference_mask, dtype=bool) & (refs >= 0)
        if refs.ndim != 2 or refs.shape[0] != len(paths) or mask.shape != refs.shape:
            raise ValueError("reference modes/mask must have shape [B,R]")
        coverage = []
        for predicted, known, keep in zip(useful, refs, mask):
            target = set(known[keep].tolist())
            coverage.append(len(set(predicted[predicted >= 0].tolist()) & target) / len(target) if target else np.nan)
        per_scene["reference_coverage"] = np.asarray(coverage)
        result["reference_coverage"] = float(np.nanmean(coverage)) if np.any(np.isfinite(coverage)) else None
        result["ReferenceCoverageAtK"] = result["reference_coverage"]
    if scores is not None:
        scores = np.asarray(scores)
        if scores.shape != valid.shape or not np.all(np.isfinite(scores)):
            raise ValueError("finite scores [B,K] required")
        selected = np.argmax(scores, axis=1)
        per_scene["selected_valid"] = valid[np.arange(len(valid)), selected]
        result["selected_valid"] = float(per_scene["selected_valid"].mean())
        result["SelectedValidAtK"] = result["selected_valid"]
    result["per_scene"] = per_scene
    return result


def _sample_scene(rng, ood=False):
    start = [-.92, rng.uniform(-.75, .75), .5]
    goal = [.92, rng.uniform(-.75, .75), .5]
    positions = [rng.uniform(-.53, -.43), rng.uniform(.43, .53)] if ood else [rng.uniform(-.38, -.22), rng.uniform(.22, .38)]
    rows = []
    for x in positions:
        count = int(rng.integers(1, 5))
        # Different widths and locations, with guaranteed separating solids.
        cell_width = 1.8 / count
        centers = np.linspace(-.9 + .5 * cell_width, .9 - .5 * cell_width, count)
        centers += rng.uniform(-.11, .11, count) * cell_width
        widths = rng.uniform(.27, .41, count) * cell_width if ood else rng.uniform(.42, .66, count) * cell_width
        openings = np.zeros((4, 2))
        openings[:count, 0], openings[:count, 1] = centers - widths / 2, centers + widths / 2
        mask = np.zeros(4)
        mask[:count] = 1.
        rows.append(np.concatenate([[x, rng.uniform(.025, .045)], openings.ravel(), mask]))
    return np.concatenate([start, goal] + rows)


def generate_dataset(output, train=768, dev_model=128, dev_score=64, calibration=64,
                     test_locked=64, ood_locked=64, seed=2718, horizon=24, variants_per_parent=1):
    """Save padded positive route sets; counts denote independent parent scenes.

    All goal-condition variants of a parent remain in its split. RNG streams
    are split-specific, so changing TRAIN size cannot move locked examples.
    """
    counts = [train, dev_model, dev_score, calibration, test_locked, ood_locked]
    if any(int(n) != n or n < 0 for n in counts) or sum(counts) < 1:
        raise ValueError("nonnegative integer counts, at least one parent required")
    if int(variants_per_parent) != variants_per_parent or variants_per_parent < 1 or horizon < 8:
        raise ValueError("positive integer variants_per_parent and horizon>=8 required")
    output = Path(output)
    if output.is_dir():
        output = output / (VERSION + ".npz")
    output = output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: [] for key in ("scenes", "paths", "modes", "path_mask", "splits", "parent_ids", "scene_ids", "instructions", "path_x")}
    for split_index, (split, count) in enumerate(zip(SPLITS, counts)):
        rng = np.random.default_rng(np.random.SeedSequence([seed, split_index]))
        for parent_index in range(int(count)):
            parent_scene = _sample_scene(rng, ood=split == "OOD_LOCKED")
            parent_id = "%s_%s_%05d" % (VERSION, split, parent_index)
            for variant in range(int(variants_per_parent)):
                scene = parent_scene.copy()
                if variant:
                    scene[4] = rng.uniform(-.75, .75)
                _, _, walls = unpack_scene(scene)
                n0, n1 = [int(row[10:14].sum()) for row in walls]
                paths = np.zeros((MAX_REFERENCES, horizon, 3), dtype=np.float32)
                modes = np.full(MAX_REFERENCES, -1, dtype=np.int64)
                mask = np.zeros(MAX_REFERENCES, dtype=bool)
                for index, (first, second) in enumerate((a, b) for a in range(n0) for b in range(n1)):
                    paths[index] = route_from_openings(scene, first, second, horizon)
                    modes[index], mask[index] = 4 * first + second, True
                # Validate the float32 representation actually stored on disk.
                scene = scene.astype(np.float32)
                if not np.all(path_validity(paths[mask], scene)["valid"]):
                    raise RuntimeError("generated positive reference failed exact collision/length checks")
                if not np.array_equal(route_modes(paths[mask], scene), modes[mask]):
                    raise RuntimeError("generated reference used an unexpected opening sequence")
                for key, value in dict(scenes=scene, paths=paths, modes=modes, path_mask=mask,
                        splits=split, parent_ids=parent_id, scene_ids=parent_id + "_c%02d" % variant,
                        instructions="Reach the specified goal through safe openings in both walls.",
                        path_x=path_x(scene, horizon).astype(np.float32)).items():
                    arrays[key].append(value)
    arrays = {key: np.asarray(value) for key, value in arrays.items()}
    np.savez_compressed(str(output), **arrays)
    manifest = dict(version=VERSION, seed=seed, parent_counts=dict(zip(SPLITS, counts)),
        examples=len(arrays["scenes"]), variants_per_parent=variants_per_parent, horizon=horizon,
        sha256=hashlib.sha256(output.read_bytes()).hexdigest(), oracle_geometry=True,
        observation_benchmark=False, reference_semantics="one positive representative per opening pair; not total continuous solution count",
        split_unit="independent parent layout; all goal variants remain together",
        model_features="start3, goal3, two rows of [wall_x, halfwidth, opening_bounds8, mask4]",
        validation=dict(clearance=CLEARANCE, endpoint_tolerance=ENDPOINT_TOL, max_length_ratio=MAX_LENGTH_RATIO,
                        collision="all closed segments against all closed clearance-expanded AABBs"),
        mode_count_histogram={str(n): int(np.sum(arrays["path_mask"].sum(1) == n)) for n in np.unique(arrays["path_mask"].sum(1))})
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return output


def load_dataset(path, split=None):
    with np.load(str(path), allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if split is not None:
        if split not in SPLITS:
            raise ValueError("unknown split: %s" % split)
        mask = arrays["splits"] == split
        arrays = {name: value[mask] for name, value in arrays.items()}
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/multigate_v1.npz")
    for name, default in zip(("train", "dev-model", "dev-score", "calibration", "test-locked", "ood-locked"), (768, 128, 64, 64, 64, 64)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--variants-per-parent", type=int, default=1)
    print(generate_dataset(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
