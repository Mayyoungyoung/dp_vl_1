"""Evaluation-only semantics and tip geometry for observed obstacle routes.

Inputs are saved predictions. Privileged obstacle/goal labels are opened only
by this evaluator, never by a generator. TipValid metrics do NOT certify arm
collision, IK, controller tracking, or real robot execution.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:
    from .collect_obstacle_reach import crossing_signature, tip_polyline_clear
except ImportError:
    from collect_obstacle_reach import crossing_signature, tip_polyline_clear


PROTOCOL = "observed_obstacle_tip_eval_v1"


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scene_metrics(paths, opened, current, geometry, specification, reference_types, clearance=.02, scores=None):
    """Evaluate all submitted K slots, including nonfinite/invalid candidates."""
    paths = np.asarray(paths)
    if paths.ndim != 3 or paths.shape[-1] != 3 or paths.shape[1] < 2 or not len(paths):
        raise ValueError("paths must be [K,H>=2,3] with K>=1")
    if opened is not None and np.asarray(opened).shape != paths.shape[:2]:
        raise ValueError("gripper_open must be [K,H]")
    if scores is not None and np.asarray(scores).shape != (len(paths),):
        raise ValueError("scores must be [K]")
    centers, halfsizes = np.asarray(geometry["obstacle_centers"]), np.asarray(geometry["obstacle_halfsizes"])
    if centers.shape != halfsizes.shape or centers.ndim != 2 or centers.shape[-1] != 3 or not len(centers):
        raise ValueError("matching nonempty obstacle arrays [O,3] required")
    if not np.isfinite(centers).all() or not np.isfinite(halfsizes).all() or np.any(halfsizes <= 0):
        raise ValueError("finite centers and positive obstacle dimensions required")
    goals = np.asarray(specification["centers"], dtype=np.float64)
    target, tolerance = int(specification["target_index"]), float(specification["tolerance"])
    if goals.ndim != 2 or goals.shape[1] != 3 or not 0 <= target < len(goals) or tolerance <= 0 or not np.isfinite(goals).all():
        raise ValueError("valid evaluation-only semantic target specification required")
    known_reference = {tuple(mode) for mode in reference_types if mode is not None}
    candidates = []
    expected_open = bool(current["gripper_open"] > .5)
    for index, path in enumerate(paths):
        finite = bool(np.isfinite(path).all())
        semantic = start_ok = clear = events_ok = False
        length = endpoint_error = start_error = None
        mode = None
        if finite:
            distances = np.linalg.norm(goals - path[-1], axis=-1)
            endpoint_error = float(distances[target])
            semantic = bool(distances.argmin() == target and endpoint_error <= tolerance)
            start_error = float(np.linalg.norm(path[0] - np.asarray(current["gripper_pose"])[:3]))
            start_ok = start_error <= .005
            clear = bool(tip_polyline_clear(path, centers, halfsizes, clearance))
            length = float(np.linalg.norm(np.diff(path, axis=0), axis=-1).sum())
            mode = crossing_signature(path, centers, halfsizes, clearance)
        event_finite = opened is not None and bool(np.isfinite(opened[index]).all())
        if event_finite:
            events_ok = bool(np.all((opened[index] > .5) == expected_open))
        valid = finite and semantic and start_ok and clear and events_ok
        candidates.append(dict(candidate=index, finite_xyz=finite, finite_event_values=event_finite,
            semantic_goal_correct=semantic, endpoint_error_m=endpoint_error, start_error_m=start_error,
            starts_at_current_state=start_ok, tip_segments_clear=clear, event_state_sequence_correct=events_ok,
            length_m=length, declared_passage_type=mode, TipValid=valid,
            classified_tip_valid=valid and mode is not None))
    valid_types = {tuple(row["declared_passage_type"]) for row in candidates if row["classified_tip_valid"]}
    valid_count = sum(row["TipValid"] for row in candidates)
    known_count = sum(row["classified_tip_valid"] for row in candidates)
    selected = None
    selected_valid = None
    if scores is not None:
        finite_scores = np.isfinite(scores)
        selected = int(np.where(finite_scores, scores, -np.inf).argmax()) if finite_scores.any() else None
        selected_valid = float(selected is not None and candidates[selected]["TipValid"])
    lengths = [row["length_m"] for row in candidates if row["length_m"] is not None]
    errors = [row["endpoint_error_m"] for row in candidates if row["endpoint_error_m"] is not None]
    result = dict(candidates=len(paths), TipValidAtK=valid_count / len(paths), AnyTipValidAtK=float(valid_count > 0),
        UniqueClassifiedTipValidAtK=len(valid_types), UnknownTypeTipValidCount=valid_count - known_count,
        DuplicateClassifiedTipValidCount=known_count - len(valid_types),
        KnownReferenceTypeCoverageAtK=len(valid_types & known_reference) / len(known_reference) if known_reference else None,
        known_reference_types=len(known_reference), SelectedTipValidAtK=selected_valid, selected_index=selected,
        semantic_goal_accuracy=float(np.mean([row["semantic_goal_correct"] for row in candidates])),
        AnySemanticGoalAtK=float(any(row["semantic_goal_correct"] for row in candidates)),
        endpoint_error_m=float(np.mean(errors)) if errors else None,
        TipClearAtK=float(np.mean([row["tip_segments_clear"] for row in candidates])),
        StartCorrectAtK=float(np.mean([row["starts_at_current_state"] for row in candidates])),
        EventSequenceCorrectAtK=float(np.mean([row["event_state_sequence_correct"] for row in candidates])),
        mean_path_length_m=float(np.mean(lengths)) if lengths else None,
        format_failure_count=sum(not row["finite_xyz"] or not row["finite_event_values"] for row in candidates),
        ValidAtK=None, AnyValidAtK=None, UniqueValidAtK=None, SelectedValidAtK=None)
    return result, candidates


def evaluate(data_root, predictions_file, output, split="DEV_MODEL", allow_subset=False):
    data_root, predictions_file, output = map(Path, (data_root, predictions_file, output))
    manifest = json.loads((data_root / "manifest.json").read_text(encoding="utf-8-sig"))
    observations = jsonl(data_root / "observations.jsonl")
    observation_by_id = {row["id"]: row for row in observations}
    if len(observation_by_id) != len(observations):
        raise ValueError("duplicate observation id")
    labels_list = jsonl(data_root / "supervision.jsonl")
    labels = {row["id"]: row for row in labels_list}
    if len(labels) != len(labels_list):
        raise ValueError("duplicate supervision id")
    parent_splits = {}
    for row in observations:
        if set(row) != {"id", "parent_id", "split", "image", "instruction"}:
            raise ValueError("observation manifest violates the observation-only contract")
        previous = parent_splits.setdefault(row["parent_id"], row["split"])
        if previous != row["split"]:
            raise ValueError("parent scene appears in multiple splits")
    expected = {row["id"] for row in observations if row["split"] == split}
    with np.load(predictions_file, allow_pickle=False) as archive:
        paths = archive["paths"]
        ids = archive["scene_ids"].astype(str)
        opened = archive["gripper_open"] if "gripper_open" in archive else None
        scores = archive["scores"] if "scores" in archive else None
        parent_ids = archive["parent_ids"].astype(str) if "parent_ids" in archive else None
    if paths.ndim != 4 or ids.ndim != 1 or paths.shape[0] != len(ids) or len(set(ids)) != len(ids):
        raise ValueError("unique scene_ids and matching paths [N,K,H,3] required")
    if opened is not None and opened.shape != paths.shape[:3]:
        raise ValueError("gripper_open must match [N,K,H]")
    if scores is not None and scores.shape != paths.shape[:2]:
        raise ValueError("scores must match [N,K]")
    if parent_ids is not None and parent_ids.shape != ids.shape:
        raise ValueError("parent_ids must match scene_ids")
    if not set(ids).issubset(expected) or (not allow_subset and set(ids) != expected):
        raise ValueError("predictions must cover the requested split, including scenes with no reference; use --allow-subset only for explicit diagnostics")
    if not len(ids):
        raise ValueError("empty evaluation split")
    per_scene, per_candidate, source_hashes = [], [], {}
    for index, identifier in enumerate(ids):
        label = labels[identifier]
        if (label["split"] != split or label["parent_id"] != observation_by_id[identifier]["parent_id"] or
                (parent_ids is not None and parent_ids[index] != label["parent_id"])):
            raise ValueError("prediction/supervision parent or split mismatch")
        observation_file, verification_file = data_root / label["observation"], data_root / label["verification_only"]
        with np.load(observation_file, allow_pickle=False) as archive:
            current = {key: archive[key] for key in ("gripper_pose", "gripper_open")}
        with np.load(verification_file, allow_pickle=False) as archive:
            geometry = {key: archive[key] for key in ("obstacle_centers", "obstacle_halfsizes")}
        row, candidates = scene_metrics(paths[index], None if opened is None else opened[index], current, geometry,
            label["semantic_targets"], label.get("route_types", []),
            clearance=float(manifest["acceptance"]["tip_polyline_clearance_m"]),
            scores=None if scores is None else scores[index])
        row.update(scene_id=identifier, parent_id=label["parent_id"], reference_count=len(label["routes"]))
        per_scene.append(row)
        per_candidate.extend(dict(scene_id=identifier, parent_id=label["parent_id"], **candidate) for candidate in candidates)
        source_hashes[str(verification_file)] = sha256(verification_file)
        source_hashes[str(observation_file)] = sha256(observation_file)
    metrics = {}
    for key in per_scene[0]:
        if key in {"scene_id", "parent_id", "selected_index", "candidates", "reference_count", "known_reference_types"}:
            continue
        values = [row[key] for row in per_scene if row[key] is not None]
        metrics[key] = float(np.mean(values)) if values else None
    metrics.update(evaluation_protocol=PROTOCOL, examples=len(ids), requested_split_examples=len(expected),
        missing_prediction_examples=len(expected - set(ids)), parents=len({r["parent_id"] for r in per_scene}),
        candidates=int(paths.shape[1]), total_submitted_candidate_budget=int(paths.shape[0] * paths.shape[1]),
        total_format_failure_candidates=sum(row["format_failure_count"] for row in per_scene),
        total_tip_valid_candidates=sum(row["TipValid"] for row in per_candidate),
        examples_without_reference=sum(not row["reference_count"] for row in per_scene),
        reference_coverage_evaluable_examples=sum(bool(row["known_reference_types"]) for row in per_scene),
        input_contract="saved predictions only; target and geometry opened exclusively for evaluation",
        validity_scope="TipValid requires correct target within original tolerance, current start within 5mm, constant reach event state, and exact full tip-segment checks against supplied physical boxes with original clearance; excludes full-arm/IK/execution validity",
        unverified_geometry="The added physical boxes are checked. Table, walls, other environment bodies, arm volume and IK feasibility are not certified by this evaluator.",
        unknown_type_policy="Valid unclassified routes remain valid; they do not contribute invented types",
        reference_policy="Coverage only of known classified positive reference types; no total-solution-count claim",
        selection_policy="optional scores rank the same submitted K; missing scores yield null, all nonfinite scores fail selection",
        paths_repaired_or_filtered=False, prediction_sha256=sha256(predictions_file), source_hashes=source_hashes,
        manifest_sha256=sha256(data_root / "manifest.json"), supervision_sha256=sha256(data_root / "supervision.jsonl"))
    requested_parents = (manifest["dev_parents"] if split == "DEV_MODEL" else
                         manifest["parents_requested"] - manifest["dev_parents"] if split == "TRAIN" else None)
    available_parents = len({row["parent_id"] for row in observations if row["split"] == split})
    metrics.update(requested_collection_parents_for_split=requested_parents,
                   available_observation_parents_for_split=available_parents,
                   collection_parents_without_observations=None if requested_parents is None else requested_parents - available_parents)
    output.mkdir(parents=True, exist_ok=True)
    for name, value in [("metrics", metrics), ("per_scene", per_scene), ("per_candidate", per_candidate)]:
        (output / (name + ".json")).write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    return metrics


def self_test():
    try:
        from .collect_obstacle_reach import collection_waypoints
    except ImportError:
        from collect_obstacle_reach import collection_waypoints
    centers, halfsizes = np.array([[0., 0., 1.]]), np.array([[.1, .1, .05]])
    start, goal = np.array([0., 0., 1.4]), np.array([0., 0., .6])
    left = np.vstack([start, collection_waypoints(goal, centers, halfsizes, ("negative_x",))])
    right = np.vstack([start, collection_waypoints(goal, centers, halfsizes, ("positive_x",))])
    collision = np.linspace(start, goal, len(left))
    paths = np.stack([left, right, left, collision])
    current = dict(gripper_pose=np.r_[start, [0., 0., 0., 1.]], gripper_open=1.)
    geometry = dict(obstacle_centers=centers, obstacle_halfsizes=halfsizes)
    semantic = dict(centers=[goal.tolist(), [.5, 0., .6]], target_index=0, tolerance=.03)
    metrics, rows = scene_metrics(paths, np.ones(paths.shape[:2]), current, geometry, semantic,
                                  [("negative_x",), ("positive_x",)], scores=np.array([0., 0., 0., 1.]))
    assert metrics["TipValidAtK"] == .75 and metrics["UniqueClassifiedTipValidAtK"] == 2
    assert metrics["DuplicateClassifiedTipValidCount"] == 1 and metrics["KnownReferenceTypeCoverageAtK"] == 1.
    assert metrics["SelectedTipValidAtK"] == 0. and metrics["ValidAtK"] is None
    paths[0, 1, 0] = np.nan
    metrics, _ = scene_metrics(paths, np.ones(paths.shape[:2]), current, geometry, semantic, [])
    assert metrics["format_failure_count"] == 1 and metrics["TipValidAtK"] == .5
    assert metrics["KnownReferenceTypeCoverageAtK"] is None and metrics["SelectedTipValidAtK"] is None
    print(json.dumps(dict(self_test="passed", protocol=PROTOCOL, simulator_launched=False)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split", default="DEV_MODEL", choices=("TRAIN", "DEV_MODEL", "DEV_SCORE", "CALIBRATION"))
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif args.data is None or args.predictions is None or args.output is None:
        parser.error("--data, --predictions and --output are required")
    else:
        print(json.dumps(evaluate(args.data, args.predictions, args.output, args.split, args.allow_subset), indent=2))


if __name__ == "__main__":
    main()
