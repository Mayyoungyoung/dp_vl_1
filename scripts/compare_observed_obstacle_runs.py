"""Paired CPU analysis of all TRAIN/DEV obstacle predictions, without selection.

Figures show every target in every predeclared DEV parent. Box-only tip metrics
remain separate from unverified full-robot/whole-environment execution validity.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

try:
    from .evaluate_observed_obstacles import evaluate, jsonl, sha256
except ImportError:
    from evaluate_observed_obstacles import evaluate, jsonl, sha256


COMPARISON_METRICS = ("semantic_goal_accuracy", "AnySemanticGoalAtK", "endpoint_error_m",
    "TipClearAtK", "EventSequenceCorrectAtK", "TipValidAtK", "AnyTipValidAtK",
    "UniqueClassifiedTipValidAtK", "UnknownTypeTipValidCount", "DuplicateClassifiedTipValidCount",
    "KnownReferenceTypeCoverageAtK", "mean_path_length_m")

# AUX is a reserved Windows device name even inside a directory. Keep the
# statistical/run alias for compatibility, but emit portable artifact paths.
OUTPUT_DIRECTORY_ALIASES = {"plain": "plain", "aux": "auxiliary"}


def reference_audit(data):
    rows = jsonl(Path(data) / "supervision.jsonl")
    result = {}
    for split in ("TRAIN", "DEV_MODEL"):
        selected = [row for row in rows if row["split"] == split]
        known_counts = [len({tuple(mode) for mode in row["route_types"] if mode is not None}) for row in selected]
        result[split] = dict(examples=len(selected), reference_trajectories=sum(len(row["routes"]) for row in selected),
            examples_with_reference=sum(bool(row["routes"]) for row in selected),
            reference_count_distribution=dict(Counter(len(row["routes"]) for row in selected)),
            known_type_count_distribution=dict(Counter(known_counts)),
            at_least_two_known_type_examples=sum(count >= 2 for count in known_counts),
            unknown_reference_trajectories=sum(mode is None for row in selected for mode in row["route_types"]),
            known_type_frequency=dict(Counter("/".join(mode) for row in selected for mode in row["route_types"] if mode is not None)))
    return result


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        return {str(identifier): path.copy() for identifier, path in zip(archive["scene_ids"], archive["paths"])}


def gate_audit(rows):
    # Independent failure counts overlap, deliberately exposing simultaneous
    # semantic and collision errors instead of hiding them behind gate order.
    gates = ("finite_xyz", "finite_event_values", "semantic_goal_correct", "starts_at_current_state",
             "tip_segments_clear", "event_state_sequence_correct", "TipValid")
    return dict(candidate_budget=len(rows), independent_failures={gate: sum(not row[gate] for row in rows) for gate in gates},
                valid_unclassified=sum(row["TipValid"] and row["declared_passage_type"] is None for row in rows),
                type_frequency_all_submitted=dict(Counter("unknown" if row["declared_passage_type"] is None else
                    "/".join(row["declared_passage_type"]) for row in rows)))


def plot_all_dev(data, predictions, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    labels = {row["id"]: row for row in jsonl(Path(data) / "supervision.jsonl") if row["split"] == "DEV_MODEL"}
    parents = sorted({row["parent_id"] for row in labels.values()})
    ids = sorted(labels, key=lambda item: (labels[item]["parent_id"], labels[item]["semantic_targets"]["target_index"]))
    if len(parents) != 4 or len(ids) != 12:
        raise ValueError("this prespecified plot requires all four pilot DEV parents and all twelve target instructions")
    # Historical two-arm colors stay fixed; named traditional or new learned
    # controls can use the same complete, prespecified 12-instruction display.
    palette = ["#3677b7", "#d85f37", "#28976a", "#8656a8"]
    colors = {name: {"plain": "#3677b7", "aux": "#d85f37"}.get(name, palette[index % len(palette)])
              for index, name in enumerate(predictions)}
    for plane, axes_index in [("xy", (0, 1)), ("xz", (0, 2))]:
        figure, axes = plt.subplots(4, 3, figsize=(13, 15))
        for index, identifier in enumerate(ids):
            axis = axes.flat[index]
            label = labels[identifier]
            with np.load(Path(data) / label["verification_only"], allow_pickle=False) as archive:
                centers, halfsizes = archive["obstacle_centers"], archive["obstacle_halfsizes"]
            for center, halfsize in zip(centers, halfsizes):
                axis.add_patch(Rectangle((center[axes_index[0]] - halfsize[axes_index[0]], center[axes_index[1]] - halfsize[axes_index[1]]),
                    2 * halfsize[axes_index[0]], 2 * halfsize[axes_index[1]], color="#999999", alpha=.3))
            for reference_index, filename in enumerate(label["routes"]):
                with np.load(Path(data) / filename, allow_pickle=False) as archive:
                    reference = archive["xyz_24"]
                axis.plot(reference[:, axes_index[0]], reference[:, axes_index[1]], color="#777777", lw=.65,
                          linestyle=":", alpha=.6, label="positive references" if reference_index == 0 else None)
            for model, sample_paths in predictions.items():
                for candidate, path in enumerate(sample_paths[identifier]):
                    if np.isfinite(path).all():
                        axis.plot(path[:, axes_index[0]], path[:, axes_index[1]], color=colors[model], lw=1.15,
                                  alpha=.7, label=model if candidate == 0 else None)
                        axis.scatter(path[-1, axes_index[0]], path[-1, axes_index[1]], color=colors[model], s=14)
                    else:
                        axis.text(.02, .02 + candidate * .04, model + " nonfinite candidate", transform=axis.transAxes, fontsize=7)
            target = np.asarray(label["semantic_targets"]["centers"])[label["semantic_targets"]["target_index"]]
            axis.scatter(target[axes_index[0]], target[axes_index[1]], marker="*", c="black", s=70, label="specified goal (eval only)")
            axis.set_title(identifier, fontsize=9)
            axis.set_xlabel("world " + plane[0] + " (m)"); axis.set_ylabel("world " + plane[1] + " (m)")
            axis.grid(alpha=.15); axis.legend(fontsize=6, loc="best")
        figure.suptitle("All four repeatedly used DEV parents / all targets / all submitted candidates\n"
                        "Projection only: tip-box checks are numerical; full-robot and other-body validity unverified", fontsize=12)
        figure.tight_layout(rect=(0, 0, 1, .955))
        figure.savefig(Path(destination) / ("all_dev_routes_" + plane + ".png"), dpi=145)
        plt.close(figure)


def compare(data, plain, aux, output, make_plots=True):
    data, plain, aux, output = map(Path, (data, plain, aux, output))
    output.mkdir(parents=True, exist_ok=True)
    configs = {name: json.loads((run / "config.json").read_text()) for name, run in [("plain", plain), ("aux", aux)]}
    controlled = ("dataset_fingerprint", "feature_dim", "horizon", "candidates", "width", "depth", "steps", "seed",
                  "lr", "batch_size", "event_scale", "pooling", "pixel_stride", "point_width", "endpoint_residual_bound")
    differences = {key: [configs["plain"].get(key), configs["aux"].get(key)] for key in controlled if configs["plain"].get(key) != configs["aux"].get(key)}
    if differences:
        raise ValueError("paired training conditions differ: " + json.dumps(differences))
    reference = reference_audit(data)
    results, scene_deltas, gates = {}, {}, {}
    for folder, split in [("dev_model", "DEV_MODEL"), ("train", "TRAIN")]:
        results[split], gates[split] = {}, {}
        scene_rows, predictions = {}, {}
        for name, run in [("plain", plain), ("aux", aux)]:
            destination = output / OUTPUT_DIRECTORY_ALIASES[name] / folder
            predictions_file = run / folder / "predictions.npz"
            results[split][name] = evaluate(data, predictions_file, destination, split=split)
            scene_rows[name] = {row["scene_id"]: row for row in json.loads((destination / "per_scene.json").read_text())}
            gates[split][name] = gate_audit(json.loads((destination / "per_candidate.json").read_text()))
            predictions[name] = load_predictions(predictions_file)
        if set(predictions["plain"]) != set(predictions["aux"]) or any(predictions["plain"][key].shape != predictions["aux"][key].shape for key in predictions["plain"]):
            raise ValueError("paired output examples or K/H budgets differ")
        scene_deltas[split] = []
        for identifier in sorted(scene_rows["plain"]):
            a, b = scene_rows["plain"][identifier], scene_rows["aux"][identifier]
            delta = {key: None if a[key] is None or b[key] is None else b[key] - a[key] for key in COMPARISON_METRICS}
            scene_deltas[split].append(dict(scene_id=identifier, parent_id=a["parent_id"], aux_minus_plain=delta))
        if split == "DEV_MODEL" and make_plots:
            plot_all_dev(data, predictions, output)
    comparisons = {split: {key: dict(plain=values["plain"][key], aux=values["aux"][key],
        aux_minus_plain=None if values["plain"][key] is None or values["aux"][key] is None else values["aux"][key] - values["plain"][key])
        for key in COMPARISON_METRICS} for split, values in results.items()}
    result = dict(scope="single-seed paired DEVELOPMENT analysis; all four reused DEV parents, not independent final confirmation",
        box_only_warning="TipValid checks the supplied physical boxes only. Full arm, table/walls/other bodies, IK and execution remain unverified.",
        reference_audit=reference, matched_config_fields={key: configs["plain"].get(key) for key in controlled},
        training_objective_weights={name: config.get("grounding_weight") for name, config in configs.items()},
        config_hashes={"plain": sha256(plain / "config.json"), "aux": sha256(aux / "config.json")},
        comparisons=comparisons, independent_gate_failures=gates, per_scene_deltas=scene_deltas,
        model_run_dirs={"plain": str(plain), "aux": str(aux)},
        output_directory_aliases=OUTPUT_DIRECTORY_ALIASES,
        prediction_files_repaired_or_filtered=False)
    (output / "paired_comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    (output / "reference_type_audit.json").write_text(json.dumps(reference, indent=2), encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plain", type=Path)
    parser.add_argument("--aux", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    if args.reference_only:
        args.output.mkdir(parents=True, exist_ok=True)
        result = reference_audit(args.data)
        (args.output / "reference_type_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif not args.plain or not args.aux:
        parser.error("--plain and --aux required for paired prediction evaluation")
    else:
        result = compare(args.data, args.plain, args.aux, args.output, not args.no_plots)
    print(json.dumps(result.get("comparisons", result), indent=2))


if __name__ == "__main__":
    main()
