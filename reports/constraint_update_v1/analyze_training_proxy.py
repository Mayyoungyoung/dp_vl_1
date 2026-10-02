"""Read-only TRAIN proxy diagnostic for bb3077; no development mask oracle.

Run with PYTHONPATH pointing at the recorded immutable release and CUDA hidden.
The injected reference is a training diagnostic, never a learned result.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from routeset.common import sha256, write_json
from routeset.constraint_update import aggregate_parents, training_targets, update_metrics
from routeset.multigate import load_dataset, path_validity, route_modes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise RuntimeError("use a fresh diagnostic output")
    torch.set_num_threads(1)
    data = load_dataset(args.data)
    if set(data["splits"].tolist()) - {"TRAIN", "DEV_MODEL"}:
        raise ValueError("development-only archive required")
    ids = np.flatnonzero(data["splits"] == "TRAIN")
    result = dict(dataset_sha256=sha256(args.data), analysis_script_sha256=sha256(__file__),
                  scope="TRAIN-only injected positive reference + proxy gate, not learned inference or a formal upper bound", arms={})
    for k in (1, 2):
        labels = training_targets(data, ids, k, threshold=.02)
        targets = labels["targets"]
        sources = labels["context"]["drafts"][:, :k]
        mixed = targets.copy()
        mixed[:, :, 1:-1] = np.where(labels["proxy"][..., None], targets[:, :, 1:-1], sources[:, :, 1:-1])
        valid = np.stack([path_validity(p, s)["valid"] for p, s in zip(mixed, data["scenes"][ids])])
        target_valid = np.stack([path_validity(p, s)["valid"] for p, s in zip(targets, data["scenes"][ids])])
        labels_after = np.stack([route_modes(p, s) for p, s in zip(mixed, data["scenes"][ids])])
        target_labels = np.stack([route_modes(p, s) for p, s in zip(targets, data["scenes"][ids])])
        summaries = {}
        for name, paths in (("injected_reference", targets), ("injected_reference_with_proxy_copy", mixed)):
            rows = update_metrics(data["old_paths"][ids], paths, data["scenes"][ids],
                                  data["modes"][ids], data["path_mask"][ids])
            for row, idx in zip(rows, ids):
                row.update(parent_id=str(data["parent_ids"][idx]), scene_id=str(data["scene_ids"][idx]))
            summaries[name] = aggregate_parents(rows)
        fractions = labels["proxy"].mean(-1)
        result["arms"]["b%d" % k] = dict(
            changed_scenes=len(ids), parent_count=len(set(data["parent_ids"][ids].tolist())),
            generated_routes=int(len(ids) * k), proxy_mean_edit_fraction=float(fractions.mean()),
            proxy_edit_fraction_quantiles=np.quantile(fractions, [0., .25, .5, .75, 1.]).tolist(),
            any_copied_interior_point_fraction=float((~labels["proxy"].all(-1)).mean()),
            proxy_copies_at_least_half_interior_fraction=float((fractions <= .5).mean()),
            injected_positive_valid_rate=float(target_valid.mean()),
            injected_proxy_copy_valid_rate=float(valid.mean()),
            injected_proxy_copy_mode_identity_rate=float((labels_after == target_labels).mean()),
            failed_injected_proxy_scene_ids=data["scene_ids"][ids[(~valid).any(-1)]].tolist(),
            parent_weighted_metrics=summaries)
    write_json(args.output, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
