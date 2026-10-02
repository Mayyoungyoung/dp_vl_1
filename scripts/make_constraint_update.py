"""Prepare developmental single-opening closures using a real frozen joint4.

Run from an immutable release. This script is CPU-only by default and refuses
locked data or overwriting an archive. Current weights were trained at K4;
their K1/K2 forwards are explicitly zero-adaptation diagnostics.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from routeset.common import decode_paths, sha256, write_json
from routeset.constraint_update import VERSION, derive_changes, update_metrics, aggregate_parents
from routeset.models import SetRegressor
from routeset.multigate import load_dataset


@torch.no_grad()
def predict(model, scenes, k):
    values = []
    for start in range(0, len(scenes), 32):
        scene = torch.as_tensor(scenes[start:start + 32])
        values.append(decode_paths(model(scene, k), scene).numpy())
    return np.concatenate(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("threads must be 1--4")
    output = Path(args.output).with_suffix(".npz")
    if output.exists():
        raise RuntimeError("Archive exists; use a new version/path, never overwrite research data")
    torch.set_num_threads(args.threads)
    data = load_dataset(args.data)
    if set(data["splits"].tolist()) - {"TRAIN", "DEV_MODEL"}:
        raise ValueError("use development.npz; do not load the locked archive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config["objective"] != "saturation" or config["candidates"] != 4 or config["cond_dim"] != 34:
        raise ValueError("expected actual controlled saturation ordinary joint4 checkpoint")
    model = SetRegressor(34, config["horizon"], 4, config["width"], config["depth"]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    start = time.perf_counter()
    old = predict(model, data["scenes"], 4)
    changed = derive_changes(data, old)
    diagnostics = {}
    dev = np.flatnonzero(changed["splits"] == "DEV_MODEL")
    for k in (1, 2, 4):
        changed["frozen_new_b%d" % k] = predict(model, changed["scenes"], k)
    for name, k in (("no_update", 0), ("frozen_k1_zero_adaptation", 1), ("frozen_k2_zero_adaptation", 2), ("joint4_recompute", 4)):
        predictions = changed["frozen_new_b%d" % k][dev] if k else np.zeros((len(dev), 0, config["horizon"], 3), dtype=np.float32)
        rows = update_metrics(changed["old_paths"][dev], predictions, changed["scenes"][dev],
                              changed["modes"][dev], changed["path_mask"][dev])
        for row, idx in zip(rows, dev):
            row.update(scene_id=str(changed["scene_ids"][idx]), parent_id=str(changed["parent_ids"][idx]))
        diagnostics[name] = dict(summary=aggregate_parents(rows), per_scene=rows)
    # Time an actual two-request CPU path, not cached throughput. Same geometry
    # tensors, checks and model used for the frozen controls; no Qwen claim.
    from routeset.constraint_update import checked_context
    from routeset.multigate import path_validity
    timings = {}
    for k in (0, 1, 2, 4):
        elapsed = []
        idx = dev[0]
        for repeat in range(12):
            t = time.perf_counter()
            original = predict(model, changed["old_scenes"][idx:idx + 1], 4)
            checked_context(original, changed["old_scenes"][idx:idx + 1], changed["scenes"][idx:idx + 1])
            fresh = predict(model, changed["scenes"][idx:idx + 1], k) if k else original[:, :0]
            path_validity(np.concatenate([original[0], fresh[0]]), changed["scenes"][idx])
            if repeat >= 2:
                elapsed.append((time.perf_counter() - t) * 1000)
        timings[str(k)] = dict(two_request_head_and_checks_cpu_ms_median=float(np.median(elapsed)),
                              two_request_head_and_checks_cpu_ms_p95=float(np.percentile(elapsed, 95)))
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **changed)
    from routeset.constraint_update import training_targets
    train_ids = np.flatnonzero(changed['splits'] == 'TRAIN')
    proxy_diagnostic = {}
    for k in (1, 2):
        labels = training_targets(changed, train_ids, k)
        proxy_diagnostic['b%d' % k] = dict(mean_point_edit_fraction=float(labels['proxy'].mean()),
            fraction_routes_editing_all_points=float(labels['proxy'].all(-1).mean()),
            fraction_routes_editing_no_points=float((~labels['proxy'].any(-1)).mean()),
            threshold_m=.02, scope='TRAIN-only source-nearest positive distance proxy; not causal impact truth')
    manifest = dict(version=VERSION, dataset_sha256=sha256(output), source_dataset_sha256=sha256(args.data),
        frozen_checkpoint_sha256=sha256(args.checkpoint), frozen_checkpoint_path=str(Path(args.checkpoint).resolve()),
        frozen_checkpoint_step=checkpoint["step"], frozen_training_config=config,
        code_commit=os.environ.get("CODE_COMMIT", "unrecorded"),
        parent_counts={split: len(set(changed["parent_ids"][changed["splits"] == split].tolist())) for split in ("TRAIN", "DEV_MODEL")},
        change_counts={split: int((changed["splits"] == split).sum()) for split in ("TRAIN", "DEV_MODEL")},
        old_invalid_after_change_histogram={str(n): int(((~changed["new_valid"]).sum(1) == n).sum()) for n in range(5)},
        elapsed_s=time.perf_counter() - start, threads=args.threads, timings=timings,
        training_proxy_diagnostic=proxy_diagnostic,
        model_inputs="old/new exact geometry, four actual old predictions, common whole-route validity; no reference/mode/edit labels",
        intervention="every single closure for R>=6 with another opening remaining in that wall; physical indices preserved",
        split_unit="original independent parent; no resplit; equal-parent sampling and aggregation",
        budget="old4 + each fresh complete route, including invalid/discarded routes; no hidden repair",
        limitations="controlled known geometry/endpoints only; frozen K1/K2 heads were trained at K4, zero-adaptation controls")
    write_json(output.with_suffix(".manifest.json"), manifest)
    write_json(output.with_suffix(".controls.json"), diagnostics)
    print(json.dumps(manifest), flush=True)


if __name__ == "__main__":
    main()
