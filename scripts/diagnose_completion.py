"""CPU-only fixed-checkpoint DEV interventions, not a training ablation.

Load route modules from an explicit immutable release; never read locked data.
The source script may be copied to a content-addressed analysis path so the
training release remains unchanged. Outputs record the script SHA256.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    sys.path.insert(0, str(args.release_root.resolve()))
    from routeset.common import decode_paths
    from routeset.completion import CompletionRegressor, draft_validity
    from routeset.multigate import load_dataset, route_metrics, route_modes, wall_boxes

    data = load_dataset(args.data)
    if set(data["splits"].tolist()) != {"TRAIN", "DEV_MODEL"}:
        raise ValueError("Only the physical TRAIN/DEV_MODEL partition is allowed")
    ids = np.flatnonzero(data["splits"] == "DEV_MODEL")
    scene_ids = data["scene_ids"][ids]
    scenes = data["scenes"][ids]
    source = np.load(args.runs / "attention/dev_model/two_valid/predictions.npz")
    if not np.array_equal(source["scene_ids"], scene_ids):
        raise ValueError("Saved predictions must align exactly with DEV_MODEL")
    original = source["drafts"].copy()
    original_present = source["draft_present"].copy()
    original_valid = source["draft_valid"].copy()
    if not np.array_equal(original_valid, draft_validity(original, scenes, original_present)):
        raise ValueError("Saved gates do not match exact geometry")

    variants = {"two_valid": (original, original_present),
                "reordered": (original[:, ::-1].copy(), original_present[:, ::-1].copy())}
    single = original_present.copy()
    single[:, 1] = False
    variants["single_first"] = (original.copy(), single)
    duplicated = original.copy()
    duplicated[:, 1] = duplicated[:, 0]
    variants["duplicate_first"] = (duplicated, original_present.copy())
    invalid = original.copy()
    for row, scene in enumerate(scenes):
        lower, upper = wall_boxes(scene)
        invalid[row, 1, invalid.shape[-2] // 2] = (lower[0] + upper[0]) / 2
    invalid_gate = draft_validity(invalid, scenes, original_present)
    if invalid_gate[:, 1].any():
        raise ValueError("The deliberately invalid second draft must be checked invalid")
    variants["invalid_second"] = (invalid, original_present.copy())
    near = duplicated.copy()
    wave = .002 * np.sin(np.linspace(0, 2 * np.pi, near.shape[-2]))
    near[:, 1, :, 1] += wave.astype(np.float32)[None]
    variants["near_duplicate_first"] = (near, original_present.copy())

    def metrics(paths, drafts, present, valid):
        checks = route_metrics(paths, scenes, data["modes"][ids], data["path_mask"][ids])
        provided, additional = [], []
        for row, scene in enumerate(scenes):
            labels = route_modes(drafts[row], scene)
            known = set(labels[valid[row] & (labels >= 0)].tolist())
            new = set(checks["per_scene"]["modes"][row].tolist()) - {-1}
            provided.append(len(known))
            additional.append(len(new - known))
        return {"valid": checks["valid_rate"], "unique_valid": checks["unique_valid"],
                "collision": checks["collision_rate"],
                "unclassified_valid": checks["unclassified_valid_rate"],
                "additional_unique_valid": float(np.mean(additional)),
                "provided_unique_valid": float(np.mean(provided)),
                "present_drafts": float(present.sum(1).mean()),
                "valid_drafts": float(valid.sum(1).mean())}

    results = {"script_sha256": digest(__file__), "data_sha256": digest(args.data),
               "release_root": str(args.release_root), "split": "DEV_MODEL",
               "scene_count": len(ids), "cpu_threads": 1, "device": "cpu",
               "new_candidates": 2, "near_duplicate_y_amplitude": .002,
               "interpretation": "fixed-checkpoint distribution interventions; not training ablations",
               "models": {}}
    for trained_pool in ("attention", "coverage"):
        checkpoint_path = args.runs / trained_pool / "best.pt"
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = ckpt["config"]
        if config["dataset_sha256"] != results["data_sha256"]:
            raise ValueError("Checkpoint belongs to a different dataset")
        model = CompletionRegressor(config["cond_dim"], config["horizon"], config["width"],
                                    config["depth"], trained_pool).eval()
        model.load_state_dict(ckpt["model"])
        stored = np.load(args.runs / trained_pool / "dev_model/two_valid/predictions.npz")
        if not np.array_equal(stored["drafts"], original):
            raise ValueError("Methods did not use identical saved drafts")
        model_results = {"checkpoint_sha256": digest(checkpoint_path), "step": ckpt["step"],
                         "code_commit": config["code_commit"], "pools": {}}
        for pool in (trained_pool, "coverage" if trained_pool == "attention" else "attention"):
            model.mechanism = pool
            predictions, stats = {}, {}
            with torch.no_grad():
                for name, (drafts, present) in variants.items():
                    valid = draft_validity(drafts, scenes, present)
                    output = []
                    for start in range(0, len(ids), 32):
                        stop = start + 32
                        scene_tensor = torch.as_tensor(scenes[start:stop])
                        output.append(decode_paths(model(scene_tensor, torch.as_tensor(drafts[start:stop]),
                                                         torch.as_tensor(valid[start:stop]), 2), scene_tensor).numpy())
                    paths = np.concatenate(output)
                    predictions[name] = paths
                    stats[name] = metrics(paths, drafts, present, valid)
            diagnostics = {}
            for name, other in (("reordered", "two_valid"), ("duplicate_first", "single_first"),
                                ("invalid_second", "single_first"), ("near_duplicate_first", "duplicate_first")):
                delta = predictions[name] - predictions[other]
                diagnostics[name + "_vs_" + other] = {
                    "maximum_absolute_output_change": float(np.abs(delta).max()),
                    "mean_route_point_l2_change": float(np.linalg.norm(delta, axis=-1).mean()),
                    "valid_delta": stats[name]["valid"] - stats[other]["valid"],
                    "additional_delta": stats[name]["additional_unique_valid"] - stats[other]["additional_unique_valid"]}
            if pool == trained_pool:
                model_results["cpu_vs_saved_gpu_max_abs"] = float(np.abs(predictions["two_valid"] - stored["paths"]).max())
            model_results["pools"][pool] = {"metrics": stats, "interventions": diagnostics}
        results["models"][trained_pool] = model_results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
