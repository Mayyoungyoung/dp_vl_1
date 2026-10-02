"""Actual K2-trained ordinary baseline, same prior TRAIN-only closure audit.

The original K4 audit and results remain unchanged. Reference selections are
reused from their hash-verified receipt, not retuned or replaced for this run.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json
from routeset.multigate import unpack_scene
from scripts.audit_multigate_closure_opportunity import close_for_diagnostic, load_train_only, set_metrics


PROTOCOL = "multigate_k2_saturation_budget_check_v1"


def validate_training_config(trained, registered):
    expected = dict(registered["training"], horizon=24, cond_dim=34,
                    dataset_sha256=registered["data_sha256"], selection_split="DEV_MODEL")
    if registered["protocol"] != PROTOCOL or expected["candidates"] != 2:
        raise ValueError("genuinely trained K2 registration required")
    for key, value in expected.items():
        if trained.get(key) != value:
            raise ValueError("actual K2 training configuration mismatch: " + key)
    if expected["steps"] * expected["batch_size"] * 2 != registered["gradient_target_slots"]:
        raise ValueError("K2 path exposure registration differs")


def read_prior(directory, data_hash):
    directory = Path(directory)
    provenance = json.loads((directory / "provenance.json").read_text())
    if provenance["data_sha256"] != data_hash or provenance["config"]["expected_parents"] != 768:
        raise ValueError("prior audit belongs to another source")
    for name, expected in provenance["outputs_sha256"].items():
        path = (directory / name).resolve()
        if path.parent != directory.resolve() or sha256(path) != expected:
            raise ValueError("original audit artifact changed")
    summary = json.loads((directory / "summary.json").read_text())
    if summary["protocol"] != "multigate_closure_opportunity_train_v1" or summary["split"] != "TRAIN":
        raise ValueError("unchanged original TRAIN audit required")
    parents = json.loads((directory / "parents.json").read_text())
    closures = json.loads((directory / "closures.json").read_text())
    if len(parents) != 768 or len(set(p["parent_id"] for p in parents)) != 768:
        raise ValueError("original TRAIN768 identity incomplete")
    return parents, closures, provenance


def evaluate_k2(data, predictions, prior_parents, prior_closures):
    if predictions.shape != (len(data["scenes"]), 2, 24, 3):
        raise ValueError("exactly two H24 actual predictions per TRAIN scene required")
    identities = [(str(p), str(s)) for p, s in zip(data["parent_ids"], data["scene_ids"])]
    if identities != [(p["parent_id"], p["scene_id"]) for p in prior_parents]:
        raise ValueError("original TRAIN parent order changed")
    old = {(r["parent_id"], r["closed_wall"], r["closed_opening"]): r for r in prior_closures}
    if len(old) != len(prior_closures):
        raise ValueError("duplicate original closure")
    parents, closures = [], []
    for i, (value, prior) in enumerate(zip(data["scenes"], prior_parents)):
        parent_id, scene_id = identities[i]
        mask = data["path_mask"][i]
        if prior["reference_modes"] != data["modes"][i, mask].tolist():
            raise ValueError("original reference identity changed")
        _, _, walls = unpack_scene(value)
        rows = []
        for wall in (0, 1):
            for gap in np.flatnonzero(walls[wall, 10:14] > .5):
                identity = (parent_id, wall, int(gap))
                previous = old.pop(identity)
                metrics = set_metrics(predictions[i], close_for_diagnostic(value, wall, int(gap)))
                if not previous["has_solution"] and metrics["any_valid"]:
                    raise ValueError("actual K2 passed a completely blocked wall")
                row = dict(parent_id=parent_id, scene_id=scene_id, closed_wall=wall, closed_opening=int(gap),
                           has_solution=previous["has_solution"], known_remaining_types=previous["known_remaining_types"],
                           actual_k2=metrics, original_controls=previous["metrics"])
                rows.append(row)
                closures.append(row)
        solvable = [r for r in rows if r["has_solution"]]
        if len(rows) != prior["closure_count"] or len(solvable) != prior["solvable_closure_count"]:
            raise ValueError("original closure denominator changed")
        parents.append(dict(parent_id=parent_id, scene_id=scene_id, r_vs_k2=prior["r_vs_k2"],
            reference_types=prior["reference_types"], closure_count=len(rows), solvable_closure_count=len(solvable),
            unsolvable_closure_count=len(rows)-len(solvable), static=set_metrics(predictions[i], value),
            any_valid_all_closures=float(np.mean([r["actual_k2"]["any_valid"] for r in rows])),
            any_valid_solvable_closures=float(np.mean([r["actual_k2"]["any_valid"] for r in solvable])) if solvable else None,
            valid_rate_solvable_closures=float(np.mean([r["actual_k2"]["valid_rate"] for r in solvable])) if solvable else None,
            unique_valid_solvable_closures=float(np.mean([r["actual_k2"]["unique_valid"] for r in solvable])) if solvable else None,
            original_controls=prior["methods"]))
    if old:
        raise ValueError("not every original closure evaluated")
    return parents, closures


def summarize(rows):
    conditional = [r for r in rows if r["solvable_closure_count"]]
    controls = list(rows[0]["original_controls"]) if rows else []
    return dict(parents=len(rows), evaluable_parents=len(conditional),
        closures=sum(r["closure_count"] for r in rows), solvable_closures=sum(r["solvable_closure_count"] for r in rows),
        unsolvable_closures=sum(r["unsolvable_closure_count"] for r in rows),
        static={key:float(np.mean([r["static"][key] for r in rows])) if rows else None for key in ("valid_rate", "any_valid", "unique_valid")},
        actual_k2={key:float(np.mean(values)) if values else None for key in ("any_valid_all_closures", "any_valid_solvable_closures", "valid_rate_solvable_closures", "unique_valid_solvable_closures")
                   for values in [[r[key] for r in rows if r[key] is not None]]},
        paired_any_valid_k2_minus_control={name:dict(
            mean=float(np.mean(values)) if values else None, positive_parents=int(np.sum(np.asarray(values) > 1e-12)),
            negative_parents=int(np.sum(np.asarray(values) < -1e-12)), equal_parents=int(np.sum(np.abs(values) <= 1e-12)))
            for name in controls for values in [[r["any_valid_solvable_closures"]-r["original_controls"][name]["any_valid_solvable_closures"] for r in conditional]]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("fresh audit output required")
    start = time.perf_counter()
    registered = json.loads(args.registration.read_text())
    config = json.loads((args.run / "config.json").read_text())
    validate_training_config(config, registered)
    complete = json.loads((args.run / "status.json").read_text())
    summary = json.loads((args.run / "summary.json").read_text())
    if complete["status"] != "completed" or complete["exit_code"] != 0 or complete["step"] != config["steps"]:
        raise ValueError("completed actual K2 training required")
    if summary["gradient_target_slots"] != registered["gradient_target_slots"]:
        raise ValueError("actual training exposure changed")
    checkpoint_path = args.run / "best.pt"
    checkpoint_hash = sha256(checkpoint_path)
    if checkpoint_hash != summary["best_checkpoint_sha256"] or sha256(registered["data"]) != registered["data_sha256"]:
        raise ValueError("registered data/selected checkpoint hash changed")
    old_parents, old_closures, provenance = read_prior(registered["prior_audit"], registered["data_sha256"])
    source_root=Path(__file__).resolve().parents[1]
    for name, expected in provenance["source_sha256"].items():
        if sha256(source_root/name) != expected:
            raise ValueError("original K4 audit/checker/model implementation changed: " + name)
    data = load_train_only(registered["data"], 768)
    import torch
    from routeset.common import decode_paths
    from routeset.models import SetRegressor
    torch.set_num_threads(1)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_training_config(saved["config"], registered)
    if saved["step"] != summary["best_step"]:
        raise ValueError("original static selection receipt changed")
    model = SetRegressor(34, 24, 2, config["width"], config["depth"]).eval()
    model.load_state_dict(saved["model"], strict=True)
    predictions = []
    with torch.no_grad():
        for first in range(0, len(data["scenes"]), 32):
            scene = torch.as_tensor(data["scenes"][first:first+32])
            predictions.append(decode_paths(model(scene, 2), scene).numpy())
    predictions = np.concatenate(predictions)
    parents, closures = evaluate_k2(data, predictions, old_parents, old_closures)
    if sha256(checkpoint_path) != checkpoint_hash or sha256(registered["data"]) != registered["data_sha256"]:
        raise ValueError("model or data changed during read-only analysis")
    result = dict(protocol=PROTOCOL, split="TRAIN", checkpoint_step=saved["step"], checkpoint_sha256=checkpoint_hash,
        actual_forward_candidate_states=len(predictions)*2, retained_paths_per_request=2, generated_paths_per_request=2,
        overall=summarize(parents), by_r_vs_k2={key:summarize([r for r in parents if r["r_vs_k2"]==key]) for key in ("less", "equal", "more")},
        elapsed_cpu_wall_s=time.perf_counter()-start, gpu_hours=0, training=False,
        research_scope="genuinely K2-trained ordinary budget baseline; no new joint-risk objective; old reference-pool upper bound remains privileged",
        comparison_scope=registered["comparison_scope"], prior_audit_provenance_sha256=sha256(Path(registered["prior_audit"])/"provenance.json"))
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/"train_k2_predictions.npz", paths=predictions, scene_ids=data["scene_ids"], parent_ids=data["parent_ids"])
    write_json(args.output/"parents.json",parents)
    write_json(args.output/"closures.json",closures)
    write_json(args.output/"summary.json",result)
    write_json(args.output/"provenance.json",dict(source_commit=os.environ.get("CODE_COMMIT"), registration=registered,
        registration_sha256=sha256(args.registration), checkpoint_sha256=checkpoint_hash, data_sha256=sha256(registered["data"]),
        original_audit_provenance=provenance,
        source_sha256={name:sha256(source_root/name) for name in ("scripts/audit_multigate_k2_closure.py", "scripts/audit_multigate_closure_opportunity.py", "routeset/models.py", "routeset/multigate.py", "routeset/geometry.py", "routeset/common.py")},
        outputs_sha256={p.name:sha256(p) for p in args.output.iterdir() if p.is_file()}))
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
