"""TRAIN-only diagnostic of co-failure under every single opening closure.

This does not train, update, repair, or select neural candidates. joint4 is an
actual K4 forward; prefix2 is its unadapted first two outputs, still charged
four generated routes. Reference-pool selections are privileged diagnostics.
"""
import argparse
import itertools
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json
from routeset.multigate import path_validity, route_modes, unpack_scene


PROTOCOL = "multigate_closure_opportunity_train_v1"
METHODS = ("joint4_actual", "joint4_prefix2_zero_adaptation",
           "reference_farthest_k2", "reference_geometric_dpp_k2", "reference_response_oracle_k2")


def close_for_diagnostic(scene, wall, opening):
    """Allow closing the last gap; leave the production helper unchanged."""
    _, _, walls = unpack_scene(scene)
    if wall not in (0, 1) or opening not in range(4) or walls[wall, 10 + opening] <= .5:
        raise ValueError("must identify a present physical opening")
    result = np.array(scene, copy=True)
    result[6 + 14 * wall + 10 + opening] = 0
    return result


def _read_npy_rows(archive, key, rows, count):
    """Decode selected rows only, never instantiate other split payloads.

    ZIP seeking can internally decompress skipped bytes; no non-TRAIN scene,
    route or label row is converted to an array or inspected. Metadata is read
    first and locked-containing archives are rejected before payload access.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) == 0 or np.any(np.diff(rows) <= 0):
        raise ValueError("strictly increasing selected rows required")
    with archive.open(key + ".npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("unsupported NPY format")
        if not shape or shape[0] != count or fortran or dtype.hasobject or rows[-1] >= count or rows[0] < 0:
            raise ValueError("invalid row-addressable archive payload")
        row_size = int(np.prod(shape[1:], dtype=np.int64)) * dtype.itemsize
        offset = stream.tell()
        selected = []
        for index in rows:
            stream.seek(offset + int(index) * row_size)
            raw = stream.read(row_size)
            if len(raw) != row_size:
                raise ValueError("truncated selected TRAIN row")
            selected.append(np.frombuffer(raw, dtype=dtype).reshape(shape[1:]).copy())
    return np.stack(selected)


def load_train_only(path, expected_parents):
    with zipfile.ZipFile(path) as archive:
        metadata = {}
        for key in ("splits", "parent_ids", "scene_ids"):
            with archive.open(key + ".npy") as stream:
                metadata[key] = np.lib.format.read_array(stream, allow_pickle=False)
        splits = metadata["splits"]
        if splits.ndim != 1 or set(splits.tolist()) - {"TRAIN", "DEV_MODEL"}:
            raise ValueError("only physically separate TRAIN/DEV_MODEL source is permitted")
        if any(v.shape != splits.shape for v in metadata.values()):
            raise ValueError("metadata dimensions disagree")
        if len(set(metadata["scene_ids"].tolist())) != len(splits):
            raise ValueError("duplicate scene identity")
        roles = {}
        for parent, role in zip(metadata["parent_ids"], splits):
            if parent in roles and roles[parent] != role:
                raise ValueError("parent crosses roles")
            roles[parent] = role
        rows = np.flatnonzero(splits == "TRAIN")
        if len(rows) != expected_parents or len(set(metadata["parent_ids"][rows].tolist())) != expected_parents:
            raise ValueError("exact registered TRAIN parents, one scene per parent, required")
        result = {key: values[rows].copy() for key, values in metadata.items()}
        for key in ("scenes", "paths", "modes", "path_mask"):
            result[key] = _read_npy_rows(archive, key, rows, len(splits))
    if (result["scenes"].shape != (expected_parents, 34) or
            result["paths"].shape != (expected_parents, 16, 24, 3) or
            result["modes"].shape != (expected_parents, 16) or
            result["path_mask"].shape != (expected_parents, 16) or
            result["path_mask"].dtype != np.bool_):
        raise ValueError("registered H24 reference archive schema required")
    return result


def k2_selections(references, responses, solvable, sigma):
    """Fixed selections made before the particular closure is revealed.

    RBF unit-quality k-DPP MAP at K2 maximizes 1-exp(-d2/sigma2),
    strictly monotonic in squared RMS distance. Use the exact farthest pair
    also for DPP to avoid floating underflow introducing arbitrary ties.
    """
    refs = np.asarray(references, dtype=np.float64)
    response = np.asarray(responses, dtype=bool)
    solvable = np.asarray(solvable, dtype=bool)
    if refs.ndim != 3 or refs.shape[-1] != 3 or not len(refs) or not np.isfinite(refs).all() or sigma <= 0:
        raise ValueError("finite positive reference bank and sigma required")
    if response.shape != (len(solvable), len(refs)):
        raise ValueError("response matrix does not match bank")
    pairs = list(itertools.combinations(range(len(refs)), 2)) if len(refs) > 1 else [(0, 0)]
    distances = np.asarray([np.square(refs[i] - refs[j]).mean() for i, j in pairs])
    farthest_index = int(np.argmax(distances))
    covered = np.asarray([np.logical_or(response[:, i], response[:, j]) for i, j in pairs])
    counts = covered[:, solvable].sum(1) if solvable.any() else np.zeros(len(pairs), dtype=int)
    # Stable lexicographic reference-index order for all equal objective values.
    best_index = int(np.argmax(counts))
    pair = pairs[farthest_index]
    return dict(farthest=pair, geometric_dpp=pair, response_oracle=pairs[best_index],
                pair_count=len(pairs), farthest_squared_rms=float(distances[farthest_index]),
                geometric_dpp_determinant=float(-np.expm1(-distances[farthest_index] / sigma ** 2)),
                response_oracle_covered_solvable=int(counts[best_index]))


def set_metrics(paths, scene):
    checks = path_validity(paths, scene)
    labels = route_modes(paths, scene)
    valid = checks["valid"]
    return dict(any_valid=bool(valid.any()), valid_rate=float(valid.mean()),
                unique_valid=len(set(labels[valid & (labels >= 0)].tolist())),
                valid=valid.tolist(), collision=checks["collision"].tolist(),
                modes=labels.tolist(), candidate_count=len(paths))


def analyze_parent(scene, references, modes, predictions, parent_id, scene_id, sigma):
    refs, modes, predicted = np.asarray(references), np.asarray(modes), np.asarray(predictions)
    if predicted.shape != (4, refs.shape[1], 3):
        raise ValueError("four actual H-matched neural predictions required")
    _, _, walls = unpack_scene(scene)
    present = [(wall, int(gap)) for wall in (0, 1) for gap in np.flatnonzero(walls[wall, 10:14] > .5)]
    complete = sorted(4 * a + b for a in np.flatnonzero(walls[0, 10:14] > .5)
                      for b in np.flatnonzero(walls[1, 10:14] > .5))
    if sorted(modes.tolist()) != complete or len(set(modes.tolist())) != len(refs):
        raise ValueError("one known positive per present physical opening pair required")
    if not path_validity(refs, scene)["valid"].all() or not np.array_equal(route_modes(refs, scene), modes):
        raise ValueError("original positive reference geometry/type failed")
    changed = [close_for_diagnostic(scene, *slot) for slot in present]
    responses = np.stack([path_validity(refs, s)["valid"] for s in changed])
    theoretical = np.asarray([modes // 4 != gap if wall == 0 else modes % 4 != gap for wall, gap in present])
    if not np.array_equal(responses, theoretical):
        raise ValueError("reference response disagrees with unchanged exact checker")
    solvable = responses.any(1)
    selection = k2_selections(refs, responses, solvable, sigma)
    sets = dict(joint4_actual=predicted, joint4_prefix2_zero_adaptation=predicted[:2],
                reference_farthest_k2=refs[list(selection["farthest"])],
                reference_geometric_dpp_k2=refs[list(selection["geometric_dpp"])],
                reference_response_oracle_k2=refs[list(selection["response_oracle"])])
    static = {name: set_metrics(paths, scene) for name, paths in sets.items()}
    changes = []
    for number, (slot, new_scene) in enumerate(zip(present, changed)):
        metrics = {name: set_metrics(paths, new_scene) for name, paths in sets.items()}
        if not solvable[number] and any(row["any_valid"] for row in metrics.values()):
            raise ValueError("route passed a completely blocked full-height wall")
        changes.append(dict(parent_id=parent_id, scene_id=scene_id, closed_wall=slot[0], closed_opening=slot[1],
                            known_remaining_types=int(responses[number].sum()), has_solution=bool(solvable[number]),
                            metrics=metrics))
    parent = dict(parent_id=parent_id, scene_id=scene_id, reference_types=len(refs),
                  r_vs_k2="less" if len(refs) < 2 else "equal" if len(refs) == 2 else "more",
                  closure_count=len(changes), solvable_closure_count=int(solvable.sum()),
                  unsolvable_closure_count=int((~solvable).sum()), static=static, selections=selection,
                  reference_modes=modes.tolist(), reference_response_matrix=responses.tolist(),
                  methods={name: dict(
                      any_valid_all_closures=float(np.mean([c["metrics"][name]["any_valid"] for c in changes])),
                      any_valid_solvable_closures=float(np.mean([c["metrics"][name]["any_valid"] for c in changes if c["has_solution"]])) if solvable.any() else None,
                      valid_rate_solvable_closures=float(np.mean([c["metrics"][name]["valid_rate"] for c in changes if c["has_solution"]])) if solvable.any() else None,
                      unique_valid_solvable_closures=float(np.mean([c["metrics"][name]["unique_valid"] for c in changes if c["has_solution"]])) if solvable.any() else None)
                           for name in METHODS})
    return parent, changes


def aggregate(parents, threshold):
    def means(rows):
        if not rows:
            return dict(parents=0, methods={})
        return dict(parents=len(rows), closure_count=sum(p["closure_count"] for p in rows),
                    solvable_closure_count=sum(p["solvable_closure_count"] for p in rows),
                    unsolvable_closure_count=sum(p["unsolvable_closure_count"] for p in rows),
                    evaluable_parents=sum(p["solvable_closure_count"] > 0 for p in rows),
                    static={name: {metric: float(np.mean([p["static"][name][metric] for p in rows]))
                                   for metric in ("any_valid", "valid_rate", "unique_valid")}
                            for name in METHODS},
                    methods={name: {metric: float(np.mean(values)) if values else None
                                    for metric in rows[0]["methods"][name]
                                    for values in [[p["methods"][name][metric] for p in rows if p["methods"][name][metric] is not None]]}
                             for name in METHODS})
    overall = means(parents)
    paired = {}
    for control in METHODS[:-1]:
        values = [p["methods"]["reference_response_oracle_k2"]["any_valid_solvable_closures"] -
                  p["methods"][control]["any_valid_solvable_closures"] for p in parents if p["solvable_closure_count"]]
        paired[control] = dict(mean=float(np.mean(values)) if values else None,
                               positive_parents=int(np.sum(np.asarray(values) > 1e-12)),
                               negative_parents=int(np.sum(np.asarray(values) < -1e-12)),
                               equal_parents=int(np.sum(np.abs(values) <= 1e-12)), parents=len(values))
    geometry_gap = paired["reference_farthest_k2"]["mean"]
    prefix_gap = paired["joint4_prefix2_zero_adaptation"]["mean"]
    principal = geometry_gap is not None and geometry_gap >= threshold
    prefix = prefix_gap is not None and prefix_gap >= threshold
    return dict(overall=overall, by_r_vs_k2={key: means([p for p in parents if p["r_vs_k2"] == key]) for key in ("less", "equal", "more")},
                paired_reference_oracle_minus_control=paired,
                screening=dict(threshold=threshold, principal_control="reference_farthest_k2",
                    principal_gap_at_least_threshold=principal, unadapted_prefix_gap_at_least_threshold=prefix,
                    decision="train_genuine_K2_saturation_before_any_new_mechanism" if principal or prefix else "no_5pp_opportunity_stop",
                    mechanism_opportunity_beyond_geometric_selection=principal,
                    limitation="TRAIN-only label-assisted opportunity, neither attainable learned gain nor independent validation; prefix2 is not a K2 trained baseline"))


def run(config_path, output):
    started = time.perf_counter()
    config_digest = sha256(config_path)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    if config["protocol"] != PROTOCOL or config["split"] != "TRAIN" or config["expected_parents"] != 768:
        raise ValueError("registered TRAIN768 protocol required")
    output = Path(output)
    if output.exists():
        raise ValueError("fresh output required")
    data_path, checkpoint_path = Path(config["data"]), Path(config["checkpoint"])
    if sha256(data_path) != config["data_sha256"] or sha256(checkpoint_path) != config["checkpoint_sha256"]:
        raise ValueError("registered input hash mismatch")
    data = load_train_only(data_path, config["expected_parents"])
    import torch
    from routeset.common import decode_paths
    from routeset.models import SetRegressor
    torch.set_num_threads(1)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    trained = checkpoint["config"]
    for key, expected in dict(objective="saturation", candidates=4, cond_dim=34, horizon=24, seed=0).items():
        if trained.get(key) != expected:
            raise ValueError("registered trained joint4 configuration mismatch: " + key)
    if trained["dataset_sha256"] != config["data_sha256"]:
        raise ValueError("model trained on a different source")
    model = SetRegressor(34, 24, 4, trained["width"], trained["depth"]).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    inference_started = time.perf_counter()
    predictions = []
    with torch.no_grad():
        for first in range(0, len(data["scenes"]), 32):
            scenes = torch.as_tensor(data["scenes"][first:first + 32], dtype=torch.float32)
            predictions.append(decode_paths(model(scenes, 4), scenes).numpy())
    predictions = np.concatenate(predictions)
    inference_s = time.perf_counter() - inference_started
    parents, changes = [], []
    for i, scene in enumerate(data["scenes"]):
        mask = data["path_mask"][i]
        parent, records = analyze_parent(scene, data["paths"][i, mask], data["modes"][i, mask], predictions[i],
                                         str(data["parent_ids"][i]), str(data["scene_ids"][i]), config["geometric_dpp_sigma_m"])
        parents.append(parent)
        changes.extend(records)
    summary = aggregate(parents, config["screening_threshold"])
    if (sha256(data_path) != config["data_sha256"] or sha256(checkpoint_path) != config["checkpoint_sha256"] or
            sha256(config_path) != config_digest):
        raise ValueError("an input changed during the read-only audit")
    output.mkdir(parents=True)
    np.savez_compressed(output / "train_joint4_predictions.npz", paths=predictions, parent_ids=data["parent_ids"], scene_ids=data["scene_ids"])
    write_json(output / "parents.json", parents)
    write_json(output / "closures.json", changes)
    summary.update(protocol=PROTOCOL, training=False, model_updates=0, gpu_hours=0, split="TRAIN", checkpoint_step=checkpoint["step"],
                   actual_forward_candidate_states=int(len(predictions) * 4), inference_cpu_s=inference_s,
                   privileged_reference_bank_states=sum(p["reference_types"] for p in parents),
                   reference_pair_searches_per_selection=sum(p["selections"]["pair_count"] for p in parents),
                   candidate_accounting=config["candidate_accounting"], aggregation="closures within parent, then equal-parent; no-solution closures retained separately",
                   selection_scope="one reference pair per parent chosen before closure is revealed; never a different best pair per closure",
                   dpp_equivalence="unit-quality RBF K2 MAP is exactly farthest pair, not an independent algorithm win",
                   elapsed_cpu_wall_s=time.perf_counter() - started)
    write_json(output / "summary.json", summary)
    write_json(output / "provenance.json", dict(config=config, config_sha256=config_digest,
        data_sha256=sha256(data_path), checkpoint_sha256=sha256(checkpoint_path), source_commit=os.environ.get("CODE_COMMIT", "unrecorded"),
        source_sha256={name: sha256(Path(__file__).resolve().parents[1] / name) for name in (
            "scripts/audit_multigate_closure_opportunity.py", "routeset/multigate.py", "routeset/geometry.py", "routeset/models.py", "routeset/common.py")},
        outputs_sha256={path.name: sha256(path) for path in output.iterdir() if path.is_file()},
        data_access="split/parent/scene IDs are metadata; only TRAIN payload rows decoded, DEV bytes may be skipped by ZIP decompression; locked-containing archive rejected before any payload",
        trained_config=trained, checkpoint_step=checkpoint["step"]))
    print(json.dumps(summary, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.config, args.output)
