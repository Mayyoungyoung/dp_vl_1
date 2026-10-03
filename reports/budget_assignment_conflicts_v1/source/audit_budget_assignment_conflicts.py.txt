"""Saved TRAIN768 assignment audit: no Torch, training, generation, or SSH.

The subproblems are prefixes of ONE historical K4 cost matrix. They are not
predictions of K1/K2-trained models. Enumerating label assignments generates no
routes. Finite gate geometry is used only to audit already saved route types.
"""
import argparse
from collections import Counter
from functools import lru_cache
import itertools
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from routeset.common import sha256, write_json
from routeset.multigate import path_validity, route_modes
from scripts.audit_multigate_closure_opportunity import load_train_only


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "budget_assignment_conflicts_v1"
DATA = "data/multigate_v1_partitions/development.npz"
POOL = "runs/multigate_closure_opportunity_train768_v1/audit"
EXPECTED = {
    DATA: "f93c8c4b54e44c365d323e4efe0bdf9f6c895fd611a830338971a242d233e057",
    POOL + "/train_joint4_predictions.npz": "53ccf3fbee6d8843a1ef448c87099d6f4d8d43d5c3bc8bb7e0e548a50a9c4bef",
    POOL + "/parents.json": "928c7972c5651c6d31db857e1069da4db8a305aaa58cf9399d1cc61e5df6a839",
}
STOPS = (1, 2, 4)
ATOL, RTOL = 1e-10, 1e-8
NO_OPPORTUNITY_FRACTION = .95
SOURCE_FILES = (
    "scripts/audit_budget_assignment_conflicts.py", "tests/test_budget_assignment_conflicts.py",
    "scripts/audit_multigate_closure_opportunity.py", "routeset/common.py",
    "routeset/multigate.py", "routeset/geometry.py", "routeset/train_v2.py",
)


@lru_cache(maxsize=64)
def assignments(references, candidates):
    """Exact admissible positive/saturation assignments, stable lexical order.

    R>=K: injective. R<K: surjective, without random reference duplication.
    K never exceeds 4 here. Limits prevent accidental combinatorial expansion.
    """
    if not 1 <= references <= 16 or not 1 <= candidates <= 4:
        raise ValueError("audit admits only R1..16 and K1..4")
    if references >= candidates:
        values = itertools.permutations(range(references), candidates)
    else:
        values = (a for a in itertools.product(range(references), repeat=candidates)
                  if len(set(a)) == references)
    result = np.asarray(list(values), dtype=np.int16).reshape(-1, candidates)
    result.flags.writeable = False
    return result


def interior_cost(predicted, reference):
    """Float64 MSE on 22 interior coordinates, from saved float32 H24 paths.

    The shared straight-line residual cancels algebraically in training loss.
    This is not a claim of bit-exact recovery of Torch float32 residual costs:
    saved decode/re-encoding and arithmetic reductions can differ in rounding.
    """
    p, r = np.asarray(predicted, dtype=np.float64), np.asarray(reference, dtype=np.float64)
    if p.shape != (4, 24, 3) or r.ndim != 3 or r.shape[1:] != (24, 3):
        raise ValueError("registered four H24 predictions and H24 references required")
    if not len(r) or len(r) > 16 or not np.isfinite(p).all() or not np.isfinite(r).all():
        raise ValueError("finite saved paths and nonempty bounded reference bank required")
    return np.square(p[:, None, 1:-1] - r[None, :, 1:-1]).mean(axis=(-1, -2))


def _keys(labels, selected):
    return {tuple(row) for row in labels[:, selected].tolist()}


def compatible(optimal, labels, visible):
    """Existence across ALL tied optima, not one Hungarian tie-break."""
    sets = {}
    for k in STOPS[:-1]:
        select = np.flatnonzero(visible[:k])
        sets[k] = _keys(labels[optimal[k]], select)
    for full in labels[optimal[4]]:
        if all(tuple(full[np.flatnonzero(visible[:k])].tolist()) in sets[k] for k in STOPS[:-1]):
            return True
    return False


def analyze_cost(cost, reference_modes, valid, predicted_modes):
    """Oracle label-assignment compatibility, never physical path editing."""
    c = np.asarray(cost, dtype=np.float64)
    modes, valid, actual = np.asarray(reference_modes), np.asarray(valid, dtype=bool), np.asarray(predicted_modes)
    if c.ndim != 2 or c.shape[0] != 4 or not 1 <= c.shape[1] <= 16 or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("finite nonnegative 4xR cost matrix required")
    if modes.shape != (c.shape[1],) or valid.shape != (4,) or actual.shape != (4,):
        raise ValueError("label/validity dimensions disagree")
    r = c.shape[1]
    opt, opt_cost, counts, physical_match = {}, {}, {}, {}
    visible = valid & (actual >= 0)
    for k in STOPS:
        a = assignments(r, k)
        value = c[np.arange(k), a].mean(axis=1)
        best = float(value.min())
        opt[k] = a[value <= best + ATOL + RTOL * abs(best)]
        opt_cost[k], counts[k] = best, len(a)
        positions = np.flatnonzero(visible[:k])
        physical_match[k] = bool(np.any(np.all(modes[opt[k]][:, positions] == actual[positions], axis=1)))
    index_ok = compatible(opt, np.arange(r), np.ones(4, dtype=bool))
    type_ok = compatible(opt, modes, np.ones(4, dtype=bool))
    valid_type_ok = compatible(opt, modes, visible)
    # A single complete assignment must also be admissible at earlier stops.
    full = assignments(r, 4)
    admissible = np.ones(len(full), dtype=bool)
    for k in STOPS[:-1]:
        admissible &= np.asarray([len(set(row)) == min(k, r) for row in full[:, :k]])
    chains = full[admissible]
    values = np.stack([c[np.arange(k), chains[:, :k]].mean(axis=1) for k in STOPS], axis=1)
    regret = np.maximum(values - np.asarray([opt_cost[k] for k in STOPS]), 0)
    worst = regret.max(axis=1)
    witness = int(np.argmin(worst))
    # Report the stable free optimizer, but never use it to decide tie conflict.
    chosen = {str(k): dict(reference_indices=opt[k][0].tolist(), assigned_modes=modes[opt[k][0]].tolist(),
                          tied_optima=len(opt[k]), optimal_mean_interior_mse=opt_cost[k],
                          exists_optimum_consistent_with_actual_valid_types=physical_match[k]) for k in STOPS}
    return dict(reference_count=r, enumerated_assignments=counts, free=chosen,
                index_conflict=not index_ok, assigned_type_conflict=not type_ok,
                valid_slot_assigned_type_conflict=not valid_type_ok,
                physical_consistency_at_every_stop=all(physical_match.values()),
                valid_candidates=int(valid.sum()), valid_known_candidates=int(visible.sum()),
                invalid_candidates=int((~valid).sum()), unknown_candidates=int((actual < 0).sum()),
                actual_predicted_modes=actual.tolist(), actual_valid=valid.tolist(),
                actual_unique_valid_types=len(set(actual[visible].tolist())),
                joint_chain_minimax_mean_mse_regret=float(worst[witness]),
                joint_chain_witness_indices=chains[witness].tolist(),
                joint_chain_witness_modes=modes[chains[witness]].tolist(),
                joint_chain_witness_prefix_regrets=regret[witness].tolist(),
                joint_chain_admissible_assignments=len(chains))


def validate_identities(data, pool, saved):
    expected_ids = ["multigate_v1_TRAIN_%05d" % i for i in range(768)]
    if data["parent_ids"].tolist() != expected_ids or data["scene_ids"].tolist() != [p + "_c00" for p in expected_ids]:
        raise ValueError("exact registered TRAIN768 identity/order required")
    if not np.all(data["splits"] == "TRAIN"):
        raise ValueError("TRAIN only")
    for key in ("parent_ids", "scene_ids"):
        if not np.array_equal(pool[key], data[key]):
            raise ValueError("saved pool identity mismatch")
    if pool["paths"].shape != (768, 4, 24, 3) or len(saved) != 768:
        raise ValueError("saved pool dimensions mismatch")
    if [p["parent_id"] for p in saved] != expected_ids:
        raise ValueError("historical checker rows mismatch")


def summarize(rows):
    n = len(rows)
    counts = {key: sum(bool(r[key]) for r in rows) for key in (
        "index_conflict", "assigned_type_conflict", "valid_slot_assigned_type_conflict", "physical_consistency_at_every_stop")}
    counts.update(parents=n, no_valid_slot_type_conflict=n-counts["valid_slot_assigned_type_conflict"],
                  invalid_candidates=sum(r["invalid_candidates"] for r in rows),
                  unknown_candidates=sum(r["unknown_candidates"] for r in rows))
    absent = counts["no_valid_slot_type_conflict"] / n
    values = np.asarray([r["joint_chain_minimax_mean_mse_regret"] for r in rows])
    counts.update(no_conflict_fraction=absent,
                  gate_decision="stop_no_nontrivial_type_conflict" if absent >= NO_OPPORTUNITY_FRACTION else "oracle_conflict_only_requires_physical_interpretation",
                  by_reference_count={str(r): dict(parents=len(g), index_conflicts=sum(x["index_conflict"] for x in g),
                      valid_slot_type_conflicts=sum(x["valid_slot_assigned_type_conflict"] for x in g))
                      for r in sorted(set(x["reference_count"] for x in rows)) for g in [[x for x in rows if x["reference_count"] == r]]},
                  minimax_regret_mse=dict(mean=float(values.mean()), maximum=float(values.max()), median=float(np.median(values))))
    return counts


def run(output):
    start, cpu_start = time.perf_counter(), time.process_time()
    output = Path(output)
    if output.exists():
        raise ValueError("fresh output required; keep earlier failed runs")
    output.mkdir(parents=True)
    status = dict(protocol=PROTOCOL, pid=os.getpid(), state="running", command=sys.argv,
                  model_forwards=0, optimizer_steps=0, newly_generated_routes=0, ssh_calls=0)
    write_json(output / "status.json", status)
    try:
        actual_hashes = {p: sha256(ROOT / p) for p in EXPECTED}
        if actual_hashes != EXPECTED:
            raise ValueError("fixed saved TRAIN input SHA mismatch")
        provenance_path = ROOT / POOL / "provenance.json"
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance["checkpoint_sha256"] != "b2cb51ae0de50785a12f129c2ff12ecc981a9df826b5ee7ae1462de6366f102c" or provenance["checkpoint_step"] != 1500:
            raise ValueError("historical K4 source checkpoint identity mismatch")
        data = load_train_only(ROOT / DATA, 768)
        with np.load(ROOT / POOL / "train_joint4_predictions.npz", allow_pickle=False) as f:
            pool = {k: f[k] for k in f.files}
        saved = json.loads((ROOT / POOL / "parents.json").read_text(encoding="utf-8"))
        validate_identities(data, pool, saved)
        rows, reference_states = [], 0
        for i, scene in enumerate(data["scenes"]):
            mask = data["path_mask"][i]
            refs, modes, pred = data["paths"][i, mask], data["modes"][i, mask], pool["paths"][i]
            if len(set(modes.tolist())) != len(refs) or modes.tolist() != saved[i]["reference_modes"]:
                raise ValueError("one positive representative per actual physical type required")
            if not path_validity(refs, scene)["valid"].all() or not np.array_equal(route_modes(refs, scene), modes):
                raise ValueError("saved reference geometry/type mismatch")
            validity, labels = path_validity(pred, scene)["valid"], route_modes(pred, scene)
            historical = saved[i]["static"]["joint4_actual"]
            if validity.tolist() != historical["valid"] or labels.tolist() != historical["modes"]:
                raise ValueError("unchanged geometry checker disagrees with original saved decisions")
            cost = interior_cost(pred, refs)
            row = analyze_cost(cost, modes, validity, labels)
            row.update(parent_id=str(data["parent_ids"][i]), scene_id=str(data["scene_ids"][i]),
                       reference_modes=modes.tolist(), mean_interior_mse_cost_matrix=cost.tolist())
            rows.append(row)
            reference_states += len(refs)
        if {p: sha256(ROOT / p) for p in EXPECTED} != actual_hashes:
            raise ValueError("source data changed during read-only audit")
        result = summarize(rows)
        result.update(protocol=PROTOCOL, split="TRAIN", fixed_parents=768, saved_K4_candidates=3072,
                      positive_reference_states=reference_states, new_model_forwards=0, new_training_steps=0,
                      new_generated_routes=0, gpu_seconds=0,
                      label_assignment_rows_enumerated=sum(sum(r["enumerated_assignments"].values()) for r in rows),
                      existing_geometry_checks=dict(saved_candidate_paths=3072, positive_reference_paths=reference_states),
                      policy=dict(stops=list(STOPS), tie_atol=ATOL, tie_rtol=RTOL, no_opportunity_fraction=NO_OPPORTUNITY_FRACTION),
                      interpretation="Oracle positive-label assignment only. Saved K4 prefixes are not K1/K2 models; no coordinate changes, model gains or prospective K8 claims.")
        write_json(output / "parents.json", rows)
        write_json(output / "summary.json", result)
        write_json(output / "provenance.json", dict(input_sha256=actual_hashes,
            historical_pool_provenance_sha256=sha256(provenance_path),
            historical_checkpoint_sha256=provenance["checkpoint_sha256"], historical_checkpoint_step=1500,
            source_sha256={p: sha256(ROOT / p) for p in SOURCE_FILES},
            cost="float64 interior H24[1:-1] coordinate MSE; algebraically shared-residual equivalent, not bit-exact original Torch float32",
            access="Metadata checked first; original row-addressed loader decodes TRAIN payload only; archive with reserved roles is rejected.",
            python=sys.version, numpy=np.__version__))
        status.update(state="completed", exit_code=0)
    except Exception as exc:
        status.update(state="failed", exit_code=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status.update(elapsed_wall_s=time.perf_counter()-start, process_cpu_s=time.process_time()-cpu_start)
        write_json(output / "status.json", status)
    print(json.dumps(result, allow_nan=False))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(parser.parse_args().output)
