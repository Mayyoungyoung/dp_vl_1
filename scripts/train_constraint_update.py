"""Three-arm, equal-exposure constraint-update experiment on DEV_MODEL only.

full_free is the strong free positive-set matching baseline. full_paired and
local_paired isolate only the output gate under identical minimum-change
teacher assignments. All arms see the same training-only edit proxy targets.
"""
import argparse
import copy
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from routeset.common import seed_all, sha256, write_json
from routeset.constraint_update import (ConstraintUpdateRegressor, aggregate_parents, checked_context,
                                       training_targets, update_metrics)
from routeset.multigate import load_dataset, path_validity
from routeset.train_v2 import (atomic_checkpoint, positive_assignment_loss, restore_rng,
                              rng_state, synchronized_time)

ARMS = ("full_free", "full_paired", "local_paired")

CONTINUATION_MATCH_KEYS = ("data", "arm", "batch_size", "width", "depth", "lr", "seed", "eval_every",
    "threads", "device", "proxy_threshold", "aux_weight", "dataset_sha256", "horizon", "objective",
    "auxiliary", "information", "candidate_budget", "selection_split", "sampling", "gpu_uuid")


def check_continuation(config, previous):
    """Permit a new output/source revision and increased total steps only."""
    for key in CONTINUATION_MATCH_KEYS:
        if config.get(key) != previous.get(key):
            raise ValueError("Continuation config mismatch: " + key)
    if config["steps"] <= previous["steps"]:
        raise ValueError("Continuation total steps must increase")


@torch.no_grad()
def evaluate_update(model, data, ids, device, output=None):
    model.eval()
    result = {}
    for k in (1, 2):
        context = checked_context(data["old_paths"][ids], data["old_scenes"][ids], data["scenes"][ids])
        predictions, gates = [], []
        for start in range(0, len(ids), 32):
            selected = ids[start:start + 32]
            predicted, logits = model(
                torch.as_tensor(data["old_scenes"][selected], device=device),
                torch.as_tensor(data["scenes"][selected], device=device),
                torch.as_tensor(context["drafts"][start:start + 32], device=device),
                torch.as_tensor(context["valid"][start:start + 32], device=device), k)
            predictions.append(predicted.cpu().numpy())
            gates.append((logits >= 0).cpu().numpy())
        predictions, gates = np.concatenate(predictions), np.concatenate(gates)
        rows = update_metrics(data["old_paths"][ids], predictions, data["scenes"][ids],
                              data["modes"][ids], data["path_mask"][ids])
        for row, idx, gate in zip(rows, ids, gates):
            row.update(scene_id=str(data["scene_ids"][idx]), parent_id=str(data["parent_ids"][idx]),
                closed_wall=int(data["closed_wall"][idx]), closed_opening=int(data["closed_opening"][idx]),
                newly_invalid_old_count=int((data["old_valid"][idx] & ~data["new_valid"][idx]).sum()),
                predicted_edit_fraction=float(gate.mean()))
        summary = aggregate_parents(rows)
        summary["strata_old_invalid"] = {str(n): aggregate_parents([r for r in rows if r["old_invalid_count"] == n])
                                         for n in range(5) if any(r["old_invalid_count"] == n for r in rows)}
        summary["strata_newly_invalid"] = {str(n): aggregate_parents([r for r in rows if r["newly_invalid_old_count"] == n])
                                           for n in range(5) if any(r["newly_invalid_old_count"] == n for r in rows)}
        # Real second-request latency includes CPU check, transfer, network,
        # hard predicted editing, transfer and all full-route output checks.
        elapsed = []
        idx = ids[0]
        for repeat in range(12):
            start_time = synchronized_time(device)
            c = checked_context(data["old_paths"][idx:idx + 1], data["old_scenes"][idx:idx + 1], data["scenes"][idx:idx + 1])
            predicted, _ = model(torch.as_tensor(data["old_scenes"][idx:idx + 1], device=device),
                torch.as_tensor(data["scenes"][idx:idx + 1], device=device),
                torch.as_tensor(c["drafts"], device=device), torch.as_tensor(c["valid"], device=device), k)
            paths = predicted.cpu().numpy()
            path_validity(np.concatenate([data["old_paths"][idx], paths[0]]), data["scenes"][idx])
            duration = synchronized_time(device) - start_time
            if repeat >= 2:
                elapsed.append(duration * 1000)
        summary.update(new_request_check_head_check_ms_median=float(np.median(elapsed)),
                       new_request_check_head_check_ms_p95=float(np.percentile(elapsed, 95)),
                       timing_scope="second controlled request; old4 generation cost is shared and separately recorded; not Qwen end-to-end",
                       budget_description="old4 plus %d fresh full routes; failures/discards counted" % k)
        result["b%d" % k] = summary
        if output:
            destination = Path(output) / ("b%d" % k)
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / "predictions.npz", paths=predictions, predicted_edit_mask=gates,
                old_paths=data["old_paths"][ids], source_order=context["source_order"],
                scene_ids=data["scene_ids"][ids], parent_ids=data["parent_ids"][ids])
            write_json(destination / "per_scene.json", rows)
            write_json(destination / "metrics.json", summary)
    result["selection_score"] = float(np.mean([result["b%d" % k]["union_unique_valid"] for k in (1, 2)]))
    if output:
        write_json(Path(output) / "metrics.json", result)
    return result


def train_one(args):
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
            raise RuntimeError('physical GPU1 must be explicitly selected via CUDA_VISIBLE_DEVICES=1')
        torch.cuda.set_per_process_memory_fraction(.35)
        torch.cuda.reset_peak_memory_stats()
    seed_all(args.seed)
    data = load_dataset(args.data)
    if set(data["splits"].tolist()) - {"TRAIN", "DEV_MODEL"}:
        raise ValueError("development-only archive required")
    train_ids = np.flatnonzero(data["splits"] == "TRAIN")
    dev_ids = np.flatnonzero(data["splits"] == "DEV_MODEL")
    if not len(train_ids) or not len(dev_ids):
        raise ValueError("nonempty training and development parents required")
    if set(data['parent_ids'][train_ids]) & set(data['parent_ids'][dev_ids]):
        raise ValueError('parent split leakage detected')
    out = Path(args.output)
    continue_from = getattr(args, "continue_from", None)
    if continue_from and (Path(continue_from).resolve().parent == out.resolve()):
        raise ValueError("Continuation output must differ from the preserved source run")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        raise RuntimeError("Existing checkpoint: use a new run_id or --resume")
    if continue_from and not args.resume and (out / "best.pt").exists():
        raise RuntimeError("Continuation requires a fresh output without an existing best checkpoint")
    lock = out / "active.lock"
    try:
        descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Run lock exists; verify its actual PID before recovery")
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        config = vars(args).copy()
        config.update(dataset_sha256=sha256(args.data), horizon=int(data["paths"].shape[-2]),
            code_commit=os.environ.get("CODE_COMMIT", "unrecorded"), gpu_uuid=os.environ.get("RESEARCH_GPU_UUID"),
            objective="saturation all remaining positives" if args.arm == "full_free" else "minimum-source-change paired positives",
            auxiliary="identical train-only point-distance proxy > threshold; not collision/causal-impact truth",
            information="controlled old/new full geometry and endpoints, actual old4, shared whole-route checker",
            candidate_budget="alternate b1/b2; each complete generated/edited path counts; total old4+b",
            selection_split="DEV_MODEL", sampling="uniform original parent, then uniform closure within parent")
        model = ConstraintUpdateRegressor(config["horizon"], args.width, args.depth,
                                          "local" if args.arm == "local_paired" else "full").to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        rng = np.random.default_rng(args.seed)
        sampler = np.random.default_rng(args.seed + 100000)
        parents = np.unique(data["parent_ids"][train_ids])
        groups = [np.flatnonzero((data["parent_ids"][train_ids] == parent)) for parent in parents]
        start_step, best, elapsed_before, exposures, positive_access = 0, -float("inf"), 0., 0, 0
        history, losses = [], []
        cost_origin = dict(prior_elapsed_s=0., prior_trajectory_exposures=0,
                           prior_positive_pool_access=0, start_step=0)
        if args.resume:
            checkpoint = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
            for key in ("dataset_sha256", "arm", "steps", "batch_size", "width", "depth", "lr", "seed", "proxy_threshold", "aux_weight"):
                if config[key] != checkpoint["config"][key]:
                    raise ValueError("Resume config mismatch: " + key)
            if continue_from:
                recorded_source = checkpoint["config"].get("continuation", {}).get("source_checkpoint")
                if recorded_source != str(Path(continue_from).resolve()):
                    raise ValueError("Resume continuation source differs from the original initialization")
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            restore_rng(checkpoint["rng"], rng)
            sampler.bit_generator.state = checkpoint["sampler_state"]
            start_step, best, elapsed_before = checkpoint["step"], checkpoint["best"], checkpoint["elapsed_s"]
            exposures, positive_access = checkpoint["trajectory_exposures"], checkpoint["positive_pool_access"]
            history, losses = checkpoint["history"], checkpoint["loss_tail"]
            cost_origin = checkpoint.get("cost_origin", cost_origin)
            # The original continuation provenance persists through resumptions.
            if "continuation" in checkpoint["config"]:
                config["continuation"] = checkpoint["config"]["continuation"]
                config["continue_from"] = checkpoint["config"].get("continue_from")
        elif continue_from:
            source = Path(continue_from).resolve()
            if source.name != "last.pt":
                raise ValueError("Continue from the source last.pt, not a selected-best model")
            checkpoint = torch.load(source, map_location=args.device, weights_only=False)
            check_continuation(config, checkpoint["config"])
            source_best = source.parent / "best.pt"
            best_checkpoint = torch.load(source_best, map_location="cpu", weights_only=False)
            check_continuation(config, best_checkpoint["config"])
            if best_checkpoint["step"] > checkpoint["step"]:
                raise ValueError("source best occurs after its last checkpoint")
            # Restore all optimization and sampling state. No warm-start-only
            # interpretation: this is the same training run extended in a new
            # output tree, preserving the original completed experiment.
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            restore_rng(checkpoint["rng"], rng)
            sampler.bit_generator.state = checkpoint["sampler_state"]
            start_step, best = checkpoint["step"], checkpoint["best"]
            exposures, positive_access = checkpoint["trajectory_exposures"], checkpoint["positive_pool_access"]
            history, losses = checkpoint["history"], checkpoint["loss_tail"]
            # last.pt predates the final selected-checkpoint evaluation. If a
            # verified completed summary exists, count that real prior cost too.
            prior_elapsed = checkpoint.get("cumulative_elapsed_s", checkpoint["elapsed_s"])
            summary_path = source.parent / "summary.json"
            prior_summary_hash = None
            if summary_path.exists():
                source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
                if source_summary["trajectory_exposures"] != exposures or source_summary["best_checkpoint_sha256"] != sha256(source_best):
                    raise ValueError("source summary/checkpoint provenance mismatch")
                prior_elapsed = max(prior_elapsed, source_summary.get("cumulative_elapsed_s", source_summary["elapsed_s"]))
                prior_summary_hash = sha256(summary_path)
            cost_origin = dict(prior_elapsed_s=prior_elapsed, prior_trajectory_exposures=exposures,
                               prior_positive_pool_access=positive_access, start_step=start_step)
            config["continuation"] = dict(source_checkpoint=str(source), source_checkpoint_sha256=sha256(source),
                source_best_checkpoint=str(source_best), source_best_checkpoint_sha256=sha256(source_best),
                source_summary_sha256=prior_summary_hash, source_code_commit=checkpoint["config"].get("code_commit"),
                source_total_steps=checkpoint["config"]["steps"], source_checkpoint_step=start_step,
                source_best_step=best_checkpoint["step"], cost_origin=cost_origin)
            shutil.copy2(source_best, out / "best.pt")
        write_json(out / "config.json", config)
        write_json(out / "status.json", dict(status="running", pid=os.getpid(), step=start_step,
                                               resume_command="repeat the recorded command with --resume"))
        start_time = synchronized_time(args.device)
        # Cached only training labels/context; counted in actual elapsed cost.
        # Validation never calls training_targets or receives its point masks.
        cache = {k: training_targets(data, train_ids, k, args.proxy_threshold) for k in (1, 2)}
        for step in range(start_step + 1, args.steps + 1):
            k = 1 if step % 2 else 2
            parent_draws = sampler.integers(len(groups), size=args.batch_size)
            positions = np.asarray([sampler.choice(groups[p]) for p in parent_draws])
            idx = train_ids[positions]
            context = cache[k]["context"]
            model.train()
            optimizer.zero_grad(set_to_none=True)
            paths, logits = model(torch.as_tensor(data["old_scenes"][idx], device=args.device),
                torch.as_tensor(data["scenes"][idx], device=args.device),
                torch.as_tensor(context["drafts"][positions], device=args.device),
                torch.as_tensor(context["valid"][positions], device=args.device), k)
            if args.arm == "full_free":
                route_loss = positive_assignment_loss(paths[:, :, 1:-1],
                    torch.as_tensor(data["paths"][idx, :, 1:-1], device=args.device),
                    cache[k]["remaining_mask"][positions], "saturation", rng)
            else:
                route_loss = F.mse_loss(paths[:, :, 1:-1],
                    torch.as_tensor(cache[k]["targets"][positions, :, 1:-1], device=args.device))
            auxiliary = F.binary_cross_entropy_with_logits(logits,
                torch.as_tensor(cache[k]["proxy"][positions], device=args.device, dtype=logits.dtype))
            loss = route_loss + args.aux_weight * auxiliary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            exposures += args.batch_size * k
            positive_access += int(cache[k]["remaining_positive_count"][positions].sum())
            losses.append([float(loss.detach()), float(route_loss.detach()), float(auxiliary.detach())])
            losses = losses[-100:]
            if step % 100 == 0:
                print(json.dumps(dict(arm=args.arm, step=step, loss=np.mean(losses, axis=0).tolist(),
                    elapsed_s=elapsed_before + synchronized_time(args.device) - start_time)), flush=True)
            if step % args.eval_every == 0 or step == args.steps or step == args.stop_after:
                metrics = evaluate_update(model, data, dev_ids, args.device)
                improved = metrics["selection_score"] > best
                best = max(best, metrics["selection_score"])
                history.append(dict(step=step, loss=np.mean(losses, axis=0).tolist(), dev_model=metrics))
                elapsed = elapsed_before + synchronized_time(args.device) - start_time
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scaler=None, step=step, config=config, rng=rng_state(rng), sampler_state=sampler.bit_generator.state,
                    best=best, history=history, loss_tail=losses, elapsed_s=elapsed, trajectory_exposures=exposures,
                    positive_pool_access=positive_access, cost_origin=cost_origin,
                    cumulative_elapsed_s=cost_origin["prior_elapsed_s"] + elapsed,
                    incremental_trajectory_exposures=exposures - cost_origin["prior_trajectory_exposures"])
                atomic_checkpoint(out / "last.pt", checkpoint)
                if improved:
                    atomic_checkpoint(out / "best.pt", checkpoint)
                write_json(out / "history.json", history)
                print(json.dumps(dict(arm=args.arm, step=step, selection_score=metrics["selection_score"],
                    b1_unique=metrics["b1"]["union_unique_valid"], b2_unique=metrics["b2"]["union_unique_valid"])), flush=True)
                if step == args.stop_after and step < args.steps:
                    write_json(out / "status.json", dict(status="interrupted_for_resume_check", step=step, exit_code=0))
                    return
        best_checkpoint = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(best_checkpoint["model"])
        metrics = evaluate_update(model, data, dev_ids, args.device, out / "dev_model")
        elapsed = elapsed_before + synchronized_time(args.device) - start_time
        summary = dict(metrics=metrics, elapsed_s=elapsed, trajectory_exposures=exposures,
            positive_pool_access=positive_access, parameters=model.active_parameter_count(), best_step=best_checkpoint["step"],
            gpu_hours_reserved=elapsed / 3600 if args.device.startswith("cuda") else 0.,
            peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.,
            best_checkpoint_sha256=sha256(out / "best.pt"),
            prediction_sha256={"b%d" % k: sha256(out / "dev_model" / ("b%d" % k) / "predictions.npz") for k in (1, 2)})
        summary.update(cost_origin=cost_origin, incremental_elapsed_s=elapsed,
            cumulative_elapsed_s=cost_origin["prior_elapsed_s"] + elapsed,
            incremental_trajectory_exposures=exposures - cost_origin["prior_trajectory_exposures"],
            incremental_positive_pool_access=positive_access - cost_origin["prior_positive_pool_access"],
            cumulative_gpu_hours_reserved=(cost_origin["prior_elapsed_s"] + elapsed) / 3600 if args.device.startswith("cuda") else 0.,
            selected_checkpoint_from_prior_run=best_checkpoint["step"] <= cost_origin["start_step"],
            cost_scope="elapsed_s/gpu_hours_reserved are additional cost in this output tree; cumulative_* include prior source cost exactly once")
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(status="completed", step=args.steps, exit_code=0))
        print(json.dumps(dict(arm=args.arm, summary=summary)), flush=True)
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", exit_code=1, exception=repr(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=ARMS + ("all",), default="all")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--aux-weight", type=float, default=.01)
    parser.add_argument("--proxy-threshold", type=float, default=.02)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-from", help="Source last.pt; preserve old output, restore full state, increase --steps")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.eval_every) < 1 or not 1 <= args.threads <= 4:
        parser.error("positive step/batch/eval counts and 1--4 threads required")
    if args.proxy_threshold < 0 or args.aux_weight < 0:
        parser.error("nonnegative proxy threshold and auxiliary weight required")
    if args.continue_from and args.arm == "all":
        parser.error("--continue-from requires one explicit arm; --resume restores the new output without reinitializing")
    if args.arm == "all":
        for arm in ARMS:
            paired = copy.copy(args)
            paired.arm, paired.output = arm, str(Path(args.output) / arm)
            train_one(paired)
    else:
        train_one(args)


if __name__ == "__main__":
    main()
