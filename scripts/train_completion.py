"""Equal-data, equal-exposure completion experiment on TRAIN / DEV_MODEL only.

Run with ``python -m scripts.train_completion --data ... --output ...``.
Default --mechanism both executes attention then coverage sequentially.
"""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch

from routeset.common import decode_paths, encode_paths, seed_all, sha256, write_json
from routeset.completion import (CONTEXT_COUNTS, CONTEXT_KINDS, CompletionRegressor,
                                 build_contexts, draft_validity)
from routeset.multigate import load_dataset, path_validity, route_metrics, route_modes
from routeset.train_v2 import (atomic_checkpoint, positive_assignment_loss, restore_rng,
                              rng_state, synchronized_time)


@torch.no_grad()
def evaluate_completion(model, data, ids, device, output=None):
    """Given-draft budgets remain visible, including failed and repeated drafts."""
    model.eval()
    result = {}
    for kind_index, kind in enumerate(CONTEXT_KINDS):
        contexts = build_contexts(data, ids, np.random.default_rng(701000 + kind_index), kind)
        k = 4 if kind == "empty" else 2
        predictions = []
        for start in range(0, len(ids), 32):
            selected = ids[start:start + 32]
            scene = torch.as_tensor(data["scenes"][selected], device=device)
            drafts = torch.as_tensor(contexts["drafts"][start:start + 32], device=device)
            valid = torch.as_tensor(contexts["valid"][start:start + 32], device=device)
            predictions.append(decode_paths(model(scene, drafts, valid, k), scene).cpu().numpy())
        predictions = np.concatenate(predictions)
        rows = []
        for row, idx in enumerate(ids):
            scene = data["scenes"][idx]
            supplied = contexts["drafts"][row, contexts["present"][row]]
            union = np.concatenate([supplied, predictions[row]], axis=0)
            new_metrics = route_metrics(predictions[row:row + 1], scene[None], data["modes"][idx:idx + 1], data["path_mask"][idx:idx + 1])
            union_metrics = route_metrics(union[None], scene[None], data["modes"][idx:idx + 1], data["path_mask"][idx:idx + 1])
            provided_modes = route_modes(contexts["drafts"][row], scene)
            provided = set(provided_modes[contexts["valid"][row] & (provided_modes >= 0)].tolist())
            generated = set(new_metrics["per_scene"]["modes"][0].tolist()) - {-1}
            record = dict(scene_id=str(data["scene_ids"][idx]), parent_id=str(data["parent_ids"][idx]),
                          context_kind=kind, supplied_candidates=len(supplied), new_candidates=k,
                          total_candidates=len(union), provided_unique_valid=len(provided),
                          additional_unique_valid=len(generated - provided),
                          all_known_types_already_covered=bool(contexts["all_known_covered"][row]),
                          reference_types=int(data["path_mask"][idx].sum()))
            for prefix, metrics in (("new_", new_metrics), ("union_", union_metrics)):
                for key in ("valid_rate", "success", "unique_valid", "reference_coverage", "collision_rate"):
                    record[prefix + key] = metrics[key]
            rows.append(record)
        scalar_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
        summary = {key: float(np.mean([row[key] for row in rows])) for key in scalar_keys}
        summary["all_known_types_already_covered_rate"] = float(contexts["all_known_covered"].mean())
        # Single-request timing includes exact input gate, transfer, model,
        # decoding, transfer back, and deterministic full-path output checking.
        scene_np = data["scenes"][ids[:1]]
        draft_np, present_np = contexts["drafts"][:1], contexts["present"][:1]
        timings, head_timings = [], []
        for repeat in range(12):
            start_time = synchronized_time(device)
            valid_np = draft_validity(draft_np, scene_np, present_np)
            scene = torch.as_tensor(scene_np, device=device)
            drafts = torch.as_tensor(draft_np, device=device)
            valid = torch.as_tensor(valid_np, device=device)
            predicted = decode_paths(model(scene, drafts, valid, k), scene).cpu().numpy()
            head_elapsed = synchronized_time(device) - start_time
            supplied = draft_np[0, present_np[0]]
            path_validity(np.concatenate([supplied, predicted[0]], axis=0), scene_np[0])
            elapsed = synchronized_time(device) - start_time
            if repeat >= 2:
                timings.append(elapsed * 1000)
                head_timings.append(head_elapsed * 1000)
        summary.update(batch1_gate_head_check_ms_median=float(np.median(timings)),
                       batch1_gate_head_check_ms_p95=float(np.percentile(timings, 95)),
                       batch1_gate_head_ms_median=float(np.median(head_timings)), forward_passes=1,
                       budget_description="given %d drafts + %d new candidates" % (CONTEXT_COUNTS[kind], k))
        result[kind] = summary
        if output is not None:
            destination = Path(output) / kind
            destination.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(destination / "predictions.npz", paths=predictions, drafts=contexts["drafts"],
                                draft_present=contexts["present"], draft_valid=contexts["valid"],
                                scene_ids=data["scene_ids"][ids], parent_ids=data["parent_ids"][ids])
            write_json(destination / "per_scene.json", rows)
            write_json(destination / "metrics.json", summary)
    result["selection_score"] = float(np.mean([result[kind]["additional_unique_valid"] for kind in CONTEXT_KINDS if kind != "empty"]))
    if output is not None:
        write_json(Path(output) / "metrics.json", result)
    return result


def train_one(args):
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.35)
        torch.cuda.reset_peak_memory_stats()
    seed_all(args.seed)
    # Locked examples are retained in their on-disk archive but never sampled,
    # scored or used for model selection by this script.
    data = load_dataset(args.data)
    train_ids = np.flatnonzero(data["splits"] == "TRAIN")
    dev_ids = np.flatnonzero(data["splits"] == "DEV_MODEL")
    if not len(train_ids) or not len(dev_ids):
        raise ValueError("nonempty TRAIN and DEV_MODEL required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        raise RuntimeError("Existing run checkpoint; use --resume or a new output directory")
    lock = out / "active.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("Run lock exists: inspect its PID before recovering a stale lock")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        config = vars(args).copy()
        config.update(dataset_sha256=sha256(args.data), horizon=int(data["paths"].shape[-2]),
                      cond_dim=int(data["scenes"].shape[-1]), code_commit=os.environ.get("CODE_COMMIT", "unrecorded"),
                      gpu_uuid=os.environ.get("RESEARCH_GPU_UUID"), selection_split="DEV_MODEL",
                      information="controlled exact geometry, endpoints, provided route coordinates and exact validity gate",
                      objective="saturation",
                      target_policy="known positives with covered types removed; full positive fallback if all known types covered; minimum-cost coverage of all remaining positives when R<K with nearest-positive excess slots",
                      budget="k=2 new routes for completion; k=4 from empty every fourth training update",
                      pairing="identical scene/context RNG streams, initialization shapes, full positive pools and exposure schedule")
        model = CompletionRegressor(config["cond_dim"], config["horizon"], args.width, args.depth, args.mechanism).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        rng = np.random.default_rng(args.seed)
        sampler = np.random.default_rng(args.seed + 100000)
        context_rng = np.random.default_rng(args.seed + 200000)
        draft_rng = np.random.default_rng(args.seed + 300000)
        self_draft_batches = 0
        step_start, best, elapsed_before, exposures = 0, -float("inf"), 0., 0
        history = []
        if args.resume and (out / "last.pt").exists():
            checkpoint = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
            for key in ("dataset_sha256", "mechanism", "objective", "width", "depth", "seed", "lr", "batch_size", "steps"):
                if config[key] != checkpoint["config"][key]:
                    raise ValueError("Resume config mismatch: " + key)
            for key, default in [('self_draft_prob', 0.), ('self_draft_start', 1000)]:
                if config[key] != checkpoint['config'].get(key, default):
                    raise ValueError('Resume config mismatch: '+key)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            restore_rng(checkpoint["rng"], rng)
            sampler.bit_generator.state = checkpoint["sampler_state"]
            context_rng.bit_generator.state = checkpoint["context_rng_state"]
            if 'draft_rng_state' in checkpoint:
                draft_rng.bit_generator.state = checkpoint['draft_rng_state']
            self_draft_batches = checkpoint.get('self_draft_batches', 0)
            step_start, best = checkpoint["step"], checkpoint["best"]
            elapsed_before, exposures, history = checkpoint["elapsed_s"], checkpoint["trajectory_exposures"], checkpoint["history"]
        write_json(out / "config.json", config)
        write_json(out / "status.json", dict(status="running", pid=os.getpid(), step=step_start,
                                               run_id=out.name, resume_command="rerun the same command with --resume"))
        started = synchronized_time(args.device)
        losses = []
        for step in range(step_start + 1, args.steps + 1):
            model.train()
            ids = sampler.choice(train_ids, args.batch_size, replace=True)
            k = 4 if step % 4 == 0 else 2
            contexts = build_contexts(data, ids, context_rng, "empty" if k == 4 else None)
            scenes = torch.as_tensor(data["scenes"][ids], device=args.device)
            # Late model-generated contexts address measured reference/model
            # draft shift. No reference coordinates enter these model inputs.
            if k == 2 and args.self_draft_prob > 0 and step >= args.self_draft_start and draft_rng.random() < args.self_draft_prob:
                with torch.no_grad():
                    empty = torch.zeros((len(ids), 2, config['horizon'], 3), device=args.device)
                    absent = torch.zeros((len(ids), 2), dtype=torch.bool, device=args.device)
                    draft_np = decode_paths(model(scenes, empty, absent, k=2), scenes).cpu().numpy()
                contexts['drafts'] = draft_np
                contexts['present'] = np.ones((len(ids), 2), dtype=bool)
                contexts['valid'] = draft_validity(draft_np, data['scenes'][ids], contexts['present'])
                remaining = data['path_mask'][ids].copy()
                for row, idx in enumerate(ids):
                    labels = route_modes(draft_np[row], data['scenes'][idx])
                    covered = set(labels[contexts['valid'][row] & (labels >= 0)].tolist())
                    for label in covered:
                        remaining[row] &= data['modes'][idx] != label
                    if not remaining[row].any():
                        remaining[row] = data['path_mask'][idx]
                contexts['remaining_mask'] = remaining
                self_draft_batches += 1
            drafts = torch.as_tensor(contexts["drafts"], device=args.device)
            valid = torch.as_tensor(contexts["valid"], device=args.device)
            targets = encode_paths(torch.as_tensor(data["paths"][ids], device=args.device), scenes)
            optimizer.zero_grad(set_to_none=True)
            loss = positive_assignment_loss(model(scenes, drafts, valid, k), targets, contexts["remaining_mask"], "saturation", rng)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            exposures += args.batch_size * k
            if step % 100 == 0:
                print(json.dumps(dict(mechanism=args.mechanism, step=step, loss=float(np.mean(losses[-100:])),
                                      elapsed_s=elapsed_before + synchronized_time(args.device) - started)), flush=True)
            if step % args.eval_every == 0 or step == args.steps or step == args.stop_after:
                metrics = evaluate_completion(model, data, dev_ids, args.device)
                score = metrics["selection_score"]
                improved = score > best
                best = max(best, score)
                history.append(dict(step=step, loss=float(np.mean(losses[-100:])), dev_model=metrics))
                elapsed = elapsed_before + synchronized_time(args.device) - started
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                  scaler=None, step=step, config=config, rng=rng_state(rng), sampler_state=sampler.bit_generator.state,
                                  context_rng_state=context_rng.bit_generator.state, best=best, history=history,
                                  draft_rng_state=draft_rng.bit_generator.state, self_draft_batches=self_draft_batches,
                                  elapsed_s=elapsed, trajectory_exposures=exposures)
                atomic_checkpoint(out / "last.pt", checkpoint)
                if improved:
                    atomic_checkpoint(out / "best.pt", checkpoint)
                write_json(out / "history.json", history)
                print(json.dumps(dict(mechanism=args.mechanism, **history[-1])), flush=True)
                if step == args.stop_after and step < args.steps:
                    write_json(out / "status.json", dict(status="interrupted_for_resume_check", step=step, exit_code=0))
                    return
        checkpoint = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate_completion(model, data, dev_ids, args.device, out / "dev_model")
        elapsed = elapsed_before + synchronized_time(args.device) - started
        summary = dict(metrics=metrics, elapsed_s=elapsed,
                       gpu_hours_reserved=elapsed / 3600 if args.device.startswith("cuda") else 0.,
                       trajectory_exposures=exposures, parameters=model.active_parameter_count(),
                       self_draft_batches=self_draft_batches, extra_training_draft_forwards=self_draft_batches,
                       peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.,
                       best_step=checkpoint["step"], best_checkpoint_sha256=sha256(out / "best.pt"),
                       prediction_sha256={kind: sha256(out / "dev_model" / kind / "predictions.npz") for kind in CONTEXT_KINDS})
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(status="completed", step=args.steps, exit_code=0))
        print(json.dumps(dict(mechanism=args.mechanism, **summary)), flush=True)
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", exit_code=1, exception=repr(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mechanism", choices=("attention", "coverage", "both"), default="both")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--self-draft-prob", type=float, default=0.)
    parser.add_argument("--self-draft-start", type=int, default=1000)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.batch_size < 1 or not 1 <= args.threads <= 4:
        parser.error("positive steps/eval/batch and 1--4 CPU threads required")
    if not 0 <= args.self_draft_prob <= 1 or args.self_draft_start < 1:
        parser.error('self draft probability must be in [0,1], start >=1')
    if args.mechanism == "both":
        for mechanism in ("attention", "coverage"):
            paired = copy.copy(args)
            paired.mechanism, paired.output = mechanism, str(Path(args.output) / mechanism)
            train_one(paired)
    else:
        train_one(args)


if __name__ == "__main__":
    main()
