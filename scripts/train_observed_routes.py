"""Train a frozen-real-Qwen route-set pilot without privileged target inputs."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from routeset.common import seed_all, sha256, write_json
from routeset.observed_route_head import (ObservedRouteHead, load_observed_dataset,
                                         semantic_endpoint_accuracy)
from routeset.train_v2 import (atomic_checkpoint, positive_assignment_loss, restore_rng,
                              rng_state, synchronized_time)


def event_sequence(values):
    states = np.asarray(values) > .5
    return states[np.r_[True, states[1:] != states[:-1]]].tolist()


def observation_metrics(paths, opened, data, ids):
    rows = []
    for row, idx in enumerate(ids):
        references = data["paths"][idx, data["path_mask"][idx]]
        events = data["events"][idx, data["path_mask"][idx]]
        distances = np.linalg.norm(paths[row, :, None] - references[None], axis=-1).mean(axis=-1)
        nearest = distances.argmin(axis=1)
        endpoints = np.linalg.norm(paths[row, :, None, -1] - references[None, :, -1], axis=-1)
        event_match = [event_sequence(candidate) == event_sequence(events[target]) for candidate, target in zip(opened[row], nearest)]
        semantics = semantic_endpoint_accuracy(paths[row, :, -1], data["semantic_targets"][idx])
        rows.append(dict(scene_id=str(data["scene_ids"][idx]), parent_id=str(data["parent_ids"][idx]),
                         candidate_matched_ADE_m=float(distances.min(axis=1).mean()),
                         reference_matched_ADE_m=float(distances.min(axis=0).mean()),
                         candidate_endpoint_error_m=float(endpoints.min(axis=1).mean()),
                         best_endpoint_error_m=float(endpoints.min()),
                         event_state_accuracy=float(np.mean((opened[row] > .5) == (events[nearest] > .5))),
                         event_sequence_accuracy=float(np.mean(event_match)),
                         semantic_goal_accuracy=None if semantics is None else float(semantics.mean()),
                         AnySemanticGoalAtK=None if semantics is None else float(semantics.any()),
                         SelectedSemanticGoalAtK=None, ValidAtK=None, AnyValidAtK=None,
                         UniqueValidAtK=None, ReferenceCoverageAtK=None, SelectedValidAtK=None))
    result = {}
    for key in rows[0]:
        if key in ("scene_id", "parent_id"):
            continue
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    result.update(examples=len(ids), parents=len(set(data["parent_ids"][ids])), candidates=int(paths.shape[1]),
                  semantic_evaluation_examples=sum(row["semantic_goal_accuracy"] is not None for row in rows),
                  geometry_validation="not implemented; no validity or route-type claims",
                  endpoint_semantics="predicted final end-effector xyz; target identities only used for evaluation")
    return result, rows


def paired_language_indices(data, ids):
    """Different target instructions on the exact same image and initial state."""
    swapped = []
    for idx in ids:
        label = data["semantic_targets"][idx]
        alternate = -1
        if label is not None:
            for other in ids:
                other_label = data["semantic_targets"][other]
                if (other != idx and data["parent_ids"][other] == data["parent_ids"][idx]
                        and data["image_hashes"][other] == data["image_hashes"][idx]
                        and np.array_equal(data["current"][other], data["current"][idx])
                        and other_label is not None and other_label["target_index"] != label["target_index"]):
                    alternate = int(other)
                    break
        swapped.append(alternate)
    return np.asarray(swapped)


@torch.no_grad()
def evaluate(model, data, ids, device, output=None):
    model.eval()
    predictions, events = [], []
    for start in range(0, len(ids), 32):
        group = ids[start:start + 32]
        paths, opened = model(torch.as_tensor(data["features"][group], device=device),
                              torch.as_tensor(data["current"][group], device=device))
        predictions.append(paths.cpu().numpy())
        events.append(opened.cpu().numpy())
    predictions, events = np.concatenate(predictions), np.concatenate(events)
    result, rows = observation_metrics(predictions, events, data, ids)
    switched_ids = paired_language_indices(data, ids)
    eligible = np.flatnonzero(switched_ids >= 0)
    swapped_predictions, swapped_events = None, None
    if len(eligible):
        alternate_features = torch.as_tensor(data["features"][switched_ids[eligible]], device=device)
        current = torch.as_tensor(data["current"][ids[eligible]], device=device)
        xyz, opened = model(alternate_features, current)
        swapped_predictions, swapped_events = xyz.cpu().numpy(), opened.cpu().numpy()
        wrong_metrics, _ = observation_metrics(swapped_predictions, swapped_events, data, ids[eligible])
        own_metrics, _ = observation_metrics(swapped_predictions, swapped_events, data, switched_ids[eligible])
        result["paired_language_control"] = dict(examples=len(eligible),
            same_image_other_target_language_original_goal_accuracy=wrong_metrics["semantic_goal_accuracy"],
            same_image_other_target_language_new_goal_accuracy=own_metrics["semantic_goal_accuracy"],
            mean_endpoint_response_m=float(np.linalg.norm(swapped_predictions[..., -1, :] - predictions[eligible, :, -1, :], axis=-1).mean()),
            explanation="Only the real-Qwen image-language feature changes; image bytes and current state remain identical")
        for local_row, source in enumerate(eligible):
            rows[source]["language_control_source_id"] = str(data["scene_ids"][switched_ids[source]])
            rows[source]["language_control_endpoint_response_m"] = float(np.linalg.norm(swapped_predictions[local_row, :, -1] - predictions[source, :, -1], axis=-1).mean())
    else:
        result["paired_language_control"] = None
    feature_np, current_np = data["features"][ids[:1]], data["current"][ids[:1]]
    timings = []
    for repeat in range(12):
        tic = synchronized_time(device)
        xyz, opened = model(torch.as_tensor(feature_np, device=device), torch.as_tensor(current_np, device=device))
        xyz.cpu().numpy()
        opened.cpu().numpy()
        elapsed = synchronized_time(device) - tic
        if repeat >= 2:
            timings.append(elapsed * 1000)
    result.update(cached_head_batch1_ms_median=float(np.median(timings)), cached_head_batch1_ms_p95=float(np.percentile(timings, 95)),
                  latency_scope="head and transfers only; Qwen feature extraction is separately measured and excluded here",
                  selection_score=-result["candidate_matched_ADE_m"])
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", paths=predictions, gripper_open=events,
                            scene_ids=data["scene_ids"][ids], parent_ids=data["parent_ids"][ids])
        if swapped_predictions is not None:
            np.savez_compressed(output / "paired_language_predictions.npz", paths=swapped_predictions,
                                gripper_open=swapped_events, original_scene_ids=data["scene_ids"][ids[eligible]],
                                language_source_ids=data["scene_ids"][switched_ids[eligible]])
        write_json(output / "metrics.json", result)
        write_json(output / "per_scene.json", rows)
    return result


def train(args):
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(.35)
        torch.cuda.reset_peak_memory_stats()
    seed_all(args.seed)
    data = load_observed_dataset(args.observations, args.supervision, args.cache_dir, args.horizon, args.pooling)
    train_ids = np.flatnonzero(data["splits"] == "TRAIN")
    dev_ids = np.flatnonzero(data["splits"] == "DEV_MODEL")
    if not len(train_ids) or not len(dev_ids):
        raise ValueError("parent-disjoint TRAIN and DEV_MODEL examples are required; DEV_COLLECTION cannot be relabelled silently")
    if set(data["parent_ids"][train_ids]) & set(data["parent_ids"][dev_ids]):
        raise ValueError("parent scene leakage")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        raise RuntimeError("existing checkpoint; use --resume or a new output")
    lock = out / "active.lock"
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError("active lock exists; inspect recorded PID before recovering")
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        config = vars(args).copy()
        config.update(dataset_fingerprint=data["fingerprint"], feature_dim=int(data["features"].shape[1]),
                      code_commit=os.environ.get("CODE_COMMIT", "unrecorded"), gpu_uuid=os.environ.get("RESEARCH_GPU_UUID"),
                      objective="saturation", selection_split="DEV_MODEL", selection_metric="negative candidate_matched_ADE_m",
                      condition_fields=["frozen real-Qwen RGB+instruction hidden states", "current gripper pose7", "current gripper open1"],
                      backbone_training="frozen cached Qwen; no LoRA or backbone update", endpoints="first fixed to current xyz; last learned",
                      reference_policy="all known positive successful recorded routes; no route-count/existence label",
                      event_loss="MSE on scaled gripper-open probability within positive assignment",
                      train_parents=len(set(data["parent_ids"][train_ids])), dev_parents=len(set(data["parent_ids"][dev_ids])),
                      train_examples=len(train_ids), dev_examples=len(dev_ids), skipped_supervision=data["skipped"],
                      cache_config=data["cache_config"])
        model = ObservedRouteHead(config["feature_dim"], args.horizon, args.candidates, args.width, args.depth).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        rng, sampler = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 100000)
        first_step, best, elapsed_before, exposures, history = 0, -float("inf"), 0., 0, []
        if args.resume:
            checkpoint = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
            for key in ("dataset_fingerprint", "feature_dim", "horizon", "candidates", "width", "depth", "steps", "seed", "lr", "batch_size", "event_scale", "pooling"):
                if config[key] != checkpoint["config"][key]:
                    raise ValueError("resume config mismatch: " + key)
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            restore_rng(checkpoint["rng"], rng)
            sampler.bit_generator.state = checkpoint["sampler_state"]
            first_step, best, elapsed_before = checkpoint["step"], checkpoint["best"], checkpoint["elapsed_s"]
            exposures, history = checkpoint["trajectory_exposures"], checkpoint["history"]
        write_json(out / "config.json", config)
        write_json(out / "source_hashes.json", data["source_hashes"])
        write_json(out / "status.json", dict(status="running", pid=os.getpid(), step=first_step))
        started, losses = synchronized_time(args.device), []
        for step in range(first_step + 1, args.steps + 1):
            model.train()
            ids = sampler.choice(train_ids, args.batch_size, replace=True)
            xyz, opened = model(torch.as_tensor(data["features"][ids], device=args.device),
                                torch.as_tensor(data["current"][ids], device=args.device))
            target_xyz = torch.as_tensor(data["paths"][ids], device=args.device)
            target_events = torch.as_tensor(data["events"][ids], device=args.device)
            prediction = torch.cat([xyz[:, :, 1:], opened[:, :, 1:, None] * args.event_scale], dim=-1)
            target = torch.cat([target_xyz[:, :, 1:], target_events[:, :, 1:, None] * args.event_scale], dim=-1)
            loss = positive_assignment_loss(prediction, target, data["path_mask"][ids], "saturation", rng)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            exposures += args.batch_size * args.candidates
            losses.append(float(loss.detach()))
            if step % 100 == 0:
                print(json.dumps(dict(step=step, loss=float(np.mean(losses[-100:])), elapsed_s=elapsed_before + synchronized_time(args.device) - started)), flush=True)
            if step % args.eval_every == 0 or step == args.steps or step == args.stop_after:
                metrics = evaluate(model, data, dev_ids, args.device)
                score = metrics["selection_score"]
                improved = score > best
                best = max(best, score)
                elapsed = elapsed_before + synchronized_time(args.device) - started
                history.append(dict(step=step, loss=float(np.mean(losses[-100:])), dev_model=metrics))
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                                  scaler=None, step=step, config=config, rng=rng_state(rng), sampler_state=sampler.bit_generator.state,
                                  best=best, history=history, elapsed_s=elapsed, trajectory_exposures=exposures)
                atomic_checkpoint(out / "last.pt", checkpoint)
                if improved:
                    atomic_checkpoint(out / "best.pt", checkpoint)
                write_json(out / "history.json", history)
                print(json.dumps(history[-1]), flush=True)
                if step == args.stop_after and step < args.steps:
                    write_json(out / "status.json", dict(status="interrupted_for_resume_check", step=step, exit_code=0))
                    return
        checkpoint = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        metrics = evaluate(model, data, dev_ids, args.device, out / "dev_model")
        elapsed = elapsed_before + synchronized_time(args.device) - started
        summary = dict(metrics=metrics, elapsed_s=elapsed,
                       gpu_hours_reserved=elapsed / 3600 if args.device.startswith("cuda") else 0.,
                       parameters=model.active_parameter_count(), trajectory_exposures=exposures,
                       peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.,
                       best_step=checkpoint["step"], best_checkpoint_sha256=sha256(out / "best.pt"),
                       prediction_sha256=sha256(out / "dev_model" / "predictions.npz"),
                       evidence_scope="frozen real-Qwen observational route baseline pilot; not mechanism advantage or robot execution")
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(status="completed", step=args.steps, exit_code=0))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", exit_code=1, exception=repr(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", required=True)
    parser.add_argument("--supervision", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--candidates", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pooling", choices=("mean", "last", "both"), default="both")
    parser.add_argument("--event-scale", type=float, default=.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.batch_size < 1 or not 1 <= args.threads <= 4:
        parser.error("positive steps, evaluation period and batch size; 1--4 threads required")
    train(args)


if __name__ == "__main__":
    main()
