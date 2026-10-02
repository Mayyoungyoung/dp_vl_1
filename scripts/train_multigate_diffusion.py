"""Matched independent/set diffusion controls on controlled development data.

No locked evaluation, no PG strength sweep, and no claim of observation input.
Invoke each arm in a separate recorded immutable-source job.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from routeset.common import decode_paths, encode_paths, seed_all, sha256, write_json
from routeset.diffusion_parameterization import ParameterizedDiffusionSchedule
from routeset.models import RouteDenoiser
from routeset.multigate import route_metrics
from routeset.multigate_diffusion import PairedTrainingStream, load_development
from routeset.train_v2 import atomic_checkpoint, restore_rng, rng_state, synchronized_time


ARMS = ("independent", "set_diffusion")


def _parent_mean(values, parents):
    return float(np.mean([np.mean(values[parents == parent]) for parent in np.unique(parents)]))


@torch.no_grad()
def evaluate(model, diffusion, data, ids, k, device, sampling_steps, seed, output=None, latency_requests=0):
    # Explicit evaluation generator never advances any training random stream.
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions = []
    for group in np.array_split(ids, max(1, (len(ids) + 31) // 32)):
        condition = torch.as_tensor(data["scenes"][group], device=device)
        residual = diffusion.sample(model, condition, k, sampling_steps, generator)
        predictions.append(decode_paths(residual, condition).cpu().numpy())
    predictions = np.concatenate(predictions)
    result = route_metrics(predictions, data["scenes"][ids],
        reference_modes=data["modes"][ids], reference_mask=data["path_mask"][ids])
    rows = result["per_scene"]
    parents = data["parent_ids"][ids]
    values = dict(valid_rate=rows["valid"].mean(1), any_valid=rows["valid"].any(1),
        unique_valid=rows["unique_count"], reference_coverage=rows["reference_coverage"],
        collision_rate=rows["collision"].mean(1), mean_length=rows["lengths"].mean(1),
        finite_rate=np.isfinite(predictions).all(axis=(-1, -2)).mean(1))
    metrics = {name: _parent_mean(value, parents) for name, value in values.items()}
    metrics.update(candidates=k, denoiser_forwards=min(sampling_steps, diffusion.num_steps),
        parents=len(np.unique(parents)), scenes=len(ids), sampling_seed=seed,
        aggregation="equal parents, then equal variants; repeats evaluated separately")
    metrics.update(ValidAtK=metrics["valid_rate"], AnyValidAtK=metrics["any_valid"],
                   UniqueValidAtK=metrics["unique_valid"], ReferenceCoverageAtK=metrics["reference_coverage"])
    if latency_requests:
        latencies = []
        # Warmup followed by complete single-request generation and exact checker.
        # No reference labels in the timed checker or generator.
        for number in range(latency_requests + 2):
            idx = ids[number % len(ids)]
            start = synchronized_time(device)
            condition = torch.as_tensor(data["scenes"][idx:idx + 1], device=device)
            paths = decode_paths(diffusion.sample(model, condition, k, sampling_steps, generator), condition).cpu().numpy()
            route_metrics(paths, data["scenes"][idx:idx + 1])
            elapsed = (synchronized_time(device) - start) * 1000
            if number >= 2:
                latencies.append(elapsed)
        metrics.update(request_generate_transfer_check_ms_p50=float(np.median(latencies)),
                       request_generate_transfer_check_ms_p95=float(np.percentile(latencies, 95)),
                       latency_requests=latency_requests,
                       timing_scope="controlled geometry encoding + denoising + decode + transfer + exact checker; no VLM")
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", paths=predictions,
            scene_ids=data["scene_ids"][ids], parent_ids=parents, **rows)
        per_scene = []
        for row, idx in enumerate(ids):
            record = {name: float(value[row]) for name, value in values.items()}
            record.update(scene_id=str(data["scene_ids"][idx]), parent_id=str(parents[row]),
                known_reference_types=int(len(np.unique(data["modes"][idx, data["path_mask"][idx]]))), candidates=k)
            per_scene.append(record)
        write_json(output / "per_scene.json", per_scene)
        write_json(output / "metrics.json", metrics)
    return metrics


def train_one(args):
    if args.arm not in ARMS or min(args.steps, args.batch_size, args.candidates, args.eval_every,
            args.sampling_steps, args.final_repeats) < 1 or not 1 <= args.threads <= 4:
        raise ValueError("valid arm, positive budgets and 1--4 threads required")
    if args.sampling_steps > args.diffusion_steps:
        raise ValueError("sampling steps cannot exceed the diffusion schedule")
    if not args.eval_candidates or min(args.eval_candidates) < 1 or args.latency_requests < 0 or args.lr <= 0:
        raise ValueError("positive evaluation K/lr and nonnegative latency request count required")
    torch.set_num_threads(args.threads)
    if args.device.startswith("cuda"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("select physical GPU1 explicitly with CUDA_VISIBLE_DEVICES=1")
        torch.cuda.set_per_process_memory_fraction(.35)
        torch.cuda.reset_peak_memory_stats()
    seed_all(args.seed)
    data, train_ids, dev_ids = load_development(args.data)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        raise RuntimeError("existing checkpoint: use --resume or a new output")
    if args.resume and not (out / "last.pt").exists():
        raise RuntimeError("resume checkpoint missing")
    lock = out / "active.lock"
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        source = Path(__file__).resolve().parents[1]
        source_hashes = {name: sha256(source / name) for name in (
            "scripts/train_multigate_diffusion.py", "routeset/multigate_diffusion.py",
            "routeset/diffusion_parameterization.py",
            "routeset/models.py", "routeset/diffusion.py", "routeset/common.py",
            "routeset/multigate.py", "routeset/train_v2.py")}
        config = vars(args).copy()
        config["parameterization"] = getattr(args, "parameterization", "epsilon")
        config.update(dataset_sha256=sha256(args.data), source_hashes=source_hashes,
            horizon=int(data["paths"].shape[-2]), cond_dim=int(data["scenes"].shape[-1]),
            code_commit=os.environ.get("CODE_COMMIT", "unrecorded"), gpu_uuid=os.environ.get("RESEARCH_GPU_UUID"),
            information="controlled true geometry and known endpoints; no observation/VLM claim",
            selection_split="DEV_MODEL", objective=config["parameterization"] + " MSE on same sampled positive routes",
            target_sampling="uniform parent then variant; distinct known modes first, legal repeats if needed",
            candidate_budget="exactly K returned, no hidden particles, repair or filtering; DDIM eta0",
            optimizer="AdamW weight_decay1e-4, constant LR, clip_grad_norm1, FP32")
        model = RouteDenoiser(config["cond_dim"], config["horizon"], args.width, args.depth,
                              set_attention=args.arm == "set_diffusion").to(args.device)
        diffusion = ParameterizedDiffusionSchedule(args.diffusion_steps, args.device,
                                                   parameterization=config["parameterization"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        stream = PairedTrainingStream(data, train_ids, args.seed)
        global_rng = np.random.default_rng(args.seed)
        start_step, best, elapsed_before, exposures, pool_access = 0, -float("inf"), 0., 0, 0
        history, losses = [], []
        if args.resume:
            ck = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
            ignored = {"resume", "stop_after", "output", "data", "code_commit", "gpu_uuid"}
            for key in set(config) | set(ck["config"]):
                if key not in ignored and config.get(key) != ck["config"].get(key):
                    raise ValueError("Resume config mismatch: " + key)
            model.load_state_dict(ck["model"])
            optimizer.load_state_dict(ck["optimizer"])
            scheduler.load_state_dict(ck["scheduler"])
            restore_rng(ck["rng"], global_rng)
            stream.load_state_dict(ck["stream"])
            start_step, best, elapsed_before = ck["step"], ck["best"], ck["elapsed_s"]
            exposures, pool_access = ck["trajectory_exposures"], ck["reference_pool_access"]
            history, losses = ck["history"], ck["loss_tail"]
        write_json(out / "config.json", config)
        write_json(out / "status.json", dict(status="running", pid=os.getpid(), step=start_step,
                                             resume_command="repeat recorded command with --resume"))
        started = synchronized_time(args.device)
        for step in range(start_step + 1, args.steps + 1):
            model.train()
            ids, _, paths, times, noise = stream.draw(args.batch_size, args.candidates, args.diffusion_steps)
            condition = torch.as_tensor(data["scenes"][ids], device=args.device)
            targets = encode_paths(torch.as_tensor(paths, device=args.device), condition)
            times, noise = times.to(args.device), noise.to(args.device)
            noisy, epsilon = diffusion.q_sample(targets, times, noise)
            objective_target = diffusion.training_target(targets, epsilon, times)
            optimizer.zero_grad(set_to_none=True)
            loss = (model(noisy, diffusion.normalized_time(times), condition) - objective_target).square().mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("nonfinite training loss at step %d" % step)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            scheduler.step()
            losses = (losses + [float(loss.detach())])[-200:]
            exposures += args.batch_size * args.candidates
            pool_access += int(data["path_mask"][ids].sum())
            if step % 200 == 0:
                print(json.dumps(dict(step=step, loss=float(np.mean(losses)),
                    elapsed_s=elapsed_before + synchronized_time(args.device) - started)), flush=True)
            if step % args.eval_every == 0 or step in (args.steps, args.stop_after):
                metrics = evaluate(model, diffusion, data, dev_ids, args.candidates, args.device,
                    args.sampling_steps, args.seed + 900000)
                score = metrics["unique_valid"] + .05 * metrics["valid_rate"]
                improved, best = score > best, max(best, score)
                history.append(dict(step=step, loss=float(np.mean(losses)), dev_model=metrics,
                                    training_stream_sha256=stream.digest))
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scaler=None, rng=rng_state(global_rng), stream=stream.state_dict(), step=step, best=best,
                    history=history, loss_tail=losses, config=config, trajectory_exposures=exposures,
                    reference_pool_access=pool_access, elapsed_s=elapsed_before + synchronized_time(args.device) - started)
                atomic_checkpoint(out / "last.pt", checkpoint)
                if improved:
                    atomic_checkpoint(out / "best.pt", checkpoint)
                write_json(out / "history.json", history)
                print(json.dumps(history[-1]), flush=True)
                if step == args.stop_after and step < args.steps:
                    write_json(out / "status.json", dict(status="interrupted_for_resume_check", step=step, exit_code=0))
                    return
        ck = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        final_metrics, prediction_hashes = {}, {}
        for k in args.eval_candidates:
            repeats = []
            for repeat in range(args.final_repeats):
                destination = out / "dev_model" / ("k%d" % k) / ("repeat%d" % repeat)
                repeats.append(evaluate(model, diffusion, data, dev_ids, k, args.device, args.sampling_steps,
                    args.seed + 910000 + repeat, destination,
                    args.latency_requests if repeat == 0 else 0))
                prediction_hashes[str(destination.relative_to(out))] = sha256(destination / "predictions.npz")
            means = {name: float(np.mean([r[name] for r in repeats])) for name in (
                "valid_rate", "any_valid", "unique_valid", "reference_coverage", "collision_rate", "mean_length", "finite_rate")}
            final_metrics["k%d" % k] = dict(mean=means, repeats=repeats,
                budget_status="trained K" if k == args.candidates else "unseen-K inference transfer; not K-conditioned training")
        elapsed = elapsed_before + synchronized_time(args.device) - started
        summary = dict(arm=args.arm, metrics=final_metrics, elapsed_s=elapsed,
            gpu_hours_reserved=elapsed / 3600 if args.device.startswith("cuda") else 0.,
            trajectory_exposures=exposures, gradient_target_slots=exposures, reference_pool_access=pool_access,
            training_stream_sha256=stream.digest, selection_score=ck["best"], best_step=ck["step"],
            parameters=sum(p.numel() for p in model.parameters()), active_parameters=model.active_parameter_count(),
            best_checkpoint_sha256=sha256(out / "best.pt"), last_checkpoint_sha256=sha256(out / "last.pt"),
            prediction_sha256=prediction_hashes,
            peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device.startswith("cuda") else 0.,
            cost_scope="training, intermediate DEV evaluation, final separate repeats and latency measurement; one real training seed")
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(status="completed", step=args.steps, exit_code=0))
        print(json.dumps(dict(arm=args.arm, best_step=ck["step"], elapsed_s=elapsed,
            mean_metrics={key: value["mean"] for key, value in final_metrics.items()},
            training_stream_sha256=stream.digest)), flush=True)
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", exit_code=1, exception=repr(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--width", type=int, default=192)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--diffusion-steps", type=int, default=100)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--parameterization", choices=("epsilon", "v"), default="epsilon")
    parser.add_argument("--eval-candidates", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--final-repeats", type=int, default=3)
    parser.add_argument("--latency-requests", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    train_one(parser.parse_args())


if __name__ == "__main__":
    main()
