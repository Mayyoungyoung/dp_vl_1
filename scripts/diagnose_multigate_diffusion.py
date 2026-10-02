"""Read-only fixed TRAIN32/DEV128 denoising and sampler diagnostics.

No parameter changes, reference-guided repair, PG sweep, or new training. Uses
the original sampler with passive hooks, so the actual output stays identical.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from routeset.common import decode_paths, encode_paths, sha256, write_json
from routeset.diffusion import DiffusionSchedule
from routeset.models import RouteDenoiser
from routeset.multigate import CLEARANCE, route_metrics, wall_boxes
from routeset.multigate_diffusion import load_development


def segment_box_distance(starts, ends, lower, upper):
    """Exact unsigned Euclidean segment/AABB distance by quadratic intervals.

    Each coordinate's nearest box face changes only at its two crossings. On
    every resulting interval, squared distance is a convex quadratic in t.
    Returns [segments, boxes]; intersections have distance zero.
    """
    p = np.asarray(starts, dtype=np.float64)[:, None, :]
    direction = (np.asarray(ends) - np.asarray(starts))[:, None, :]
    lower, upper = np.asarray(lower)[None], np.asarray(upper)[None]
    shape = (len(starts), lower.shape[1], 3)
    faces = []
    for bound in (lower, upper):
        ratio = np.zeros(shape)
        np.divide(bound - p, direction, out=ratio, where=np.abs(direction) > 1e-12)
        faces.append(np.clip(ratio, 0., 1.))
    times = np.sort(np.concatenate([np.zeros(shape[:2] + (1,)), *faces, np.ones(shape[:2] + (1,))], axis=-1), axis=-1)
    left, right = times[..., :-1], times[..., 1:]
    middle = (left + right) * .5
    probe = p[..., None, :] + middle[..., None] * direction[..., None, :]
    active = (probe < lower[..., None, :]) | (probe > upper[..., None, :])
    closest_face = np.where(probe < lower[..., None, :], lower[..., None, :], upper[..., None, :])
    quadratic = (direction[..., None, :] ** 2 * active).sum(-1)
    linear = (direction[..., None, :] * (p[..., None, :] - closest_face) * active).sum(-1)
    minimizer = middle.copy()
    np.divide(-linear, quadratic, out=minimizer, where=quadratic > 1e-20)
    minimizer = np.minimum(np.maximum(minimizer, left), right)
    points = p[..., None, :] + minimizer[..., None] * direction[..., None, :]
    delta = np.maximum(lower[..., None, :] - points, 0.) + np.minimum(upper[..., None, :] - points, 0.)
    return np.sqrt((delta ** 2).sum(-1).min(-1))


def _distribution(values):
    values = np.asarray(values)
    if not values.size:
        return dict(count=0, mean=None, p50=None, p95=None, max=None)
    return dict(count=int(values.size), mean=float(values.mean()), p50=float(np.median(values)),
                p95=float(np.percentile(values, 95)), max=float(values.max()))


@torch.no_grad()
def denoise_diagnostics(model, schedule, data, ids, device):
    condition = torch.as_tensor(data["scenes"][ids], device=device)
    rng = np.random.default_rng(20261002)
    selected = []
    for idx in ids:
        valid = np.flatnonzero(data["path_mask"][idx] & (data["modes"][idx] >= 0))
        selected.append(np.resize(rng.permutation(valid), 4))
    selected = np.asarray(selected)
    paths = torch.as_tensor(data["paths"][ids[:, None], selected], device=device)
    target = encode_paths(paths, condition)
    noise = torch.randn(target.shape, generator=torch.Generator().manual_seed(501)).to(device)
    rows = []
    for timestep in (0, 10, 25, 50, 75, 90, 95, 99):
        times = torch.full((len(ids),), timestep, device=device)
        noisy, _ = schedule.q_sample(target, times, noise)
        alpha = float(schedule.alpha_bars[timestep])
        predictions = {}
        for name, cond in (("correct", condition), ("shuffled", condition.roll(1, 0))):
            epsilon = model(noisy, schedule.normalized_time(times), cond)
            raw = schedule.x0_from_eps(noisy, times, epsilon)
            clipped = raw.clamp(-schedule.clip_x0, schedule.clip_x0)
            predictions[name] = epsilon
            rows.append(dict(timestep=timestep, condition=name, alpha_bar=alpha,
                epsilon_to_x0_error_multiplier=((1 - alpha) / alpha) ** .5,
                epsilon_mse=float((epsilon - noise).square().mean()),
                x0_rmse=float((raw - target).square().mean().sqrt()),
                x0_axis_rmse=(raw - target).square().mean((0, 1, 2)).sqrt().cpu().tolist(),
                clipped_x0_rmse=float((clipped - target).square().mean().sqrt()),
                clip_fraction=float((raw.abs() > schedule.clip_x0).float().mean())))
        rows[-2]["shuffled_epsilon_change_rmse"] = float((predictions["correct"] - predictions["shuffled"]).square().mean().sqrt())
    return dict(scene_ids=data["scene_ids"][ids].tolist(), parent_ids=data["parent_ids"][ids].tolist(),
        target_reference_ids=selected.tolist(), target_axis_rms=target.square().mean((0, 1, 2)).sqrt().cpu().tolist(),
        target_axis_std=target.flatten(0, 2).std(0).cpu().tolist(), rows=rows)


@torch.no_grad()
def sample_diagnostics(model, schedule, data, ids, device, output):
    predictions, last_deltas = [], []
    trace = {}
    generator = torch.Generator(device=device).manual_seed(900000)
    for group in np.array_split(ids, max(1, (len(ids) + 31) // 32)):
        condition = torch.as_tensor(data["scenes"][group], device=device)
        last = {}
        def observe(module, inputs, epsilon):
            noisy, normalized_time, _ = inputs
            timestep = int((normalized_time[0] * (schedule.num_steps - 1)).round())
            times = torch.full((len(noisy),), timestep, device=noisy.device)
            raw = schedule.x0_from_eps(noisy, times, epsilon)
            trace.setdefault(timestep, []).append(dict(elements=raw.numel(),
                clip_count=int((raw.abs() > schedule.clip_x0).sum()),
                x0_square_sum=float(raw.square().sum()), epsilon_square_sum=float(epsilon.square().sum())))
            if timestep == 0:
                last["noisy"] = noisy.clone()
        hook = model.register_forward_hook(observe)
        try:
            residual = schedule.sample(model, condition, 4, 40, generator)
        finally:
            hook.remove()
        last_deltas.append((residual - last["noisy"]).square().mean((-1, -2)).sqrt().cpu().numpy())
        predictions.append(decode_paths(residual, condition).cpu().numpy())
    predictions = np.concatenate(predictions)
    last_deltas = np.concatenate(last_deltas)
    checked = route_metrics(predictions, data["scenes"][ids], data["modes"][ids], data["path_mask"][ids])
    valid = checked["per_scene"]["valid"]
    zbase = np.stack([np.linspace(s[2], s[5], predictions.shape[-2]) for s in data["scenes"][ids]])
    z_mae = np.abs(predictions[..., 2] - zbase[:, None]).mean(-1)
    clearance = []
    for paths, scene in zip(predictions, data["scenes"][ids]):
        lower, upper = wall_boxes(scene)
        distances = segment_box_distance(paths[:, :-1].reshape(-1, 3), paths[:, 1:].reshape(-1, 3),
                                          lower - CLEARANCE, upper + CLEARANCE)
        clearance.append(distances.reshape(4, paths.shape[-2] - 1, -1).min((1, 2)))
    clearance = np.asarray(clearance)
    result = dict(metrics={key: value for key, value in checked.items() if key != "per_scene"},
        z_mae=_distribution(z_mae), z_mae_valid=_distribution(z_mae[valid]), z_mae_invalid=_distribution(z_mae[~valid]),
        final_denoise_rms=_distribution(last_deltas), final_denoise_valid=_distribution(last_deltas[valid]),
        final_denoise_invalid=_distribution(last_deltas[~valid]),
        minimum_segment_expanded_box_distance=_distribution(clearance),
        distance_zero_fraction=float((clearance < 1e-10).mean()),
        clearance_scope="exact unsigned segment-to-clearance-expanded-AABB Euclidean minimum; intersecting routes have0; does not measure penetration depth",
        trace=[dict(timestep=t, elements=sum(v["elements"] for v in rows),
            clip_fraction=sum(v["clip_count"] for v in rows) / sum(v["elements"] for v in rows),
            x0_rms=(sum(v["x0_square_sum"] for v in rows) / sum(v["elements"] for v in rows)) ** .5,
            epsilon_rms=(sum(v["epsilon_square_sum"] for v in rows) / sum(v["elements"] for v in rows)) ** .5)
            for t, rows in sorted(trace.items(), reverse=True)])
    np.savez_compressed(output, paths=predictions, scene_ids=data["scene_ids"][ids], parent_ids=data["parent_ids"][ids],
        valid=valid, z_mae=z_mae, last_denoise_rms=last_deltas, min_segment_box_distance=clearance)
    result["prediction_sha256"] = sha256(output)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--runs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.device.startswith("cuda"):
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise RuntimeError("GPU1 required")
        torch.cuda.set_per_process_memory_fraction(.35)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Numeric geometry sanity check with known crossing and parallel distances.
    got = segment_box_distance([[0., 0., 0.], [2., 2., 0.]], [[2., 0., 0.], [2., 3., 0.]], [[.5, -.5, -.5]], [[1., .5, .5]])
    np.testing.assert_allclose(got[:, 0], [0., np.sqrt(3.25)], atol=1e-12)
    data, train, dev = load_development(args.data)
    if len(train) < 32 or len(dev) != 128:
        raise ValueError("fixed diagnostic requires first32 TRAIN and exactly128 DEV")
    result = dict(data_sha256=sha256(args.data), code_commit=os.environ.get("CODE_COMMIT"), device=args.device,
                  protocol="fixed first32 TRAIN + all128 DEV; deterministic true/shuffled condition; no training or repair", checkpoints={})
    started = time.perf_counter()
    for arm in ("independent", "set_diffusion"):
        last_step, last_model_state, last_key = None, None, None
        for checkpoint_name in ("best", "last"):
            path = Path(args.runs) / arm / (checkpoint_name + ".pt")
            checkpoint = torch.load(path, map_location=args.device, weights_only=False)
            config = checkpoint["config"]
            if config.get("parameterization", "epsilon") != "epsilon":
                raise ValueError("this epsilon diagnostic does not yet interpret v model outputs")
            if config["dataset_sha256"] != result["data_sha256"]:
                raise ValueError("checkpoint data mismatch")
            key = arm + "_" + checkpoint_name
            record = dict(checkpoint=str(path), checkpoint_sha256=sha256(path), step=checkpoint["step"])
            if last_step == checkpoint["step"] and all(torch.equal(last_model_state[k], v) for k, v in checkpoint["model"].items()):
                record["identical_weights_diagnostic_reference"] = last_key
                result["checkpoints"][key] = record
                continue
            model = RouteDenoiser(config["cond_dim"], config["horizon"], config["width"], config["depth"],
                                  set_attention=arm == "set_diffusion").to(args.device).eval()
            model.load_state_dict(checkpoint["model"])
            schedule = DiffusionSchedule(config["diffusion_steps"], args.device)
            record["denoise"] = {name: denoise_diagnostics(model, schedule, data, ids, args.device)
                                  for name, ids in (("TRAIN32", train[:32]), ("DEV128", dev))}
            record["actual_sample_dev128"] = sample_diagnostics(model, schedule, data, dev, args.device, output / (key + ".npz"))
            result["checkpoints"][key] = record
            last_step, last_model_state, last_key = checkpoint["step"], checkpoint["model"], key
            print(json.dumps(dict(checkpoint=key, step=checkpoint["step"], elapsed_s=time.perf_counter() - started)), flush=True)
    result["elapsed_s"] = time.perf_counter() - started
    write_json(output / "diagnostic.json", result)
    print(json.dumps(dict(completed=True, elapsed_s=result["elapsed_s"])), flush=True)


if __name__ == "__main__":
    main()
