"""Observation-only Qwen3-VL route training with real online last-layer LoRA.

No frozen hidden-state cache is accepted. Only RGB, instruction and current
gripper state enter the model. Route/target labels are supervision/evaluation.
Checkpoint files contain the route head and adapters, never the frozen base.
Run --self-test for a small actual Qwen architecture CPU gradient/resume check.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from routeset.common import seed_all, sha256, write_json
from routeset.observed_route_head import (ObservedRouteHead, QWEN_REVISION,
    resample_event_segments)
from routeset.train_v2 import (atomic_checkpoint, positive_assignment_loss,
    restore_rng, rng_state, synchronized_time)
from scripts.observation_cache_qwen import read_manifest
from scripts.train_observed_routes import observation_metrics, paired_language_indices


class LoRALinear(nn.Module):
    """Frozen linear + FP32 low-rank update; zero B gives exact initial base."""

    def __init__(self, base, rank=8, alpha=16.):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank < 1:
            raise ValueError("LoRA requires a linear layer and positive rank")
        self.base = base.requires_grad_(False)
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x):
        output = self.base(x)
        update = F.linear(F.linear(x.float(), self.lora_A), self.lora_B) * self.scale
        return output + update.to(output.dtype)


def install_lora(backbone, rank=8, alpha=16.):
    backbone.requires_grad_(False)
    layers = backbone.model.language_model.layers
    if len(layers) < 2:
        raise ValueError("At least two language layers required")
    replaced = []
    for index in range(len(layers) - 2, len(layers)):
        for projection in ("q_proj", "v_proj"):
            attention = layers[index].self_attn
            setattr(attention, projection, LoRALinear(getattr(attention, projection), rank, alpha))
            replaced.append("model.language_model.layers.%d.self_attn.%s" % (index, projection))
    return replaced


def adapters(backbone):
    return {name: value for name, value in backbone.named_parameters()
            if name.endswith(".lora_A") or name.endswith(".lora_B")}


def tensor_hash(tensor):
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def adapter_state(backbone):
    return {name: value.detach().cpu().clone() for name, value in adapters(backbone).items()}


def load_adapters(backbone, state):
    parameters = adapters(backbone)
    if set(parameters) != set(state):
        raise ValueError("Checkpoint adapter modules do not match the model")
    with torch.no_grad():
        for name, parameter in parameters.items():
            parameter.copy_(state[name].to(parameter))


def head_state_hash(head):
    payload = {name: tensor_hash(value) for name, value in head.state_dict().items()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_head_initialization(head, path, expected, observations, supervision):
    """Strictly initialize from the same-data, frozen-Qwen ordinary route head."""
    path = Path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source_config = checkpoint.get("config", {})
    for key in ("feature_dim", "horizon", "candidates", "width", "depth", "pooling", "event_scale"):
        if source_config.get(key) != expected[key]:
            raise ValueError("Head initialization architecture/target mismatch: " + key)
    cache = source_config.get("cache_config", {})
    if (cache.get("model") != "Qwen/Qwen3-VL-2B-Instruct" or cache.get("revision") != QWEN_REVISION
            or cache.get("processor") != QWEN_REVISION or cache.get("model_trainable_parameter_count") != 0
            or cache.get("max_pixels") != expected["max_pixels"]):
        raise ValueError("Head initialization requires matching frozen Qwen/processor/pixel provenance")
    if cache.get("manifest_sha256") != sha256(observations):
        raise ValueError("Head initialization uses different observation manifest")
    source_hash_path = path.parent / "source_hashes.json"
    source_hashes = json.loads(source_hash_path.read_text(encoding="utf-8"))
    if (source_hashes.get(source_config.get("observations")) != sha256(observations)
            or source_hashes.get(source_config.get("supervision")) != sha256(supervision)):
        raise ValueError("Head initialization observation/supervision hashes differ")
    state = checkpoint.get("model")
    expected_state = head.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ValueError("Head initialization must contain the ordinary route head state only")
    if any(state[name].shape != value.shape for name, value in expected_state.items()):
        raise ValueError("Head initialization parameter shapes do not match")
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError("Head initialization contains non-finite parameters")
    head.load_state_dict(state, strict=True)
    return dict(path=str(path.resolve()), sha256=sha256(path), source_config=source_config,
                source_hashes_sha256=sha256(source_hash_path), source_step=checkpoint["step"],
                source_trajectory_exposures=checkpoint.get("trajectory_exposures"),
                head_state_sha256=head_state_hash(head), same_observation_and_supervision_hashes=True,
                initialization_scope="head weights only; both online methods reset optimizer and new adapter B=0")


def begin_audit(backbone):
    return {name: dict(initial_sha256=tensor_hash(value), shape=list(value.shape),
                       ever_nonzero_gradient=False, max_gradient_abs=0., first_gradient_abs=None)
            for name, value in adapters(backbone).items()}


def audit_gradients(backbone, audit):
    for name, parameter in adapters(backbone).items():
        if parameter.grad is None:
            raise RuntimeError("LoRA gradient missing: " + name)
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("LoRA gradient is non-finite: " + name)
        maximum = float(parameter.grad.detach().abs().max())
        record = audit[name]
        if record["first_gradient_abs"] is None:
            record["first_gradient_abs"] = maximum
        record["ever_nonzero_gradient"] |= maximum > 0
        record["max_gradient_abs"] = max(record["max_gradient_abs"], maximum)


def finish_audit(backbone, audit, require_update=False):
    result = {}
    for name, parameter in adapters(backbone).items():
        record = dict(audit[name], final_sha256=tensor_hash(parameter))
        record["changed_from_initial"] = record["initial_sha256"] != record["final_sha256"]
        result[name] = record
        if require_update and not (record["ever_nonzero_gradient"] and record["changed_from_initial"]):
            raise RuntimeError("LoRA parameter has no measured gradient/update: " + name)
    return result


def resolve(base, filename):
    path = Path(filename)
    return path if path.is_absolute() else base / path


def load_raw_observations(observations, supervision, horizon):
    """Same event-preserving labels as frozen head, with no hidden-state cache."""
    observations, supervision = Path(observations), Path(supervision)
    rows = read_manifest(observations)
    label_rows = [json.loads(line) for line in supervision.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    labels = {row["id"]: row for row in label_rows}
    if len(labels) != len(label_rows):
        raise ValueError("duplicate supervision id")
    hashes = {str(observations): sha256(observations), str(supervision): sha256(supervision)}
    samples, skipped = [], []
    for row in rows:
        if row["split"] not in ("TRAIN", "DEV_MODEL"):
            continue
        label = labels.get(row["id"])
        if label is None or not label.get("routes"):
            skipped.append(dict(id=row["id"], reason="missing_supervision_or_successful_routes"))
            continue
        if label["parent_id"] != row["parent_id"] or label.get("split", row["split"]) != row["split"]:
            raise ValueError("supervision identity/split mismatch")
        image = resolve(observations.parent, row["image"])
        current_file = resolve(supervision.parent, label["observation"])
        with np.load(current_file, allow_pickle=False) as archive:
            # Explicit whitelist: never flatten task_low_dim_state or targets.
            current = np.r_[archive["gripper_pose"].reshape(7), archive["gripper_open"].reshape(1)].astype(np.float32)
        if not np.isfinite(current).all():
            raise ValueError("non-finite current gripper state")
        paths, events = [], []
        for filename in label["routes"]:
            route = resolve(supervision.parent, filename)
            with np.load(route, allow_pickle=False) as archive:
                xyz, opened = resample_event_segments(archive["gripper_pose"], archive["gripper_open"], horizon)
            if np.linalg.norm(xyz[0] - current[:3]) > .005:
                raise ValueError("reference does not start at recorded current state")
            paths.append(xyz)
            events.append(opened)
            hashes[str(route)] = sha256(route)
        hashes[str(image)], hashes[str(current_file)] = sha256(image), sha256(current_file)
        samples.append(dict(row, image_path=str(image), image_hash=hashes[str(image)], current=current,
                            paths=paths, events=events, semantic_targets=label.get("semantic_targets")))
    if not samples:
        raise ValueError("No TRAIN/DEV_MODEL observation supervision; collection-only data is not training data")
    max_refs = max(len(sample["paths"]) for sample in samples)
    paths = np.zeros((len(samples), max_refs, horizon, 3), dtype=np.float32)
    events = np.zeros((len(samples), max_refs, horizon), dtype=np.float32)
    mask = np.zeros((len(samples), max_refs), dtype=bool)
    for index, sample in enumerate(samples):
        count = len(sample["paths"])
        paths[index, :count], events[index, :count], mask[index, :count] = sample["paths"], sample["events"], True
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(current=np.stack([sample["current"] for sample in samples]), paths=paths, events=events,
                path_mask=mask, scene_ids=np.asarray([s["id"] for s in samples]),
                parent_ids=np.asarray([s["parent_id"] for s in samples]), splits=np.asarray([s["split"] for s in samples]),
                image_paths=[s["image_path"] for s in samples], image_hashes=np.asarray([s["image_hash"] for s in samples]),
                instructions=[s["instruction"] for s in samples], semantic_targets=[s["semantic_targets"] for s in samples],
                source_hashes=hashes, fingerprint=fingerprint, skipped=skipped)


def online_feature(backbone, processor, image_path, instruction, device, pooling):
    """Autograd intentionally enabled; no labels or answer tokens in messages."""
    from PIL import Image
    with Image.open(image_path) as image:
        rgb = image.convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[rgb], return_tensors="pt").to(device)
    # No inference_mode/no_grad here: LoRA must participate in the real forward.
    outputs = backbone.model(**inputs, use_cache=False, return_dict=True)
    selected = outputs.last_hidden_state[0, inputs.attention_mask[0].bool()].float()
    mean, last = selected.mean(0), selected[-1]
    feature = torch.cat([mean, last]) if pooling == "both" else (mean if pooling == "mean" else last)
    if not torch.isfinite(feature).all():
        raise RuntimeError("non-finite online Qwen feature")
    return feature[None], int(inputs.attention_mask.sum())


@torch.no_grad()
def evaluate(backbone, processor, head, data, ids, args, output=None):
    backbone.eval()
    head.eval()
    predictions, events, timings = [], [], []
    for idx in ids:
        start = synchronized_time(args.device)
        feature, _ = online_feature(backbone, processor, data["image_paths"][idx], data["instructions"][idx], args.device, args.pooling)
        paths, opened = head(feature, torch.as_tensor(data["current"][idx:idx + 1], device=args.device))
        predictions.append(paths.cpu().numpy())
        events.append(opened.cpu().numpy())
        timings.append((synchronized_time(args.device) - start) * 1000)
    predictions, events = np.concatenate(predictions), np.concatenate(events)
    metrics, rows = observation_metrics(predictions, events, data, ids)
    switched = paired_language_indices(data, ids)
    positions = {int(idx): position for position, idx in enumerate(ids)}
    eligible = np.flatnonzero(switched >= 0)
    if len(eligible):
        swapped = np.asarray([positions[int(switched[position])] for position in eligible])
        wrong, _ = observation_metrics(predictions[swapped], events[swapped], data, ids[eligible])
        own, _ = observation_metrics(predictions[swapped], events[swapped], data, switched[eligible])
        metrics["paired_language_control"] = dict(examples=len(eligible),
            same_image_other_target_language_original_goal_accuracy=wrong["semantic_goal_accuracy"],
            same_image_other_target_language_new_goal_accuracy=own["semantic_goal_accuracy"],
            mean_endpoint_response_m=float(np.linalg.norm(predictions[swapped, :, -1] - predictions[eligible, :, -1], axis=-1).mean()),
            explanation="Online predictions at this adapter checkpoint; alternate instructions share exact image and current state")
    else:
        metrics["paired_language_control"] = None
    measured = timings[2:] if len(timings) > 2 else timings
    metrics.update(selection_score=-metrics["candidate_matched_ADE_m"],
                   online_request_ms_median=float(np.median(measured)), online_request_ms_p95=float(np.percentile(measured, 95)),
                   timed_requests=len(measured), latency_scope="RGB file read/processor/transfers/Qwen/head/output transfer; excludes model load and any unimplemented geometry/score/execution")
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output / "predictions.npz", paths=predictions, gripper_open=events,
                            scene_ids=data["scene_ids"][ids], parent_ids=data["parent_ids"][ids])
        write_json(output / "metrics.json", metrics)
        write_json(output / "per_scene.json", rows)
    return metrics


def train(args):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    import transformers
    torch.set_num_threads(args.threads)
    if args.device == "cuda":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
            raise ValueError("Current authorization requires CUDA_VISIBLE_DEVICES=1")
        torch.cuda.set_per_process_memory_fraction(.35)
        torch.cuda.reset_peak_memory_stats()
    if transformers.__version__ != "4.57.1" or not torch.__version__.startswith("2.4.1"):
        raise ValueError("Use pinned .venv-qwen: torch 2.4.1, transformers 4.57.1")
    seed_all(args.seed)
    model_path = Path(args.model)
    provenance = json.loads((model_path / "provenance.json").read_text())
    if provenance.get("model_id") != "Qwen/Qwen3-VL-2B-Instruct" or provenance.get("revision") != QWEN_REVISION or not provenance.get("all_hashes_verified"):
        raise ValueError("Pinned official model/processor provenance required")
    data = load_raw_observations(args.observations, args.supervision, args.horizon)
    train_ids, dev_ids = [np.flatnonzero(data["splits"] == split) for split in ("TRAIN", "DEV_MODEL")]
    if not len(train_ids) or not len(dev_ids) or set(data["parent_ids"][train_ids]) & set(data["parent_ids"][dev_ids]):
        raise ValueError("Nonempty parent-disjoint TRAIN and DEV_MODEL required")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "last.pt").exists() and not args.resume:
        raise RuntimeError("Existing checkpoint: use --resume or a new output")
    lock = out / "active.lock"
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    try:
        setup_start = synchronized_time(args.device)
        backbone = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
            attn_implementation="sdpa", device_map={"": args.device})
        if len(backbone.model.language_model.layers) != 28 or backbone.config.text_config.hidden_size != 2048:
            raise ValueError("Unexpected pinned 2B architecture")
        seed_all(args.seed + 200000)
        modules = install_lora(backbone, args.rank, args.alpha) if args.adapter_mode == "lora" else []
        if args.adapter_mode == "frozen":
            backbone.requires_grad_(False)
        backbone.eval()  # eval mode does not disable autograd; all LoRA stays trainable.
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
            min_pixels=64 * 32 * 32, max_pixels=args.max_pixels)
        feature_dim = backbone.config.text_config.hidden_size * (2 if args.pooling == "both" else 1)
        # LoRA initialization consumes RNG; reset so frozen and LoRA controls
        # start from the same route-head parameters for the same seed.
        seed_all(args.seed)
        head = ObservedRouteHead(feature_dim, args.horizon, args.candidates, args.width, args.depth).to(args.device)
        head_init = None
        if args.head_init:
            head_init = load_head_initialization(head, args.head_init, dict(vars(args), feature_dim=feature_dim),
                                                args.observations, args.supervision)
        adapter_params = list(adapters(backbone).values())
        trainable = list(head.parameters()) + adapter_params
        if set(name for name, parameter in backbone.named_parameters() if parameter.requires_grad) != set(adapters(backbone)):
            raise RuntimeError("Unexpected trainable base parameters")
        groups = [{"params": head.parameters(), "lr": args.lr, "weight_decay": 1e-4}]
        if adapter_params:
            groups.append({"params": adapter_params, "lr": args.lora_lr, "weight_decay": 0.})
        optimizer = torch.optim.AdamW(groups)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        audit = begin_audit(backbone)
        config = dict(vars(args), feature_dim=feature_dim, dataset_fingerprint=data["fingerprint"],
            head_initialization=head_init, head_init_sha256=None if head_init is None else head_init["sha256"],
            initial_head_state_sha256=head_state_hash(head),
            model_revision=QWEN_REVISION, processor_revision=QWEN_REVISION,
            model_provenance_sha256=sha256(model_path / "provenance.json"), torch=torch.__version__, transformers=transformers.__version__,
            code_commit=os.environ.get("CODE_COMMIT", "unrecorded"), gpu_uuid=os.environ.get("RESEARCH_GPU_UUID"),
            lora_modules=modules, adapter_initialization_seed=args.seed + 200000,
            lora_parameters=sum(p.numel() for p in adapter_params), head_parameters=head.active_parameter_count(),
            input_contract=["RGB", "instruction", "current gripper pose7", "current gripper open1"],
            hidden_cache_used=False, gradient_explanation="B initialized zero: A first-step gradient is zero; B must receive gradient, A can update after B changes; adapter weight decay is zero",
            objective="saturation assignment on xyz and event_scale*open probability; same as frozen route head",
            effective_batch=args.batch_size * args.accumulation, selection_split="DEV_MODEL", selection_metric="negative candidate_matched_ADE_m",
            train_examples=len(train_ids), dev_examples=len(dev_ids), train_parents=len(set(data["parent_ids"][train_ids])),
            dev_parents=len(set(data["parent_ids"][dev_ids])), skipped_supervision=data["skipped"])
        rng, sampler = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 100000)
        first, best, elapsed_before, setup_before, exposures, requests, history = 0, -float("inf"), 0., 0., 0, 0, []
        if args.resume:
            checkpoint = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
            for key in ("dataset_fingerprint", "model_provenance_sha256", "feature_dim", "adapter_mode", "rank", "alpha", "horizon", "candidates", "width", "depth", "steps", "seed", "lr", "lora_lr", "batch_size", "accumulation", "pooling", "event_scale", "max_pixels", "head_init_sha256"):
                if config[key] != checkpoint["config"].get(key):
                    raise ValueError("Resume config mismatch: " + key)
            head.load_state_dict(checkpoint["head"])
            load_adapters(backbone, checkpoint["adapters"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            restore_rng(checkpoint["rng"], rng)
            sampler.bit_generator.state = checkpoint["sampler_state"]
            first, best, elapsed_before = checkpoint["step"], checkpoint["best"], checkpoint["elapsed_s"]
            setup_before = checkpoint["setup_seconds_total"]
            exposures, requests, history, audit = checkpoint["trajectory_exposures"], checkpoint["online_train_requests"], checkpoint["history"], checkpoint["adapter_audit"]
        setup_seconds = synchronized_time(args.device) - setup_start
        setup_total = setup_before + setup_seconds
        write_json(out / "config.json", config)
        if head_init:
            write_json(out / "head_initialization.json", head_init)
        write_json(out / "source_hashes.json", data["source_hashes"])
        write_json(out / "status.json", dict(status="running", run_id=out.name, pid=os.getpid(), step=first, resume_command="rerun the exact command with --resume"))
        started, losses = synchronized_time(args.device), []
        if args.head_init and not args.resume:
            # Include the common pretrained starting point in DEV selection;
            # a harmful fine-tune must not silently replace its stronger head.
            metrics = evaluate(backbone, processor, head, data, dev_ids, args)
            best = metrics["selection_score"]
            elapsed = synchronized_time(args.device) - started
            history.append(dict(step=0, loss=None, dev_model=metrics, role="shared pretrained head before new updates"))
            checkpoint = dict(head=head.state_dict(), adapters=adapter_state(backbone), optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), scaler=None, step=0, config=config, rng=rng_state(rng),
                sampler_state=sampler.bit_generator.state, best=best, history=history, adapter_audit=audit,
                elapsed_s=elapsed, setup_seconds_total=setup_total, trajectory_exposures=0, online_train_requests=0)
            atomic_checkpoint(out / "last.pt", checkpoint)
            atomic_checkpoint(out / "best.pt", checkpoint)
            write_json(out / "initial_metrics.json", metrics)
            write_json(out / "history.json", history)
            print(json.dumps(history[-1]), flush=True)
        for step in range(first + 1, args.steps + 1):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            ids = sampler.choice(train_ids, config["effective_batch"], replace=True)
            step_loss = 0.
            for idx in ids:
                feature, _ = online_feature(backbone, processor, data["image_paths"][idx], data["instructions"][idx], args.device, args.pooling)
                if adapter_params and not feature.requires_grad:
                    raise RuntimeError("Online Qwen feature is detached from LoRA")
                xyz, opened = head(feature, torch.as_tensor(data["current"][idx:idx + 1], device=args.device))
                target_xyz = torch.as_tensor(data["paths"][idx:idx + 1], device=args.device)
                target_events = torch.as_tensor(data["events"][idx:idx + 1], device=args.device)
                prediction = torch.cat([xyz[:, :, 1:], opened[:, :, 1:, None] * args.event_scale], dim=-1)
                target = torch.cat([target_xyz[:, :, 1:], target_events[:, :, 1:, None] * args.event_scale], dim=-1)
                loss = positive_assignment_loss(prediction, target, data["path_mask"][idx:idx + 1], "saturation", rng)
                if not torch.isfinite(loss):
                    raise RuntimeError("non-finite route loss")
                (loss / config["effective_batch"]).backward()
                step_loss += float(loss.detach()) / config["effective_batch"]
                exposures += args.candidates
                requests += 1
            audit_gradients(backbone, audit)
            torch.nn.utils.clip_grad_norm_(trainable, 1.)
            optimizer.step()
            scheduler.step()
            losses.append(step_loss)
            if step <= 2:
                write_json(out / ("adapter_step%d.json" % step), finish_audit(backbone, audit))
            if step % 10 == 0:
                print(json.dumps(dict(step=step, loss=float(np.mean(losses[-10:])), online_train_requests=requests,
                    elapsed_s=elapsed_before + synchronized_time(args.device) - started)), flush=True)
            if step % args.eval_every == 0 or step == args.steps or step == args.stop_after:
                metrics = evaluate(backbone, processor, head, data, dev_ids, args)
                improved = metrics["selection_score"] > best
                best = max(best, metrics["selection_score"])
                elapsed = elapsed_before + synchronized_time(args.device) - started
                history.append(dict(step=step, loss=float(np.mean(losses[-10:])), dev_model=metrics))
                checkpoint = dict(head=head.state_dict(), adapters=adapter_state(backbone), optimizer=optimizer.state_dict(),
                    scheduler=scheduler.state_dict(), scaler=None, step=step, config=config, rng=rng_state(rng),
                    sampler_state=sampler.bit_generator.state, best=best, history=history, adapter_audit=audit,
                    elapsed_s=elapsed, setup_seconds_total=setup_total,
                    trajectory_exposures=exposures, online_train_requests=requests)
                atomic_checkpoint(out / "last.pt", checkpoint)
                if improved:
                    atomic_checkpoint(out / "best.pt", checkpoint)
                write_json(out / "history.json", history)
                write_json(out / "adapter_last.json", finish_audit(backbone, audit, require_update=step >= 2))
                print(json.dumps(history[-1]), flush=True)
                if step == args.stop_after and step < args.steps:
                    write_json(out / "status.json", dict(status="interrupted_for_resume_check", step=step, exit_code=0))
                    return
        last_audit = finish_audit(backbone, audit, require_update=args.steps >= 2)
        checkpoint = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
        head.load_state_dict(checkpoint["head"])
        load_adapters(backbone, checkpoint["adapters"])
        metrics = evaluate(backbone, processor, head, data, dev_ids, args, out / "dev_model")
        best_audit = finish_audit(backbone, checkpoint["adapter_audit"], require_update=checkpoint["step"] >= 2)
        write_json(out / "adapter_best.json", best_audit)
        elapsed = elapsed_before + synchronized_time(args.device) - started
        summary = dict(metrics=metrics, elapsed_s=elapsed, this_process_setup_s=setup_seconds, setup_seconds_total=setup_total,
            gpu_hours_reserved=(elapsed + setup_total) / 3600 if args.device == "cuda" else 0.,
            trajectory_exposures=exposures, online_train_requests=requests, head_parameters=head.active_parameter_count(),
            lora_parameters=config["lora_parameters"], peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0.,
            best_step=checkpoint["step"], best_checkpoint_sha256=sha256(out / "best.pt"),
            prediction_sha256=sha256(out / "dev_model/predictions.npz"), adapter_last=last_audit, adapter_best=best_audit,
            evidence_scope="online Qwen observational route pilot; no geometry validity, route-type, robot execution or new-mechanism claim")
        write_json(out / "summary.json", summary)
        write_json(out / "status.json", dict(status="completed", step=args.steps, exit_code=0))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(out / "status.json", dict(status="failed", exit_code=1, exception=repr(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


def self_test():
    """Small real Qwen modules, no downloaded weights or CUDA allocation."""
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    import tempfile
    torch.set_num_threads(1)
    seed_all(41)
    config = Qwen3VLConfig(text_config=dict(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        rope_scaling={"rope_type": "default", "mrope_section": [1, 1, 2], "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=4,
            out_hidden_size=32, deepstack_visual_indexes=[0, 1]),
        image_token_id=125, video_token_id=126, vision_start_token_id=124, vision_end_token_id=127)
    config._attn_implementation = "sdpa"
    backbone = Qwen3VLForConditionalGeneration(config).float().eval()
    replaced = install_lora(backbone, rank=2, alpha=4.)
    head = ObservedRouteHead(64, horizon=6, max_candidates=2, width=32, depth=1)
    audit = begin_audit(backbone)
    frozen_before = {name: tensor_hash(parameter) for name, parameter in backbone.named_parameters() if not parameter.requires_grad}
    optimizer = torch.optim.AdamW([{"params": head.parameters()}, {"params": list(adapters(backbone).values()), "weight_decay": 0.}], lr=1e-3)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    current = torch.zeros(1, 8)
    current[:, 7] = 1.
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        hidden = backbone.model(input_ids=ids, use_cache=False).last_hidden_state.float()
        feature = torch.cat([hidden.mean(1), hidden[:, -1]], dim=-1)
        if not feature.requires_grad:
            raise AssertionError("Qwen hidden state detached")
        xyz, opened = head(feature, current)
        loss = (xyz[:, :, 1:] - .2).square().mean() + (opened[:, :, 1:] - .7).square().mean()
        loss.backward()
        audit_gradients(backbone, audit)
        optimizer.step()
        losses.append(float(loss.detach()))
    result = finish_audit(backbone, audit, require_update=True)
    for name, record in result.items():
        if name.endswith("lora_A") and record["first_gradient_abs"] != 0:
            raise AssertionError("A initial gradient should be zero with B=0")
        if name.endswith("lora_B") and record["first_gradient_abs"] <= 0:
            raise AssertionError("B initial gradient must be nonzero")
    if frozen_before != {name: tensor_hash(parameter) for name, parameter in backbone.named_parameters() if not parameter.requires_grad}:
        raise AssertionError("Frozen base changed")
    saved = adapter_state(backbone)
    continuation_rng = np.random.default_rng(47)
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "adapter.pt"
        atomic_checkpoint(checkpoint, dict(head=head.state_dict(), adapters=saved, optimizer=optimizer.state_dict(),
            rng=rng_state(continuation_rng), step=2))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for parameter in adapters(backbone).values():
            with torch.no_grad():
                parameter.add_(1)
        load_adapters(backbone, state["adapters"])
        if any(not torch.equal(value, adapters(backbone)[name]) for name, value in saved.items()):
            raise AssertionError("Adapter checkpoint restore mismatch")
        def continuation():
            optimizer.zero_grad(set_to_none=True)
            random_ids = torch.as_tensor(continuation_rng.integers(1, 50, size=(1, 5)))
            hidden = backbone.model(input_ids=random_ids, use_cache=False).last_hidden_state.float()
            xyz, opened = head(torch.cat([hidden.mean(1), hidden[:, -1]], -1), current)
            target = torch.randn_like(xyz[:, :, 1:]) * .1
            loss = (xyz[:, :, 1:] - target).square().mean() + (opened[:, :, 1:] - .7).square().mean()
            loss.backward()
            optimizer.step()
            return {**{"head." + name: parameter.detach().clone() for name, parameter in head.named_parameters()},
                    **{name: parameter.detach().clone() for name, parameter in adapters(backbone).items()}}
        uninterrupted = continuation()
        head.load_state_dict(state["head"])
        load_adapters(backbone, state["adapters"])
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], continuation_rng)
        resumed = continuation()
        resume_difference = max(float((value - resumed[name]).abs().max()) for name, value in uninterrupted.items())
        if resume_difference != 0.:
            raise AssertionError("Optimizer/RNG resume differs from uninterrupted training")
        checkpoint_bytes = checkpoint.stat().st_size
    print(json.dumps(dict(status="passed", device="cpu", script_sha256=sha256(__file__), real_qwen_modules=replaced,
        losses=losses, frozen_base_unchanged=True, adapter_restore_exact=True,
        optimizer_rng_resume_max_parameter_difference=resume_difference,
        checkpoint_bytes=checkpoint_bytes, adapter_audit=result), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--observations")
    parser.add_argument("--supervision")
    parser.add_argument("--output")
    parser.add_argument("--head-init", help="same-data frozen-Qwen ordinary head checkpoint; verifies structure and provenance")
    parser.add_argument("--adapter-mode", choices=("lora", "frozen"), default="lora")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--candidates", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pooling", choices=("mean", "last", "both"), default="both")
    parser.add_argument("--event-scale", type=float, default=.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--max-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.self_test:
        if torch.cuda.is_initialized():
            parser.error("self-test must not run after CUDA initialization")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self_test()
        return
    if not all((args.model, args.observations, args.supervision, args.output)):
        parser.error("model, observations, supervision and output are required")
    if args.batch_size != 1 or min(args.accumulation, args.steps, args.eval_every, args.rank) < 1 or not 1 <= args.threads <= 4:
        parser.error("online microbatch must be 1; positive accumulation/steps/eval/rank; 1--4 CPU threads")
    if args.max_pixels < 64 * 32 * 32 or args.alpha <= 0:
        parser.error("max pixels must accommodate processor minimum, alpha must be positive")
    train(args)


if __name__ == "__main__":
    main()
