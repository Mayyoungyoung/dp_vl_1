"""Same-pretrained-head online Qwen RGB-D frozen/LoRA comparison.

Only raw RGB/instruction enters Qwen; only current RGB-D/camera enters geometry.
No hidden-feature cache is accepted, including after an adapter update. Raw
unlearned RGBXYZ grids may be preprocessed once for training; evaluation reads
the raw RGB-D again for each request. All geometry/head parameters keep training.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import torch

from routeset.common import seed_all, sha256, write_json
from routeset.observed_geometry import ObservedGeometryRouteHead, positive_endpoint_attention_loss
from routeset.observed_route_head import QWEN_REVISION, OBSERVATION_EVAL_PROTOCOL
from routeset.train_v2 import atomic_checkpoint, positive_assignment_loss, restore_rng, rng_state, synchronized_time
from scripts.train_observed_lora import (install_lora, adapters, adapter_state, load_adapters,
    tensor_hash, head_state_hash, begin_audit, audit_gradients, finish_audit,
    load_raw_observations, online_feature)
from scripts.train_observed_geometry import load_geometry, read_geometry, POINT_FIELDS
from scripts.train_observed_routes import observation_metrics, paired_language_indices

PROJECT = Path(__file__).resolve().parents[1]
DEPENDENCIES = [PROJECT/name for name in (
    'scripts/train_observed_lora.py', 'scripts/train_observed_geometry.py',
    'scripts/train_observed_routes.py', 'routeset/observed_geometry.py',
    'routeset/observed_route_head.py', 'routeset/train_v2.py')]


def geometry_inputs(feature, data, geometry, idx, device, reread=False, pixel_stride=2):
    raw = read_geometry(*geometry['sources'][geometry['index'][idx]], pixel_stride) if reread else {
        name:geometry['points'][name][geometry['index'][idx]] for name in POINT_FIELDS}
    inputs = {name:torch.as_tensor(value[None], device=device) for name,value in raw.items()}
    inputs.update(features=feature, current=torch.as_tensor(data['current'][idx:idx+1], device=device))
    return inputs


def load_geometry_initialization(head, path, expected, observations, supervision):
    path = Path(path)
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    source = checkpoint.get('config', {})
    for key in ('feature_dim', 'horizon', 'candidates', 'width', 'depth', 'pooling', 'event_scale',
                'point_width', 'pixel_stride', 'endpoint_residual_bound', 'grounding_weight', 'grounding_sigma'):
        if source.get(key) != expected[key]:
            raise ValueError('RGB-D initialization architecture/objective mismatch: '+key)
    cache = source.get('cache_config', {})
    if (cache.get('model') != 'Qwen/Qwen3-VL-2B-Instruct' or cache.get('revision') != QWEN_REVISION
            or cache.get('processor') != QWEN_REVISION or cache.get('model_trainable_parameter_count') != 0
            or cache.get('max_pixels') != expected['max_pixels']):
        raise ValueError('Common initialization requires matching genuine frozen Qwen provenance')
    if cache.get('manifest_sha256') != sha256(observations):
        raise ValueError('Common initialization uses a different observation manifest')
    source_file = path.parent/'source_hashes.json'
    sources = json.loads(source_file.read_text())
    if sources.get(source.get('observations')) != sha256(observations) or sources.get(source.get('supervision')) != sha256(supervision):
        raise ValueError('Common initialization observation/supervision content differs')
    state = checkpoint.get('model')
    expected_state = head.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ValueError('Common initialization must contain the complete ordinary RGB-D head')
    if any(state[name].shape != value.shape or not torch.isfinite(state[name]).all() for name,value in expected_state.items()):
        raise ValueError('Common initialization invalid parameter shape or finite state')
    head.load_state_dict(state, strict=True)
    if not all(parameter.requires_grad for parameter in head.parameters()):
        raise ValueError('Every geometry and head parameter must continue training')
    summary_path = path.parent/'summary.json'
    summary = json.loads(summary_path.read_text())
    common = dict(run=str(path.parent), selected_step=checkpoint['step'], training_steps=source['steps'],
                  total_trajectory_exposures=summary['trajectory_exposures'],
                  selected_trajectory_exposures=checkpoint['trajectory_exposures'],
                  elapsed_s=summary['elapsed_s'], gpu_hours_reserved=summary['gpu_hours_reserved'],
                  summary_sha256=sha256(summary_path), accounting='shared pretraining listed separately; additional online exposure is not total exposure')
    return dict(path=str(path.resolve()), sha256=sha256(path), source_config=source,
                source_hashes_sha256=sha256(source_file), source_step=checkpoint['step'],
                head_state_sha256=head_state_hash(head), common_pretraining=common,
                same_observation_and_supervision_hashes=True,
                initialization_scope='all geometry and head weights; reset optimizer and zero-B new adapters for the paired arm')


@torch.no_grad()
def evaluate(backbone, processor, head, data, geometry, ids, args, output=None):
    backbone.eval(); head.eval()
    predictions, events, anchors, timings = [], [], [], []
    for idx in ids:
        started = synchronized_time(args.device)
        feature, _ = online_feature(backbone, processor, data['image_paths'][idx], data['instructions'][idx], args.device, args.pooling)
        inputs = geometry_inputs(feature, data, geometry, int(idx), args.device, reread=True, pixel_stride=args.pixel_stride)
        xyz, opened, details = head(**inputs)
        predictions.append(xyz.cpu().numpy()); events.append(opened.cpu().numpy())
        anchors.append(details['anchor_xyz'].cpu().numpy())
        timings.append((synchronized_time(args.device)-started)*1000)
    predictions, events, anchors = map(np.concatenate, (predictions, events, anchors))
    metrics, rows = observation_metrics(predictions, events, data, ids)
    anchor_errors = [float(np.linalg.norm(anchors[position]-data['paths'][idx, data['path_mask'][idx], -1], axis=-1).min())
                     if data['path_mask'][idx].any() else None for position,idx in enumerate(ids)]
    finite = [value for value in anchor_errors if value is not None]
    metrics['learned_surface_anchor_reference_endpoint_error_m'] = float(np.mean(finite)) if finite else None
    for row,error in zip(rows,anchor_errors):
        row['learned_surface_anchor_reference_endpoint_error_m'] = error
    switched = paired_language_indices(data, ids)
    positions = {int(idx):position for position,idx in enumerate(ids)}
    eligible = np.flatnonzero(switched >= 0)
    if len(eligible):
        swapped = np.asarray([positions[int(switched[position])] for position in eligible])
        wrong, _ = observation_metrics(predictions[swapped], events[swapped], data, ids[eligible])
        own, _ = observation_metrics(predictions[swapped], events[swapped], data, switched[eligible])
        metrics['paired_language_control'] = dict(examples=len(eligible),
            same_image_other_target_language_original_goal_accuracy=wrong['semantic_goal_accuracy'],
            same_image_other_target_language_new_goal_accuracy=own['semantic_goal_accuracy'],
            mean_endpoint_response_m=float(np.linalg.norm(predictions[swapped, :, -1]-predictions[eligible, :, -1], axis=-1).mean()),
            explanation='Separate online predictions with current adapters; identical image/current RGB-D, changed instruction')
    else:
        metrics['paired_language_control'] = None
    measured = timings[2:] if len(timings)>2 else timings
    metrics.update(selection_score=-metrics['candidate_matched_ADE_m'],
                   online_request_ms_median=float(np.median(measured)), online_request_ms_p95=float(np.percentile(measured,95)),
                   first_request_ms=timings[0], timed_requests=len(measured),
                   latency_scope='RGB file/processor/Qwen plus raw RGB-D read/backprojection/trainable geometry/head/output; no feature cache; excludes model loading, checks, scoring, execution')
    if output is not None:
        output=Path(output); output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'predictions.npz', paths=predictions, gripper_open=events, learned_surface_anchor=anchors,
                            scene_ids=data['scene_ids'][ids], parent_ids=data['parent_ids'][ids])
        write_json(output/'metrics.json', metrics); write_json(output/'per_scene.json', rows)
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
    load_started = synchronized_time(args.device)
    data = load_raw_observations(args.observations, args.supervision, args.horizon)
    geometry = load_geometry(data, args.observations, args.supervision, args.pixel_stride)
    data_load_preprocess_s = synchronized_time(args.device)-load_started
    train_ids = np.flatnonzero((data["splits"] == "TRAIN") & data["path_mask"].any(axis=1))
    dev_ids = np.flatnonzero(data["splits"] == "DEV_MODEL")
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
        head = ObservedGeometryRouteHead(feature_dim, args.horizon, args.candidates, args.width, args.depth,
                                         args.point_width, args.endpoint_residual_bound).to(args.device)
        head_init = None
        if not args.head_init:
            raise ValueError("A common pretrained RGB-D head checkpoint is required")
        if args.head_init:
            head_init = load_geometry_initialization(head, args.head_init, dict(vars(args), feature_dim=feature_dim),
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
        config = dict(vars(args), feature_dim=feature_dim, dataset_fingerprint=geometry["fingerprint"],
            head_initialization=head_init, head_init_sha256=None if head_init is None else head_init["sha256"],
            initial_head_state_sha256=head_state_hash(head),
            model_revision=QWEN_REVISION, processor_revision=QWEN_REVISION,
            model_provenance_sha256=sha256(model_path / "provenance.json"), torch=torch.__version__, transformers=transformers.__version__,
            code_commit=os.environ.get("CODE_COMMIT", "unrecorded"), gpu_uuid=os.environ.get("RESEARCH_GPU_UUID"),
            lora_modules=modules, adapter_initialization_seed=args.seed + 200000,
            lora_parameters=sum(p.numel() for p in adapter_params), head_parameters=head.active_parameter_count(),
            input_contract=["RGB", "instruction", "current gripper pose7", "current gripper open1", "current metric depth and camera calibration"],
            geometry_preprocessing=geometry["metadata"], geometry_parameters=head.geometry.active_parameter_count(),
            source_script_sha256=sha256(__file__), source_dependency_sha256={str(path.relative_to(PROJECT)):sha256(path) for path in DEPENDENCIES},
            hidden_cache_used=False, gradient_explanation="B initialized zero: A first-step gradient is zero; B must receive gradient, A can update after B changes; adapter weight decay is zero",
            objective="saturation xyz/event assignment plus identical endpoint-derived attention supervision in frozen and LoRA arms",
            effective_batch=args.batch_size * args.accumulation, selection_split="DEV_MODEL", selection_metric="negative candidate_matched_ADE_m",
            train_examples=len(train_ids), dev_examples=len(dev_ids), train_parents=len(set(data["parent_ids"][train_ids])),
            dev_parents=len(set(data["parent_ids"][dev_ids])), skipped_supervision=data["skipped"],
            unreferenced_observations=data["unreferenced"], evaluation_protocol=OBSERVATION_EVAL_PROTOCOL)
        rng, sampler = np.random.default_rng(args.seed), np.random.default_rng(args.seed + 100000)
        first, best, elapsed_before, setup_before, exposures, requests, history = 0, -float("inf"), 0., 0., 0, 0, []
        if args.resume:
            checkpoint = torch.load(out / "last.pt", map_location=args.device, weights_only=False)
            for key in ("dataset_fingerprint", "model_provenance_sha256", "feature_dim", "adapter_mode", "rank", "alpha", "horizon", "candidates", "width", "depth", "steps", "seed", "lr", "lora_lr", "batch_size", "accumulation", "pooling", "event_scale", "max_pixels", "head_init_sha256", "pixel_stride", "point_width", "endpoint_residual_bound", "grounding_weight", "grounding_sigma", "source_script_sha256", "source_dependency_sha256"):
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
        setup_seconds = synchronized_time(args.device) - setup_start + data_load_preprocess_s
        setup_total = setup_before + setup_seconds
        write_json(out / "config.json", config)
        if head_init:
            write_json(out / "head_initialization.json", head_init)
        write_json(out / "source_hashes.json", dict(data["source_hashes"], **geometry["metadata"]["source_hashes"]))
        write_json(out / "status.json", dict(status="running", run_id=out.name, pid=os.getpid(), step=first, resume_command="rerun the exact command with --resume"))
        started, losses, path_losses, grounding_losses = synchronized_time(args.device), [], [], []
        if args.head_init and not args.resume:
            # Include the common pretrained starting point in DEV selection;
            # a harmful fine-tune must not silently replace its stronger head.
            metrics = evaluate(backbone, processor, head, data, geometry, dev_ids, args)
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
            step_loss, step_path_loss, step_grounding_loss = 0., 0., 0.
            for idx in ids:
                feature, _ = online_feature(backbone, processor, data["image_paths"][idx], data["instructions"][idx], args.device, args.pooling)
                if adapter_params and not feature.requires_grad:
                    raise RuntimeError("Online Qwen feature is detached from LoRA")
                inputs = geometry_inputs(feature, data, geometry, int(idx), args.device)
                xyz, opened, details = head(**inputs)
                target_xyz = torch.as_tensor(data["paths"][idx:idx + 1], device=args.device)
                target_events = torch.as_tensor(data["events"][idx:idx + 1], device=args.device)
                prediction = torch.cat([xyz[:, :, 1:], opened[:, :, 1:, None] * args.event_scale], dim=-1)
                target = torch.cat([target_xyz[:, :, 1:], target_events[:, :, 1:, None] * args.event_scale], dim=-1)
                path_loss = positive_assignment_loss(prediction, target, data["path_mask"][idx:idx + 1], "saturation", rng)
                grounding_loss = positive_endpoint_attention_loss(details["attention"], inputs["world_xyz"],
                    inputs["valid_mask"], target_xyz[:, :, -1],
                    torch.as_tensor(data["path_mask"][idx:idx + 1], device=args.device), args.grounding_sigma)
                loss = path_loss + args.grounding_weight*grounding_loss
                step_path_loss += float(path_loss.detach()) / config["effective_batch"]
                step_grounding_loss += float(grounding_loss.detach()) / config["effective_batch"]
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
            path_losses.append(step_path_loss)
            grounding_losses.append(step_grounding_loss)
            if step <= 2:
                write_json(out / ("adapter_step%d.json" % step), finish_audit(backbone, audit))
            if step % 10 == 0:
                print(json.dumps(dict(step=step, loss=float(np.mean(losses[-10:])), path_loss=float(np.mean(path_losses[-10:])),
                    grounding_loss=float(np.mean(grounding_losses[-10:])), online_train_requests=requests,
                    elapsed_s=elapsed_before + synchronized_time(args.device) - started)), flush=True)
            if step % args.eval_every == 0 or step == args.steps or step == args.stop_after:
                metrics = evaluate(backbone, processor, head, data, geometry, dev_ids, args)
                improved = metrics["selection_score"] > best
                best = max(best, metrics["selection_score"])
                elapsed = elapsed_before + synchronized_time(args.device) - started
                history.append(dict(step=step, loss=float(np.mean(losses[-10:])), path_loss=float(np.mean(path_losses[-10:])),
                                    grounding_loss=float(np.mean(grounding_losses[-10:])), dev_model=metrics))
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
        last_head_hash = head_state_hash(head)
        last_metrics = evaluate(backbone, processor, head, data, geometry, dev_ids, args, out / "last_dev_model")
        checkpoint = torch.load(out / "best.pt", map_location=args.device, weights_only=False)
        head.load_state_dict(checkpoint["head"])
        load_adapters(backbone, checkpoint["adapters"])
        metrics = evaluate(backbone, processor, head, data, geometry, dev_ids, args, out / "dev_model")
        best_audit = finish_audit(backbone, checkpoint["adapter_audit"], require_update=checkpoint["step"] >= 2)
        write_json(out / "adapter_best.json", best_audit)
        elapsed = elapsed_before + synchronized_time(args.device) - started
        summary = dict(metrics=metrics, last_metrics=last_metrics, elapsed_s=elapsed, this_process_setup_s=setup_seconds, setup_seconds_total=setup_total,
            data_load_preprocess_s=data_load_preprocess_s, geometry_preprocessing=geometry["metadata"],
            common_pretraining=head_init["common_pretraining"],
            initial_head_state_sha256=config["initial_head_state_sha256"], last_head_state_sha256=last_head_hash,
            best_head_state_sha256=head_state_hash(head), head_changed_at_last=last_head_hash != config["initial_head_state_sha256"],
            gpu_hours_reserved=(elapsed + setup_total) / 3600 if args.device == "cuda" else 0.,
            trajectory_exposures=exposures, online_train_requests=requests, head_parameters=head.active_parameter_count(),
            lora_parameters=config["lora_parameters"], peak_cuda_memory_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else 0.,
            best_step=checkpoint["step"], best_checkpoint_sha256=sha256(out / "best.pt"),
            prediction_sha256=sha256(out / "dev_model/predictions.npz"), adapter_last=last_audit, adapter_best=best_audit,
            evidence_scope="same-initialization online Qwen RGB-D frozen/LoRA baseline improvement; no path validity, route-type, robot execution or core-mechanism claim")
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
    initial_tokens = torch.tensor([[1, 2, 3, 4, 5]])
    with torch.no_grad():
        frozen_initial = backbone.model(input_ids=initial_tokens, use_cache=False).last_hidden_state.clone()
    replaced = install_lora(backbone, rank=2, alpha=4.)
    with torch.no_grad():
        assert torch.equal(frozen_initial, backbone.model(input_ids=initial_tokens, use_cache=False).last_hidden_state)
    head = ObservedGeometryRouteHead(64, horizon=6, max_candidates=2, width=32, depth=1, point_width=16)
    audit = begin_audit(backbone)
    frozen_before = {name: tensor_hash(parameter) for name, parameter in backbone.named_parameters() if not parameter.requires_grad}
    optimizer = torch.optim.AdamW([{"params": head.parameters()}, {"params": list(adapters(backbone).values()), "weight_decay": 0.}], lr=1e-3)
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    current = torch.zeros(1, 8)
    current[:, 7] = 1.
    points = dict(world_xyz=torch.rand(1, 16, 3)*.2, rgb=torch.rand(1,16,3),
                  uv=torch.rand(1,16,2), depth=torch.ones(1,16), valid_mask=torch.ones(1,16,dtype=torch.bool))
    initial_head = {name:tensor_hash(parameter) for name,parameter in head.named_parameters()}
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _:1.)
    sampler = np.random.default_rng(411)
    losses = []
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        hidden = backbone.model(input_ids=ids, use_cache=False).last_hidden_state.float()
        feature = torch.cat([hidden.mean(1), hidden[:, -1]], dim=-1)
        if not feature.requires_grad:
            raise AssertionError("Qwen hidden state detached")
        xyz, opened, details = head(features=feature, current=current, **points)
        loss = (xyz[:, :, 1:] - .2).square().mean() + (opened[:, :, 1:] - .7).square().mean()
        loss = loss + .02*positive_endpoint_attention_loss(details["attention"], points["world_xyz"], points["valid_mask"], points["world_xyz"][:, :1], torch.ones(1,1,dtype=torch.bool))
        loss.backward()
        audit_gradients(backbone, audit)
        optimizer.step(); scheduler.step()
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
            rng=rng_state(continuation_rng), scheduler=scheduler.state_dict(), sampler=sampler.bit_generator.state, step=2))
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        for parameter in adapters(backbone).values():
            with torch.no_grad():
                parameter.add_(1)
        load_adapters(backbone, state["adapters"])
        if any(not torch.equal(value, adapters(backbone)[name]) for name, value in saved.items()):
            raise AssertionError("Adapter checkpoint restore mismatch")
        def continuation():
            optimizer.zero_grad(set_to_none=True)
            random_ids = torch.as_tensor(continuation_rng.integers(1, 40, size=(1, 5))+sampler.choice(np.arange(9),1)[0])
            hidden = backbone.model(input_ids=random_ids, use_cache=False).last_hidden_state.float()
            xyz, opened, details = head(features=torch.cat([hidden.mean(1), hidden[:, -1]], -1), current=current, **points)
            target = torch.randn_like(xyz[:, :, 1:]) * .1
            loss = (xyz[:, :, 1:] - target).square().mean() + (opened[:, :, 1:] - .7).square().mean()
            loss = loss + .02*positive_endpoint_attention_loss(details["attention"], points["world_xyz"], points["valid_mask"], points["world_xyz"][:, :1], torch.ones(1,1,dtype=torch.bool))
            loss.backward()
            optimizer.step(); scheduler.step()
            return {**{"head." + name: parameter.detach().clone() for name, parameter in head.named_parameters()},
                    **{name: parameter.detach().clone() for name, parameter in adapters(backbone).items()}}
        uninterrupted = continuation()
        uninterrupted_sampler = sampler.bit_generator.state
        uninterrupted_scheduler = scheduler.state_dict()
        head.load_state_dict(state["head"])
        load_adapters(backbone, state["adapters"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        sampler.bit_generator.state = state["sampler"]
        restore_rng(state["rng"], continuation_rng)
        resumed = continuation()
        assert uninterrupted_sampler == sampler.bit_generator.state
        assert uninterrupted_scheduler == scheduler.state_dict()
        resume_difference = max(float((value - resumed[name]).abs().max()) for name, value in uninterrupted.items())
        if resume_difference != 0.:
            raise AssertionError("Optimizer/RNG resume differs from uninterrupted training")
        checkpoint_bytes = checkpoint.stat().st_size
    print(json.dumps(dict(status="passed", device="cpu", script_sha256=sha256(__file__), real_qwen_modules=replaced,
        losses=losses, frozen_base_unchanged=True, zero_B_matches_frozen_initial=True,
        adapter_restore_exact=True, sampler_and_scheduler_resume_exact=True,
        all_geometry_and_head_trainable=all(p.requires_grad for p in head.parameters()),
        geometry_and_head_updated_tensors=sum(initial_head[n]!=tensor_hash(p) for n,p in head.named_parameters()),
        geometry_and_head_total_tensors=len(initial_head),
        optimizer_rng_resume_max_parameter_difference=resume_difference,
        checkpoint_bytes=checkpoint_bytes, adapter_audit=result), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--observations")
    parser.add_argument("--supervision")
    parser.add_argument("--output")
    parser.add_argument("--head-init", help="required same-data pretrained RGB-D auxiliary best checkpoint; validates architecture, data and provenance")
    parser.add_argument("--adapter-mode", choices=("lora", "frozen"), default="lora")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation", type=int, default=4)
    parser.add_argument("--candidates", type=int, choices=(1, 2, 4), default=4)
    parser.add_argument("--horizon", type=int, default=24)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--pooling", choices=("mean", "last", "both"), default="both")
    parser.add_argument('--point-width', type=int, default=64)
    parser.add_argument('--pixel-stride', type=int, default=2)
    parser.add_argument('--endpoint-residual-bound', type=float, default=.05)
    parser.add_argument('--grounding-weight', type=float, default=.02)
    parser.add_argument('--grounding-sigma', type=float, default=.025)
    parser.add_argument("--event-scale", type=float, default=.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-pixels", type=int, default=256 * 32 * 32)
    parser.add_argument("--threads", type=int, default=1)
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
    if not all((args.model, args.observations, args.supervision, args.output, args.head_init)):
        parser.error("model, observations, supervision, output and common RGB-D head-init are required")
    if args.batch_size != 1 or min(args.accumulation, args.steps, args.eval_every, args.rank) < 1 or not 1 <= args.threads <= 4:
        parser.error("online microbatch must be 1; positive accumulation/steps/eval/rank; 1--4 CPU threads")
    if args.max_pixels < 64 * 32 * 32 or args.alpha <= 0:
        parser.error("max pixels must accommodate processor minimum, alpha must be positive")
    if args.pixel_stride < 1 or args.point_width < 1 or args.grounding_weight < 0 or args.grounding_sigma <= 0 or args.endpoint_residual_bound <= 0:
        parser.error('invalid point geometry or grounding supervision configuration')
    train(args)


if __name__ == "__main__":
    main()
