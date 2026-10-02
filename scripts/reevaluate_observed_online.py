"""Reevaluate fixed online-Qwen checkpoints on every DEV observation.

Protocol v2 includes instructions with no successful collected reference in
semantic evaluation. Reference errors keep their own denominator. This script
does not retrain or select checkpoints and never uses frozen feature caches.
"""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from routeset.common import sha256, write_json
from routeset.observed_route_head import ObservedRouteHead, QWEN_REVISION, OBSERVATION_EVAL_PROTOCOL
from scripts.train_observed_lora import (LoRALinear, adapters, evaluate, head_state_hash,
    install_lora, load_adapters, load_raw_observations, tensor_hash)


def remove_lora(backbone):
    """Restore the original frozen linear objects before each run's adapters."""
    for layer in backbone.model.language_model.layers:
        for name in ("q_proj", "v_proj"):
            module = getattr(layer.self_attn, name)
            if isinstance(module, LoRALinear):
                setattr(layer.self_attn, name, module.base)
    backbone.requires_grad_(False)
    if adapters(backbone):
        raise RuntimeError("Adapter reset left stale parameters")


def read_run(run):
    checkpoint_path = run / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    if config.get("model_revision") != QWEN_REVISION or config.get("processor_revision") != QWEN_REVISION:
        raise ValueError("Checkpoint requires pinned official model and processor")
    if config.get("hidden_cache_used") is not False or config.get("adapter_mode") not in ("frozen", "lora"):
        raise ValueError("Expected online frozen/LoRA checkpoint")
    data = load_raw_observations(config["observations"], config["supervision"], config["horizon"])
    original_hashes = json.loads((run / "source_hashes.json").read_text())
    for field in ("observations", "supervision"):
        if original_hashes.get(config[field]) != sha256(config[field]):
            raise ValueError("Training input/label manifest changed: " + field)
    # Every shared original file must still match. New no-reference inputs may
    # legitimately add image/current-state files to the v2 fingerprint.
    for filename, digest in original_hashes.items():
        if data["source_hashes"].get(filename) != digest:
            raise ValueError("Original training/evaluation source changed: " + filename)
    ids = np.flatnonzero(data["splits"] == "DEV_MODEL")
    train = np.flatnonzero((data["splits"] == "TRAIN") & data["path_mask"].any(axis=1))
    if not len(ids) or set(data["parent_ids"][train]) & set(data["parent_ids"][ids]):
        raise ValueError("Need nonempty, parent-disjoint DEV_MODEL")
    if any(data["semantic_targets"][idx] is None for idx in ids):
        raise ValueError("All DEV instructions require semantic targets for this evaluation")
    head = ObservedRouteHead(config["feature_dim"], config["horizon"], config["candidates"], config["width"], config["depth"])
    head.load_state_dict(checkpoint["head"], strict=True)
    if any(not torch.isfinite(value).all() for value in head.state_dict().values()):
        raise ValueError("Nonfinite checkpoint head")
    if bool(checkpoint["adapters"]) != (config["adapter_mode"] == "lora"):
        raise ValueError("Adapter mode and stored adapter parameters disagree")
    return checkpoint, config, data, ids, head


def original_reference_check(run, output):
    """Retain the old values; report agreement on the original reference rows."""
    path = run / "dev_model" / "per_scene.json"
    if not path.exists():
        return dict(available=False)
    old = {row["scene_id"]: row for row in json.loads(path.read_text())}
    new = {row["scene_id"]: row for row in json.loads((output / "per_scene.json").read_text())}
    differences = []
    for identifier, row in old.items():
        if identifier not in new:
            raise RuntimeError("Previously evaluated instruction disappeared")
        for key in ("candidate_matched_ADE_m", "candidate_endpoint_error_m"):
            if row.get(key) is not None:
                differences.append(abs(row[key] - new[identifier][key]))
    maximum = max(differences, default=0.)
    if maximum > 1e-5:
        raise RuntimeError("Fixed-checkpoint reference errors changed beyond 1e-5 m: %g" % maximum)
    return dict(available=True, original_rows=len(old), new_rows=len(new),
                max_reference_metric_absolute_difference_m=maximum,
                tolerance_m=1e-5, original_per_scene_sha256=sha256(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--validate-only", action="store_true", help="CPU: check manifests/checkpoint head; no Qwen load")
    args = parser.parse_args()
    if not 1 <= args.threads <= 4:
        parser.error("CPU threads must stay within 1--4")
    torch.set_num_threads(args.threads)
    backbone, loaded_model = None, None
    run_ids = [run.parent.name + "__" + run.name for run in args.runs]
    if len(run_ids) != len(set(run_ids)):
        parser.error("Run output identities must be unique")
    args.output.mkdir(parents=True, exist_ok=True)
    audit = []
    if not args.validate_only:
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        if transformers.__version__ != "4.57.1" or not torch.__version__.startswith("2.4.1"):
            raise ValueError("Use pinned .venv-qwen runtime")
        if args.device == "cuda":
            if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
                raise ValueError("Only authorized physical GPU1 may be visible")
            torch.cuda.set_per_process_memory_fraction(.35)
    source_files = [Path(__file__), Path("scripts/train_observed_lora.py"),
                    Path("scripts/train_observed_routes.py"), Path("routeset/observed_route_head.py")]
    source_hashes = {str(path): sha256(path) for path in source_files}
    for run, identifier in zip(args.runs, run_ids):
        started = time.perf_counter()
        checkpoint, config, data, ids, head = read_run(run)
        model_path = Path(config["model"])
        model_provenance = json.loads((model_path / "provenance.json").read_text())
        if (model_provenance.get("model_id") != "Qwen/Qwen3-VL-2B-Instruct"
                or model_provenance.get("revision") != QWEN_REVISION
                or not model_provenance.get("all_hashes_verified")
                or sha256(model_path / "provenance.json") != config["model_provenance_sha256"]):
            raise ValueError("Pinned official model provenance mismatch")
        provenance = dict(run_id=identifier, run=str(run.resolve()), checkpoint_sha256=sha256(run / "best.pt"),
            checkpoint_step=checkpoint["step"], original_training_config=config,
            original_config_file_sha256=sha256(run / "config.json"),
            original_training_source_commit=config.get("code_commit"),
            evaluation_source_commit=os.environ.get("CODE_COMMIT", "unrecorded"),
            evaluation_source_sha256=source_hashes, evaluation_protocol=OBSERVATION_EVAL_PROTOCOL,
            model_revision=QWEN_REVISION, processor_revision=QWEN_REVISION,
            model_provenance_sha256=sha256(model_path / "provenance.json"),
            evaluation_dataset_fingerprint=data["fingerprint"], input_manifest_sha256=sha256(config["observations"]),
            supervision_manifest_sha256=sha256(config["supervision"]),
            head_state_sha256=head_state_hash(head),
            adapter_sha256={name: tensor_hash(value) for name, value in checkpoint["adapters"].items()},
            checkpoint_selection="unchanged saved best.pt; no checkpoint reselection",
            examples=len(ids), reference_evaluation_examples=int(data["path_mask"][ids].any(axis=1).sum()),
            semantic_evaluation_examples=len(ids), parents=len(set(data["parent_ids"][ids])),
            training_reference_examples=int(((data["splits"] == "TRAIN") & data["path_mask"].any(axis=1)).sum()),
            unreferenced_observations=data["unreferenced"], threads=args.threads,
            device="cpu-validation" if args.validate_only else args.device,
            hidden_cache_used=False, gpu_uuid=os.environ.get("RESEARCH_GPU_UUID"))
        if args.validate_only:
            provenance["status"] = "CPU manifest and head validation passed; no Qwen forward"
            audit.append(provenance)
            print(json.dumps({key: provenance[key] for key in ("run_id", "examples", "reference_evaluation_examples", "training_reference_examples", "status")}), flush=True)
            continue
        output = args.output / identifier
        if (output / "metrics.json").exists():
            raise RuntimeError("Preserve existing reevaluation; choose a fresh output: " + str(output))
        if backbone is None:
            backbone = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
                torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
                attn_implementation="sdpa", device_map={"": args.device})
            loaded_model = str(model_path.resolve())
        if loaded_model != str(model_path.resolve()):
            raise ValueError("All runs must share the exact frozen model snapshot")
        remove_lora(backbone)
        if config["adapter_mode"] == "lora":
            install_lora(backbone, config["rank"], config["alpha"])
        load_adapters(backbone, checkpoint["adapters"])
        restored = {name: tensor_hash(value) for name, value in adapters(backbone).items()}
        if restored != provenance["adapter_sha256"]:
            raise RuntimeError("Restored adapter hashes differ from checkpoint")
        backbone.requires_grad_(False).eval()
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
            min_pixels=64 * 32 * 32, max_pixels=config["max_pixels"])
        head = head.to(args.device).eval()
        evaluation_args = argparse.Namespace(device=args.device, pooling=config["pooling"])
        metrics = evaluate(backbone, processor, head, data, ids, evaluation_args, output)
        provenance.update(prediction_sha256=sha256(output / "predictions.npz"),
            original_reference_agreement=original_reference_check(run, output),
            elapsed_s_including_shared_setup_if_first=time.perf_counter() - started,
            status="completed", fresh_qwen_requests=len(ids), adapter_reset_before_run=True)
        write_json(output / "provenance.json", provenance)
        write_json(output / "source_hashes.json", data["source_hashes"])
        print(json.dumps(dict(run_id=identifier, metrics=metrics, checkpoint_step=checkpoint["step"])), flush=True)
        del head, checkpoint
    if args.validate_only:
        write_json(args.output / "cpu_validation.json", audit)


if __name__ == "__main__":
    main()
