"""Registered conventional K2 training followed by frozen TRAIN-only analysis.

Run only after root authorizes this immutable source in the single-GPU queue.
Existing trainer is unchanged. Its four Torch workers are affinity-limited to
one allowed CPU; this is recorded, not silently called four-core throughput.
"""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from routeset.common import sha256, write_json
from scripts.audit_multigate_k2_closure import PROTOCOL


def validate_registration(registration, historical):
    if registration["protocol"] != PROTOCOL:
        raise ValueError("registered conventional K2 protocol required")
    train = registration["training"]
    if train["objective"] != "saturation" or train["candidates"] != 2 or historical["candidates"] != 4:
        raise ValueError("actual historical K4 and registered ordinary K2 required")
    for key in ("objective", "steps", "batch_size", "width", "depth", "lr", "seed", "eval_every", "device"):
        if train[key] != historical[key]:
            raise ValueError("changed historical ordinary training setting: " + key)
    if historical["dataset_sha256"] != registration["data_sha256"]:
        raise ValueError("different historical dataset")
    if train["steps"]*train["batch_size"]*2 != registration["gradient_target_slots"]:
        raise ValueError("wrong K2 path budget")
    if train["steps"]*train["batch_size"] != registration["training_example_exposures"]:
        raise ValueError("wrong example budget")
    return train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--registration", type=Path, required=True)
    args = parser.parse_args()
    source_root = Path(__file__).resolve().parents[1]
    if not re.fullmatch(r"[0-9a-f]{40}", args.source) or source_root.name != args.source:
        raise ValueError("run from the named immutable full-commit release")
    registration_path = args.registration.resolve()
    if source_root not in registration_path.parents:
        raise ValueError("registration must be inside the same immutable release")
    registration = json.loads(registration_path.read_text())
    historical_path = Path(registration["historical_k4_run"])/"config.json"
    historical = json.loads(historical_path.read_text())
    train = validate_registration(registration, historical)
    if sha256(registration["data"]) != registration["data_sha256"]:
        raise ValueError("registered dataset changed")
    output = Path(registration["output"])
    if output.exists():
        raise ValueError("fresh registered run required; use recorded trainer resume command for recovery")
    actual_uuid = subprocess.check_output(["nvidia-smi", "-i", "1", "--query-gpu=uuid", "--format=csv,noheader"],text=True).strip()
    if actual_uuid != registration["gpu"]["uuid"]:
        raise ValueError("authorized physical GPU1 UUID changed")
    allowed = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {allowed[0]})
    os.chdir(source_root)
    os.environ.update(CODE_COMMIT=args.source, CUDA_VISIBLE_DEVICES="1", RESEARCH_GPU_UUID=actual_uuid,
                      OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", PYTHONPATH=str(source_root))
    output.mkdir(parents=True)
    training_output = output/"seed0"
    command = [sys.executable, "-m", "routeset.train_v2", "--data", registration["data"], "--output", str(training_output)]
    for key, value in train.items():
        command.extend(["--"+key.replace("_", "-"), str(value)])
    analysis_command = [sys.executable, "scripts/audit_multigate_k2_closure.py", "--registration", str(registration_path),
                        "--run", str(training_output), "--output", str(output/"train_closure_audit")]
    write_json(output/"launch_receipt.json", dict(source_commit=args.source, registration=registration,
        registration_sha256=sha256(registration_path), historical_config_sha256=sha256(historical_path),
        actual_gpu_uuid=actual_uuid, gpu_memory_fraction_from_unchanged_trainer=.35,
        allowed_cpu_before=allowed, effective_cpu_affinity=sorted(os.sched_getaffinity(0)),
        trainer_requests_torch_threads=4, omp_num_threads=4, allowed_execution_cpus=1,
        affinity_enforcement="sched_setaffinity on launcher and explicit taskset on each recorder/child",
        train_command=command, audit_command=analysis_command, training_resume_command=command+["--resume"],
        source_sha256={name:sha256(source_root/name) for name in ("scripts/launch_multigate_k2_saturation.py", "scripts/audit_multigate_k2_closure.py",
            "scripts/audit_multigate_closure_opportunity.py", "routeset/train_v2.py", "routeset/models.py", "routeset/multigate.py", "routeset/common.py", "scripts/record_job.py")}))
    recorder = ["taskset", "-c", str(allowed[0]), sys.executable, "scripts/record_job.py", "--output", str(output/"receipts")]
    subprocess.run(recorder+["--run-id", "train_seed0", "--resume-strategy", "flag", "--"]+command, check=True)
    # The follow-up audit cannot initialize CUDA and only decodes TRAIN rows.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    subprocess.run(recorder+["--run-id", "train_closure_audit", "--resume-strategy", "fresh-output", "--"]+analysis_command, check=True)


if __name__ == "__main__":
    main()
