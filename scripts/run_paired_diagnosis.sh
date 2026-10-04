#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/paired_modes_v1"
C="$R/coordinator_train_diagnosis_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
while [[ ! -f "$R/coordinator_finish_v1/closure.json" ]]; do sleep 10; done
# Diagnostics require completed original models, regardless of a later control's technical failure.
{
  date -u
  nvidia-smi -i 1 --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
  ps -u "$(id -u)" -o pid,pcpu,pmem,comm --sort=-pcpu | sed -n '1,12p'
  uptime
  free -h
  df -h "$P"
} | tee "$C/resources_before_gpu.txt"
bash scripts/launch_paired_modes_gpu.sh --id train_correspondence_diagnosis_v1 -- \
  "$P/.venv/bin/python" -m scripts.diagnose_paired_training --output "$R/train_correspondence_diagnosis_v1"
