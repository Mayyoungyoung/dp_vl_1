#!/usr/bin/env bash
# One predeclared mechanism revision, all three seeds; no result-driven sweep.
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/paired_modes_v1"
C="$R/coordinator_revision1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
while [[ ! -f "$R/coordinator_finish_recovery_v1/closure.json" ]]; do sleep 10; done
PY="$P/.venv/bin/python"
"$PY" -c "import json; assert json.load(open('$R/coordinator_finish_recovery_v1/closure.json'))['exit_code']==0"
{
  date -u
  nvidia-smi -i 1 --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
  ps -u "$(id -u)" -o pid,pcpu,pmem,comm --sort=-pcpu | sed -n '1,12p'
  uptime
  free -h
  df -h "$P"
} | tee "$C/resources_before_gpu.txt"
job() { local id="$1"; shift; bash scripts/launch_paired_modes_gpu.sh --id "$id" -- "$@"; }
job cpu_tests_revision1 "$PY" -m pytest -q tests/test_paired_modes_loss.py tests/test_paired_modes_evaluation.py
for seed in 0 1 2; do
  job "train_R3_seed${seed}" "$PY" -m scripts.train_paired_modes --arm R3 --seed "$seed"
  for split in paired_dev old_dev dev32; do
    job "eval_R3_seed${seed}_${split}" "$PY" -m scripts.evaluate_paired_modes --arm R3 --seed "$seed" --split "$split"
  done
done
job analyze_revision1 "$PY" -m scripts.analyze_paired_modes --arms R0 R1 R2 R3 --seeds 0 1 2 --output "$R/revision1_three_seed_analysis"
job plot_revision1 "$PY" -m scripts.plot_paired_results --arms R0 R1 R2 R3 --seeds 0 1 2 --output "$R/revision1_three_seed_figures"
