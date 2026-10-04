#!/usr/bin/env bash
# Immutable first comparison. Any failed stage stops before subsequent stages.
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/paired_modes_v1"
C="$R/coordinator_seed0_v2"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
export CODE_COMMIT="$(basename "$PWD")"
printf '%s\n' "$CODE_COMMIT" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
echo 'Waiting for the already-running frozen collector to close all scenes.'
while [[ ! -f "$R/collection_v2_006_480/summary.json" ]]; do sleep 10; done
taskset -c 0-3 "$P/.venv/bin/python" -m scripts.summarize_paired_data
taskset -c 0-3 "$P/.venv/bin/python" -m scripts.paired_modes_data export
{
  date -u
  nvidia-smi -i 1 --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv
  uptime
  free -h
  df -h "$P"
} | tee "$C/resources_before_gpu.txt"
job() { local identifier="$1"; shift; bash scripts/launch_paired_modes_gpu.sh --id "$identifier" -- "$@"; }
PY="$P/.venv/bin/python"
D="$P/data/paired_modes_v1_v2/export"
job qwen_paired "$P/.venv-qwen/bin/python" -m scripts.observation_cache_qwen \
  --model "$P/data/qwen3-vl-2b-instruct-89644892" --manifest "$D/observations.jsonl" \
  --output "$D/qwen_cache" --device cuda --threads 4 --gpu-memory-fraction .35
job train_loss_probe "$PY" -m scripts.train_paired_modes --probe
job profile_continuous "$PY" -m scripts.train_paired_modes --arm R2 --steps 100 --run-name profile_R2_continuous
job profile_interrupted "$PY" -m scripts.train_paired_modes --arm R2 --steps 100 --stop-after 50 --run-name profile_R2_resumed
job profile_resumed "$PY" -m scripts.train_paired_modes --arm R2 --steps 100 --run-name profile_R2_resumed --resume
job verify_resume "$PY" -m scripts.verify_paired_resume
for arm in R0 R1 R2; do
  job "train_${arm}_seed0" "$PY" -m scripts.train_paired_modes --arm "$arm" --seed 0
done
job eval_B_paired "$PY" -m scripts.evaluate_paired_modes --arm B --seed 0 --split paired_dev
for arm in R0 R1 R2; do
  for split in paired_dev old_dev dev32; do
    job "eval_${arm}_seed0_${split}" "$PY" -m scripts.evaluate_paired_modes --arm "$arm" --seed 0 --split "$split"
  done
done
job analyze_seed0 "$PY" -m scripts.analyze_paired_modes --output "$R/seed0_analysis"
job plot_seed0 "$PY" -m scripts.plot_paired_results --output "$R/seed0_figures"
echo 'Seed0 comparison finished. Continuation requires its actual scientific results.'
