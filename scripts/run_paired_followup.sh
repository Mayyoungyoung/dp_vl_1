#!/usr/bin/env bash
# Invoke only after reading the completed seed0 comparison. Fresh outputs only.
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/paired_modes_v1"
stage="$1"; shift
case "$stage" in
  replicate) label=replication ;;
  score) seed="$1"; shift; label="score_seed${seed}_$(IFS=_; echo "$*")" ;;
  *) echo 'Expected replicate, or score SEED ARM...' >&2; exit 2 ;;
esac
C="$R/coordinator_$label"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
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
PY="$P/.venv/bin/python"
if [[ "$stage" == replicate ]]; then
  for seed in 1 2; do
    for arm in R0 R1 R2; do
      job "train_${arm}_seed${seed}" "$PY" -m scripts.train_paired_modes --arm "$arm" --seed "$seed"
      for split in paired_dev old_dev dev32; do
        job "eval_${arm}_seed${seed}_${split}" "$PY" -m scripts.evaluate_paired_modes --arm "$arm" --seed "$seed" --split "$split"
      done
    done
  done
  job analyze_three_seeds "$PY" -m scripts.analyze_paired_modes --seeds 0 1 2 --output "$R/three_seed_analysis"
  job plot_three_seeds "$PY" -m scripts.plot_paired_results --seeds 0 1 2 --output "$R/three_seed_figures"
else
  for arm in "$@"; do
    for role in SCORE_TRAIN DEV_SCORE CALIBRATION; do
      job "eval_${arm}_seed${seed}_${role}" "$PY" -m scripts.evaluate_paired_modes --arm "$arm" --seed "$seed" --split "$role"
    done
    for action in fit calibrate package; do
      job "q_${action}_${arm}_seed${seed}" "$PY" -m scripts.paired_modes_reliability "$action" --arm "$arm" --generator-seed "$seed" --seed "$seed"
    done
    job "q_plot_${arm}_seed${seed}" "$PY" -m scripts.plot_paired_results \
      --reliability-folder "$R/reliability/${arm}_seed${seed}/calibration_seed${seed}" \
      --output "$R/reliability/${arm}_seed${seed}/figures_seed${seed}"
  done
fi
