#!/usr/bin/env bash
# Continue only the unstarted full-set control after the preserved list-index test failure.
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/paired_modes_v1"
C="$R/coordinator_finish_recovery_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
PY="$P/.venv/bin/python"
"$PY" -c "import json; assert json.load(open('$R/coordinator_finish_v1/closure.json'))['exit_code']==1"
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
job cpu_tests_full_control_fixed "$PY" -m pytest -q tests/test_paired_modes_loss.py tests/test_paired_modes_evaluation.py
bash scripts/run_paired_diagnosis.sh
job full_control_probe "$PY" -m scripts.train_paired_modes --arm R_full --probe
job train_R_full_seed0 "$PY" -m scripts.train_paired_modes --arm R_full --seed 0
for split in paired_dev old_dev dev32; do
  job "eval_R_full_seed0_${split}" "$PY" -m scripts.evaluate_paired_modes --arm R_full --seed 0 --split "$split"
done
job analyze_full_seed0 "$PY" -m scripts.analyze_paired_modes --arms R0 R1 R_full R2 --output "$R/full_control_seed0_analysis"
if "$PY" -c "import json,sys; d=json.load(open('$R/full_control_seed0_analysis/RESULTS.json')); sys.exit(0 if d['full_control_seed0_gate']['passed'] else 1)"; then
  bash scripts/run_paired_followup.sh full_replicate
else
  printf '%s\n' '{"status":"needs_mechanism_diagnosis","technical_failure":false}' > "$C/scientific_stop.json"
fi
