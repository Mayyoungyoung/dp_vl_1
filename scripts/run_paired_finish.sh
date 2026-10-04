#!/usr/bin/env bash
# Finite continuation of the already active replication, never a recurring job.
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/paired_modes_v1"
C="$R/coordinator_finish_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
PY="$P/.venv/bin/python"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
echo 'Waiting for the existing immutable three-arm replication to finish.'
while [[ ! -f "$R/coordinator_replication/closure.json" ]]; do sleep 10; done
"$PY" -c "import json; assert json.load(open('$R/coordinator_replication/closure.json'))['exit_code']==0"
for seed in 1 2; do
  bash scripts/run_paired_followup.sh score "$seed" R0 R2
done
job() { local id="$1"; shift; bash scripts/launch_paired_modes_gpu.sh --id "$id" -- "$@"; }
job public_cli_R2_seed0 "$PY" -m scripts.check_paired_cli --arm R2 --seed 0
job analyze_q_three_seeds "$PY" -m scripts.analyze_paired_reliability --root "$R" --seeds 0 1 2 --output "$R/reliability_three_seed_analysis"
bash scripts/run_paired_followup.sh full_control
if "$PY" -c "import json,sys; d=json.load(open('$R/full_control_seed0_analysis/RESULTS.json')); sys.exit(0 if d['full_control_seed0_gate']['passed'] else 1)"; then
  bash scripts/run_paired_followup.sh full_replicate
  echo 'Full-set control replicated. Inspect all scientific outcomes before concluding.'
else
  printf '%s\n' 'Full-set control gate did not pass. Stop here for mechanism diagnosis.'
  printf '%s\n' '{"status":"needs_mechanism_diagnosis","technical_failure":false}' > "$C/scientific_stop.json"
fi
