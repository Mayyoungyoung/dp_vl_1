#!/usr/bin/env bash
# Complete the strong within-scene baseline's actual-route scoring, serially.
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/paired_modes_v1"
C="$R/coordinator_score_compare"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
while [[ ! -f "$R/coordinator_revision1/closure.json" ]]; do sleep 10; done
PY="$P/.venv/bin/python"
"$PY" -c "import json; assert json.load(open('$R/coordinator_revision1/closure.json'))['exit_code']==0"
job() { local id="$1"; shift; bash scripts/launch_paired_modes_gpu.sh --id "$id" -- "$@"; }
job assess_revision1 "$PY" -m scripts.assess_paired_revision --source "$R/revision1_three_seed_analysis/RESULTS.json" --output "$R/revision1_gate.json"
for seed in 0 1 2; do bash scripts/run_paired_followup.sh score "$seed" R1; done
job public_cli_R1_seed0 "$PY" -m scripts.check_paired_cli --arm R1 --seed 0
arms=(R0 R1 R2)
if "$PY" -c "import json,sys; sys.exit(0 if json.load(open('$R/revision1_gate.json'))['passed'] else 1)"; then
  for seed in 0 1 2; do bash scripts/run_paired_followup.sh score "$seed" R3; done
  job public_cli_R3_seed0 "$PY" -m scripts.check_paired_cli --arm R3 --seed 0
  arms+=(R3)
fi
job analyze_q_complete "$PY" -m scripts.analyze_paired_reliability --root "$R" --arms "${arms[@]}" --seeds 0 1 2 --output "$R/reliability_complete_analysis"
