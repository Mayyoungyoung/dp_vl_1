#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/matched_q_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
for role in SCORE_TRAIN DEV_SCORE CALIBRATION paired_dev; do
  job "matched_q_pool_$role" "$PY" -m scripts.research_v3_matched_q pool --role "$role"
done
job matched_q_roles "$PY" -m scripts.research_v3_matched_q roles
for seed in 0 1 2; do
  job "matched_q_fit_$seed" "$PY" -m scripts.research_v3_matched_q fit --seed "$seed"
  job "matched_q_calibrate_$seed" "$PY" -m scripts.research_v3_matched_q calibrate --seed "$seed"
done
job matched_q_analysis "$PY" -m scripts.research_v3_matched_q analyze
