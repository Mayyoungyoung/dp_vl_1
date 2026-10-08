#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/linear_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
"$PY" -c "import json; assert json.load(open('$P/runs/research_v3_v1/linear_gradient_v1/RESULTS.json'))['ordinary_linear_gate']"
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job linear_existing_tests "$PY" -m pytest -q tests/test_segment_clearance.py tests/test_research_v3_frequency.py tests/test_research_v3_sampled.py
job linear_parent_evaluate timeout 60 "$PY" -m scripts.research_v3_frequency evaluate_fixed_q --name safety_mean --paired-score
job linear_parent_replay "$PY" -m scripts.research_v3_analyze_linear --replay-only
for arm in mean linear; do
  job "linear_train_$arm" timeout 360 "$PY" -m scripts.research_v3_frequency train --arm set_matching --name "margin_$arm" --linear-control "$arm"
  job "linear_evaluate_$arm" timeout 60 "$PY" -m scripts.research_v3_frequency evaluate_fixed_q --name "margin_$arm" --paired-score
done
job linear_analysis "$PY" -m scripts.research_v3_analyze_linear --output linear_analysis_v1
