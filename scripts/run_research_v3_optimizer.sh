#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/optimizer_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job optimizer_tests timeout 60 "$PY" -m pytest -q tests/test_segment_clearance.py tests/test_research_v3_frequency.py tests/test_research_v3_sampled.py
job optimizer_train timeout 360 "$PY" -m scripts.research_v3_frequency train --arm set_matching --name optimizer_restored --linear-control mean --restore-optimizer
job optimizer_evaluate timeout 60 "$PY" -m scripts.research_v3_frequency evaluate_fixed_q --name optimizer_restored --paired-score
job optimizer_train_audit timeout 60 "$PY" -m scripts.research_v3_posttrain_collision --output optimizer_restored_train_audit_v1 --checkpoint-name optimizer_restored
job optimizer_analysis timeout 60 "$PY" -m scripts.research_v3_analyze_optimizer
