#!/usr/bin/env bash
set -euo pipefail
LAUNCHER="$(realpath "$0")"
cd "$(dirname "$LAUNCHER")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/all_mode_completion_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$LAUNCHER" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job all_mode_completion_tests timeout 10 "$PY" -m pytest -q tests/test_research_v3_all_mode_support.py tests/test_research_v3_frequency.py tests/test_segment_clearance.py
job all_mode_completion_prepare timeout 30 "$PY" -m scripts.research_v3_prepare_edit_support --all-modes
job all_mode_completion_train timeout 300 "$PY" -m scripts.research_v3_frequency train --arm set_matching --name verified_edit_all_modes --linear-control mean --all-mode-edit-support
job all_mode_completion_evaluate timeout 20 "$PY" -m scripts.research_v3_frequency evaluate_fixed_q --name verified_edit_all_modes --paired-score
job all_mode_completion_analysis timeout 10 "$PY" -m scripts.research_v3_analyze_all_mode_completion
