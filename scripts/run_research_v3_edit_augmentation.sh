#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/verified_edit_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job verified_edit_tests timeout 60 "$PY" -m pytest -q tests/test_research_v3_frequency.py tests/test_research_v3_counterfactual.py tests/test_segment_clearance.py
job verified_edit_prepare timeout 90 "$PY" -m scripts.research_v3_prepare_edit_support
job verified_edit_train timeout 420 "$PY" -m scripts.research_v3_frequency train --arm set_matching --name verified_edit_augmented --linear-control mean --verified-edit-support
job verified_edit_evaluate timeout 60 "$PY" -m scripts.research_v3_frequency evaluate_fixed_q --name verified_edit_augmented --paired-score
job verified_edit_analysis timeout 60 "$PY" -m scripts.research_v3_analyze_edit_augmentation
