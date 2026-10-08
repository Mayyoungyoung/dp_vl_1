#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/frozen_factorial_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job frozen_factorial_tests timeout 30 "$PY" -m pytest -q tests/test_research_v3_frozen_input.py tests/test_research_v3_frequency.py tests/test_segment_clearance.py
job frozen_plain_train timeout 360 "$PY" -m scripts.research_v3_frequency train --arm set_matching --name frozen_encoder_plain --linear-control mean --freeze-input-encoders
job frozen_augmented_train timeout 360 "$PY" -m scripts.research_v3_frequency train --arm set_matching --name frozen_encoder_augmented --linear-control mean --verified-edit-support --freeze-input-encoders
job frozen_plain_evaluate timeout 30 "$PY" -m scripts.research_v3_frequency evaluate_fixed_q --name frozen_encoder_plain --paired-score
job frozen_augmented_evaluate timeout 30 "$PY" -m scripts.research_v3_frequency evaluate_fixed_q --name frozen_encoder_augmented --paired-score
job frozen_factorial_analysis timeout 30 "$PY" -m scripts.research_v3_analyze_frozen_factorial
