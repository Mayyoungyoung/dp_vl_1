#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/canonical_resume_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
cp "$P/runs/research_v3_v1/frequency_set_canonical/recovery.pt" "$C/recovery_before_resume.pt"
sha256sum "$C/recovery_before_resume.pt" > "$C/recovery_before_sha256.txt"
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job canonical_train_resume timeout 240 "$PY" -m scripts.research_v3_frequency train --arm set_canonical --name frequency_set_canonical --resume
job canonical_evaluate timeout 60 "$PY" -m scripts.research_v3_frequency evaluate_fixed_q --name frequency_set_canonical
job canonical_analysis timeout 60 "$PY" -m scripts.research_v3_analyze_canonical
