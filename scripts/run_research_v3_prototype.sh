#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
PY="$P/.venv/bin/python"
C="$P/runs/research_v3_v1/prototype_coordinator_v1"
mkdir "$C"
exec > >(tee "$C/stdout.log") 2>&1
printf '%s\n' "$$" > "$C/pid.txt"
sha256sum "$0" > "$C/launcher_sha256.txt"
printf '%s\n' "$(basename "$PWD")" > "$C/source_commit.txt"
trap 'code=$?; printf "{\"exit_code\":%d,\"elapsed_seconds\":%d}\n" "$code" "$SECONDS" > "$C/closure.json"' EXIT
R="$P/runs/research_v3_v1/calibration_control_coordinator_v1/closure.json"
while [[ ! -f "$R" ]]; do
  if (( SECONDS > 2700 )); then echo 'Follow-up wait limit reached'; exit 1; fi
  sleep 10
done
"$PY" -c "import json; assert json.load(open('$R'))['exit_code']==0"
job() { local id="$1"; shift; bash scripts/launch_research_v3.sh --id "$id" -- "$@"; }
job prototype_stride_tests "$PY" -m pytest -q tests/test_research_v3_prototype.py
job prototype_localization_v1 timeout 300 "$PY" -m scripts.research_v3_prototype --output prototype_localization_v1
