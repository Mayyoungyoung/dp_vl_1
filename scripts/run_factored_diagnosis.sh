#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run=/home/wzy/dpvlm/route_set_v1/runs/factored_q_v1
py=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
mkdir "$run/coordinator_diagnosis"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$run/coordinator_diagnosis/closure.json"' EXIT
while test ! -e "$run/coordinator_calibration_fix/closure.json"; do sleep 10; done
"$py" -c 'import json;assert json.load(open("/home/wzy/dpvlm/route_set_v1/runs/factored_q_v1/coordinator_calibration_fix/closure.json"))["exit_code"]==0'
bash scripts/launch_factored_q.sh --id train_task_invariance -- "$py" -m scripts.diagnose_factored_task
