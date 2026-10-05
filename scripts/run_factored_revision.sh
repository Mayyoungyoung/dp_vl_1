#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
run=/home/wzy/dpvlm/route_set_v1/runs/factored_q_v1
py=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
mkdir "$run/coordinator_endpoint_revision"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$run/coordinator_endpoint_revision/closure.json"' EXIT
"$py" -c 'import json;from pathlib import Path;r=Path("/home/wzy/dpvlm/route_set_v1/runs/factored_q_v1");assert json.load(open(r/"coordinator_diagnosis/closure.json"))["exit_code"]==0;assert json.load(open(r/"task_invariance_diagnosis.json"))["revision_gate"]'
{ nvidia-smi; uptime; free -h; df -h "$run"; } > "$run/coordinator_endpoint_revision/resources_before.txt"
bash scripts/launch_factored_q.sh --id endpoint_tests -- "$py" -m pytest -q tests/test_factored_q.py
for gs in 0 1 2; do
  for ss in 0 1 2; do
    bash scripts/launch_factored_q.sh --id "endpoint_g${gs}_s${ss}" -- "$py" -m scripts.run_factored_q run --arm conditional_endpoint --generator-seed "$gs" --scorer-seed "$ss" --evaluation-version evaluation_v2
    bash scripts/launch_factored_q.sh --id "endpoint_package_g${gs}_s${ss}" -- "$py" -m scripts.run_factored_q package --arm conditional_endpoint --generator-seed "$gs" --scorer-seed "$ss" --evaluation-version evaluation_v2 --deployment-version deployment_v2
  done
done
bash scripts/launch_factored_q.sh --id endpoint_cli -- "$py" -m scripts.check_factored_cli --arm conditional_endpoint
bash scripts/launch_factored_q.sh --id endpoint_invariance -- "$py" -m scripts.diagnose_factored_task --arm conditional_endpoint
bash scripts/launch_factored_q.sh --id analyze_final -- "$py" -m scripts.analyze_factored_q --evaluation-version evaluation_v2 --output "$run/analysis_final" --arms single joint marginal conditional conditional_endpoint --primary conditional_endpoint
