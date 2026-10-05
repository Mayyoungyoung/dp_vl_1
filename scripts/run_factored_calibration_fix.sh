#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=/home/wzy/dpvlm/route_set_v1
run="$root/runs/factored_q_v1"
py="$root/.venv/bin/python"
mkdir "$run/coordinator_calibration_fix"
date -u +%FT%TZ > "$run/coordinator_calibration_fix/started"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$run/coordinator_calibration_fix/closure.json"' EXIT
while test ! -e "$run/coordinator/closure.json"; do sleep 10; done
"$py" -c 'import json;assert json.load(open("/home/wzy/dpvlm/route_set_v1/runs/factored_q_v1/coordinator/closure.json"))["exit_code"]==0'
{ nvidia-smi; uptime; free -h; df -h "$root"; } > "$run/coordinator_calibration_fix/resources_before.txt"
bash scripts/launch_factored_q.sh --id calibration_fix_tests -- "$py" -m pytest -q tests/test_factored_q.py
for gs in 0 1 2; do
  for ss in 0 1 2; do
    for arm in single joint marginal conditional; do
      bash scripts/launch_factored_q.sh --id "calibration_v2_${arm}_g${gs}_s${ss}" -- "$py" -m scripts.run_factored_q evaluate --arm "$arm" --generator-seed "$gs" --scorer-seed "$ss" --evaluation-version evaluation_v2
    done
    bash scripts/launch_factored_q.sh --id "package_v2_g${gs}_s${ss}" -- "$py" -m scripts.run_factored_q package --generator-seed "$gs" --scorer-seed "$ss" --evaluation-version evaluation_v2 --deployment-version deployment_v2
  done
done
bash scripts/launch_factored_q.sh --id public_cli_v2 -- "$py" -m scripts.check_factored_cli
bash scripts/launch_factored_q.sh --id analyze_v2 -- "$py" -m scripts.analyze_factored_q --evaluation-version evaluation_v2 --output "$run/analysis_v2"
