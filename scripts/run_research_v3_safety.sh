#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=/home/wzy/dpvlm/route_set_v1
run="$root/runs/research_v3_v1"
py="$root/.venv/bin/python"
test ! -e "$run/active.lock"
mkdir "$run/safety_coordinator_v1"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$run/safety_coordinator_v1/closure.json"' EXIT
for arm in mean worst; do
  bash scripts/launch_research_v3.sh --id "safety_train_$arm" -- "$py" -m scripts.research_v3_frequency train --arm set_matching --name "safety_$arm" --safety-control "$arm"
  bash scripts/launch_research_v3.sh --id "safety_evaluate_$arm" -- "$py" -m scripts.research_v3_frequency evaluate_fixed_q --name "safety_$arm"
done
bash scripts/launch_research_v3.sh --id safety_analysis_v1 -- "$py" -m scripts.research_v3_analyze_safety --output safety_analysis_v1
