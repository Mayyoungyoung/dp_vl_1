#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=/home/wzy/dpvlm/route_set_v1
run="$root/runs/research_v3_v1"
py="$root/.venv/bin/python"
test -f "$run/fixed_q_coordinator_v2/closure.json"
test ! -e "$run/active.lock"
mkdir "$run/sampled_coordinator_v1"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$run/sampled_coordinator_v1/closure.json"' EXIT
bash scripts/launch_research_v3.sh --id sampled_control_tests_v1 -- "$py" -m pytest -q tests/test_research_v3_sampled.py tests/test_research_v3_counterfactual.py
bash scripts/launch_research_v3.sh --id frequency_train_set_sampled -- "$py" -m scripts.research_v3_frequency train --arm set_sampled --name frequency_set_sampled
bash scripts/launch_research_v3.sh --id fixed_q_evaluate_set_sampled -- "$py" -m scripts.research_v3_frequency evaluate_fixed_q --name frequency_set_sampled
bash scripts/launch_research_v3.sh --id frequency_sampled_analysis_v1 -- "$py" -m scripts.research_v3_analyze_sampled --output frequency_sampled_analysis_v1
bash scripts/launch_research_v3.sh --id frequency_counterfactual_v1 -- "$py" -m scripts.research_v3_counterfactual --output frequency_counterfactual_v1
