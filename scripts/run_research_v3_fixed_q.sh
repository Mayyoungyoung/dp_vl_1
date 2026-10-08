#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=/home/wzy/dpvlm/route_set_v1
run="$root/runs/research_v3_v1"
py="$root/.venv/bin/python"
test -f "$run/frequency_coordinator_v3/closure.json"
test ! -e "$run/active.lock"
mkdir "$run/fixed_q_coordinator_v2"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$run/fixed_q_coordinator_v2/closure.json"' EXIT
bash scripts/launch_research_v3.sh --id fixed_q_exposure_tests_v2 -- "$py" -m pytest -q tests/test_research_v3_frequency_integrity.py tests/test_research_v3_frequency.py
bash scripts/launch_research_v3.sh --id fixed_q_exact_replay_v2 -- "$py" -m scripts.research_v3_check_fixed_q --output "$run/fixed_q_exact_replay_v2.json"
for arm in empirical_uniform empirical_90 empirical_98 balanced set_matching; do
  bash scripts/launch_research_v3.sh --id "fixed_q_evaluate_$arm" -- "$py" -m scripts.research_v3_frequency evaluate_fixed_q --name "frequency_$arm"
done
bash scripts/launch_research_v3.sh --id frequency_fixed_q_analysis_v2 -- "$py" -m scripts.research_v3_analyze_frequency --root "$run" --output "$run/frequency_fixed_q_analysis_v2" --evaluation evaluation_fixed_q_v2
bash scripts/launch_research_v3.sh --id frequency_integrity_v2 -- "$py" -m scripts.research_v3_frequency_integrity --root "$run" --output "$run/frequency_integrity_v2"
