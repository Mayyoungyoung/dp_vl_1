#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=/home/wzy/dpvlm/route_set_v1
py="$root/.venv/bin/python"
run="$root/runs/research_v3_v1"
mkdir "$run/frequency_coordinator_v3"
trap 'code=$?; printf "{\"exit_code\":%s}\n" "$code" > "$run/frequency_coordinator_v3/closure.json"' EXIT
{ nvidia-smi; free -h; df -h "$root"; } > "$run/frequency_coordinator_v3/resources_before.txt"
for family in paired_modes_v1 factored_q_v1 observed_probability_v1; do
  test ! -e "$root/runs/$family/active.lock"
done
bash scripts/launch_research_v3.sh --id frequency_history_tests_v3 -- "$py" -m pytest -q tests/test_research_v3_frequency.py
bash scripts/launch_research_v3.sh --id frequency_resume_continuous_v3 -- "$py" -m scripts.research_v3_frequency train --arm empirical_uniform --name frequency_resume_continuous_v3 --steps 100
bash scripts/launch_research_v3.sh --id frequency_resume_part1_v3 -- "$py" -m scripts.research_v3_frequency train --arm empirical_uniform --name frequency_resume_split_v3 --steps 100 --stop-after 50
bash scripts/launch_research_v3.sh --id frequency_resume_part2_v3 -- "$py" -m scripts.research_v3_frequency train --arm empirical_uniform --name frequency_resume_split_v3 --steps 100 --resume
bash scripts/launch_research_v3.sh --id frequency_resume_exact_v3 -- "$py" -m scripts.research_v3_resume_check --a "$run/frequency_resume_continuous_v3/last.pt" --b "$run/frequency_resume_split_v3/last.pt" --output "$run/frequency_resume_exact_v3.json"
for arm in empirical_uniform empirical_90 empirical_98 balanced set_matching; do
  bash scripts/launch_research_v3.sh --id "frequency_train_$arm" -- "$py" -m scripts.research_v3_frequency train --arm "$arm" --name "frequency_$arm"
  bash scripts/launch_research_v3.sh --id "frequency_evaluate_$arm" -- "$py" -m scripts.research_v3_frequency evaluate --name "frequency_$arm"
done
bash scripts/launch_research_v3.sh --id frequency_analysis_v1 -- "$py" -m scripts.research_v3_analyze_frequency --root "$run" --output "$run/frequency_analysis_v1"
