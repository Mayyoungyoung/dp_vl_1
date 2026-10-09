#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_selective_repair_v1.sh"
P=/home/wzy/dpvlm/route_set_v1
R="$P/runs/selective_repair_v1"
# This queue waits for the already running concentrated analysis, then never
# overwrites it. Each child is independently receipted and has a fresh output.
while [[ -e "$R/active.lock" ]]; do sleep 5; done
[[ -e "$R/prepared_v1/MANIFEST.json" ]] || { echo 'Opportunity preparation did not finish'; exit 1; }
bash "$L" --id goal_prepare_recovery_v1 -- "$P/.venv/bin/python" -c "from research_selective_repair_v1.goal_data import recover; recover()"
bash "$L" --id structural_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p test_selective_repair_v1.py
for arm in selective residual recenter; do
  bash "$L" --id screen_${arm}_seed0 -- "$P/.venv/bin/python" -m research_selective_repair_v1.train --name screen_${arm}_seed0 --arm "$arm" --goal-tail --data goal_prepared_v1
  bash "$L" --id screen_eval_${arm}_seed0 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name screen_eval_${arm}_seed0 --checkpoint "$R/screen_${arm}_seed0/last.pt" --prototype "$R/goal_prepared_v1/PROTOTYPES.json"
done
bash "$L" --id baseline_zero_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name baseline_zero_v1 --kind zero
bash "$L" --id baseline_geometry_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name baseline_geometry_v1 --kind geometric

for trigger in 0 .01 .025 .05; do
 bash "$L" --id goal_rule_${trigger}_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name goal_rule_${trigger}_v1 --kind goal_rule --threshold "$trigger" --prototype "$R/goal_prepared_v1/PROTOTYPES.json"
done
