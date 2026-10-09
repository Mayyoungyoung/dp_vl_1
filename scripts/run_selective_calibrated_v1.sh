#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
D="$P/data/selective_repair_interventions_v1"
while kill -0 622165 2>/dev/null; do sleep 5; done
[[ -e "$R/interventions_goal_v1/MANIFEST.json" ]] || { echo 'Previous coordinator stopped; inspect failure before continuation'; exit 1; }
bash "$L" --id structural_tests_v3 -- "$P/.venv/bin/python" -m unittest discover -s tests -p 'test_selective*v1.py'
bash "$L" --id calibrated_prototype_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.grounding --name calibrated_prototype_v1 --parent "$R/goal_prepared_v1/PROTOTYPES.json"
for population in old interventions; do
 if [[ "$population" == old ]]; then
  data="$P/data/paired_modes_v1_v2";src=prepared_v1
 else data="$D";src=interventions_prepared_v1;fi
 bash "$L" --id ${population}_observed_cache_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.observed_cache --name ${population}_observed_cache_v1 --source "$src" --data "$data"
 bash "$L" --id ${population}_calibrated_targets_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.goal_data --name ${population}_calibrated_targets_v1 --source ${population}_observed_cache_v1 --data "$data" --prototype "$R/calibrated_prototype_v1/PROTOTYPES.json"
 bash "$L" --id ${population}_zero_calibrated_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name ${population}_zero_calibrated_v1 --kind zero --data "$data"
 for threshold in 0 .01 .025 .05; do
  bash "$L" --id ${population}_rule_calibrated_${threshold}_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name ${population}_rule_calibrated_${threshold}_v1 --kind goal_rule --threshold "$threshold" --data "$data" --prototype "$R/calibrated_prototype_v1/PROTOTYPES.json"
 done
 bash "$L" --id ${population}_geometry_calibrated_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name ${population}_geometry_calibrated_v1 --kind geometric --data "$data" --prototype "$R/calibrated_prototype_v1/PROTOTYPES.json"
done
for seed in 0 1 2; do
 for arm in selective residual recenter; do
  name=calibrated_${arm}_seed${seed}
  bash "$L" --id "$name" -- "$P/.venv/bin/python" -m research_selective_repair_v1.train --name "$name" --arm "$arm" --goal-tail --seed "$seed" --data old_calibrated_targets_v1,interventions_calibrated_targets_v1
  for population in old interventions; do
   if [[ "$population" == old ]]; then data="$P/data/paired_modes_v1_v2";else data="$D";fi
   grid=${name}_${population}_grid
   bash "$L" --id "$grid" -- "$P/.venv/bin/python" -m research_selective_repair_v1.grid --name "$grid" --checkpoint "$R/$name/last.pt" --data-name ${population}_calibrated_targets_v1 --data "$data"
  done
 done
done
