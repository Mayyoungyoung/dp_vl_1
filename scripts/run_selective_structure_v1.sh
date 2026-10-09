#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
[[ -e "$R/local_diffusion_interventions_grid_v1/RESULTS.json" ]] || { echo 'Previous control incomplete';exit 1; }
for population in old interventions;do
 if [[ "$population" == old ]];then data="$P/data/paired_modes_v1_v2";else data="$P/data/selective_repair_interventions_v1";fi
 bash "$L" --id ${population}_tail_bound_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.tail_bound --name ${population}_tail_bound_v1 --data-name ${population}_calibrated_targets_v1 --data "$data"
done
bash "$L" --id calibrated_recenter_stable_seed0 -- "$P/.venv/bin/python" -m research_selective_repair_v1.train --name calibrated_recenter_stable_seed0 --arm recenter --goal-tail --center-lr .00003 --detach-local --data old_calibrated_targets_v1,interventions_calibrated_targets_v1
for population in old interventions;do
 if [[ "$population" == old ]];then data="$P/data/paired_modes_v1_v2";else data="$P/data/selective_repair_interventions_v1";fi
 bash "$L" --id calibrated_recenter_stable_${population}_grid_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.grid --name calibrated_recenter_stable_${population}_grid_v1 --checkpoint "$R/calibrated_recenter_stable_seed0/last.pt" --data-name ${population}_calibrated_targets_v1 --data "$data"
done
