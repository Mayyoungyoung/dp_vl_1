#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
R="$P/runs/selective_repair_v1"
while kill -0 630925 2>/dev/null; do sleep 5; done
[[ -e "$R/calibrated_recenter_seed2_interventions_grid/RESULTS.json" ]] || { echo 'Paired calibrated coordinator stopped; inspect failure'; exit 1; }
bash "$L" --id local_diffusion_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p test_selective_diffusion_v1.py
bash "$L" --id local_diffusion_seed0 -- "$P/.venv/bin/python" -m research_selective_repair_v1.patch_diffusion --name local_diffusion_seed0
for population in old interventions; do
 if [[ "$population" == old ]];then data="$P/data/paired_modes_v1_v2";else data="$P/data/selective_repair_interventions_v1";fi
 bash "$L" --id local_diffusion_${population}_grid_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.grid --name local_diffusion_${population}_grid_v1 --checkpoint "$R/local_diffusion_seed0/last.pt" --data-name ${population}_calibrated_targets_v1 --data "$data"
done
