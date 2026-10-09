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
for arm in selective residual recenter; do
  bash "$L" --id screen_${arm}_seed0 -- "$P/.venv/bin/python" -m research_selective_repair_v1.train --name screen_${arm}_seed0 --arm "$arm"
  bash "$L" --id screen_eval_${arm}_seed0 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name screen_eval_${arm}_seed0 --checkpoint "$R/screen_${arm}_seed0/last.pt"
done
bash "$L" --id baseline_zero_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name baseline_zero_v1 --kind zero
bash "$L" --id baseline_geometry_v1 -- "$P/.venv/bin/python" -m research_selective_repair_v1.evaluate --name baseline_geometry_v1 --kind geometric
