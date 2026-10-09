#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
for arm in xyz bounded; do
  bash "$L" --id feedback_${arm}_seed0 -- "$P" -m research_feasible_space_v1.feedback --name feedback_${arm}_seed0 --checkpoint "$R/screen_${arm}_seed0/last.pt"
  bash "$L" --id success_${arm}_seed0 -- "$P" -m research_feasible_space_v1.feedback --name success_${arm}_seed0 --pool feedback_${arm}_seed0
  bash "$L" --id eval_refreshed_${arm}_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_${arm}_seed0 --checkpoint "$R/screen_${arm}_seed0/last.pt" --head "$R/success_${arm}_seed0/last.pt" --perturb
done
bash "$L" --id eval_refreshed_center_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_center_seed0 --checkpoint "$R/screen_bounded_seed0/last.pt" --head "$R/success_bounded_seed0/last.pt" --kind center
bash "$L" --id eval_refreshed_projection_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_projection_seed0 --checkpoint "$R/screen_xyz_seed0/last.pt" --head "$R/success_xyz_seed0/last.pt" --kind projection
