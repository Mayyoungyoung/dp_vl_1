#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
for seed in 1 2; do
  for arm in xyz bounded; do
    bash "$L" --id tapered_${arm}_seed${seed} -- "$P" -m research_feasible_space_v1.train --name tapered_${arm}_seed${seed} --arm "$arm" --steps 2400 --seed "$seed" --cell-loss --tapered-cells
    bash "$L" --id feedback_tapered_${arm}_seed${seed} -- "$P" -m research_feasible_space_v1.feedback --name feedback_tapered_${arm}_seed${seed} --checkpoint "$R/tapered_${arm}_seed${seed}/last.pt"
    bash "$L" --id success_tapered_${arm}_seed${seed} -- "$P" -m research_feasible_space_v1.feedback --name success_tapered_${arm}_seed${seed} --pool feedback_tapered_${arm}_seed${seed} --seed "$seed"
    bash "$L" --id eval_refreshed_tapered_${arm}_seed${seed} -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_tapered_${arm}_seed${seed} --checkpoint "$R/tapered_${arm}_seed${seed}/last.pt" --head "$R/success_tapered_${arm}_seed${seed}/last.pt" --perturb
  done
  for kind in center projection; do
    bash "$L" --id eval_refreshed_tapered_${kind}_seed${seed} -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_tapered_${kind}_seed${seed} --checkpoint "$R/tapered_xyz_seed${seed}/last.pt" --head "$R/success_tapered_xyz_seed${seed}/last.pt" --kind "$kind"
  done
  bash "$L" --id eval_refreshed_tapered_boundcenter_seed${seed} -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_tapered_boundcenter_seed${seed} --checkpoint "$R/tapered_bounded_seed${seed}/last.pt" --head "$R/success_tapered_bounded_seed${seed}/last.pt" --kind center
done
