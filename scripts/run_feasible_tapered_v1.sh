#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
bash "$L" --id tapered_tests_v1 -- "$P" -m unittest discover -s tests -p test_feasible_space.py -v
for arm in xyz bounded; do
  bash "$L" --id tapered_${arm}_seed0 -- "$P" -m research_feasible_space_v1.train --name tapered_${arm}_seed0 --arm "$arm" --steps 2400 --cell-loss --tapered-cells
  bash "$L" --id eval_tapered_${arm}_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name tapered_${arm}_seed0 --checkpoint "$R/tapered_${arm}_seed0/last.pt" --perturb
  bash "$L" --id feedback_tapered_${arm}_seed0 -- "$P" -m research_feasible_space_v1.feedback --name feedback_tapered_${arm}_seed0 --checkpoint "$R/tapered_${arm}_seed0/last.pt"
  bash "$L" --id success_tapered_${arm}_seed0 -- "$P" -m research_feasible_space_v1.feedback --name success_tapered_${arm}_seed0 --pool feedback_tapered_${arm}_seed0
  bash "$L" --id eval_refreshed_tapered_${arm}_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_tapered_${arm}_seed0 --checkpoint "$R/tapered_${arm}_seed0/last.pt" --head "$R/success_tapered_${arm}_seed0/last.pt" --perturb
done
for kind in center projection; do
  bash "$L" --id eval_refreshed_tapered_${kind}_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_tapered_${kind}_seed0 --checkpoint "$R/tapered_xyz_seed0/last.pt" --head "$R/success_tapered_xyz_seed0/last.pt" --kind "$kind"
done
bash "$L" --id eval_refreshed_tapered_boundcenter_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_tapered_boundcenter_seed0 --checkpoint "$R/tapered_bounded_seed0/last.pt" --head "$R/success_tapered_bounded_seed0/last.pt" --kind center
