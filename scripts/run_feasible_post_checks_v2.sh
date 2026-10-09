#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
bash "$L" --id observation_tests_v2 -- "$P" -m unittest discover -s tests -p test_feasible_space.py -v
for arm in xyz bounded; do
  bash "$L" --id deployment_${arm}_v2 -- "$P" -m research_feasible_space_v1.check_deployment --name deployment_${arm}_v2 --checkpoint "$R/tapered_${arm}_seed0/last.pt" --head "$R/success_tapered_${arm}_seed0/last.pt" --evaluation refreshed_tapered_${arm}_seed0
done
bash "$L" --id actual_reachable_figures_v2 -- "$P" -m research_feasible_space_v1.figures --name actual_reachable_figures_v2 --analysis fixed_witness_v1 --xyz refreshed_tapered_xyz_seed0 --bounded refreshed_tapered_bounded_seed0
bash "$L" --id observation_visibility_v2 -- "$P" -m research_feasible_space_v1.audit_visibility --name observation_visibility_v2 --names refreshed_tapered_xyz_seed0 refreshed_tapered_bounded_seed0
