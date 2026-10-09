#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
bash "$L" --id stageA_model_tests_1 -- "$P" -m unittest discover -s tests -p test_feasible_space.py -v
for arm in xyz bounded; do
  bash "$L" --id oracle_${arm}_0 -- "$P" -m research_feasible_space_v1.train --name oracle_${arm}_0 --arm "$arm" --steps 600 --oracle
done
bash "$L" --id oracle_diagnostic_0 -- "$P" -m research_feasible_space_v1.oracle_diagnostic --name oracle_diagnostic_0 --checkpoints "$R/oracle_xyz_0/last.pt" "$R/oracle_bounded_0/last.pt"
