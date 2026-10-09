#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
bash "$L" --id oracle_diagnostic_1 -- "$P" -m research_feasible_space_v1.oracle_diagnostic --name oracle_diagnostic_1 --checkpoints "$R/oracle_xyz_0/last.pt" "$R/oracle_bounded_0/last.pt"
for arm in xyz bounded; do
  bash "$L" --id envelope_${arm}_seed0 -- "$P" -m research_feasible_space_v1.train --name envelope_${arm}_seed0 --arm "$arm" --steps 2400 --cell-loss
  bash "$L" --id eval_envelope_${arm}_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name envelope_${arm}_seed0 --checkpoint "$R/envelope_${arm}_seed0/last.pt" --perturb
  bash "$L" --id feedback_envelope_${arm}_seed0 -- "$P" -m research_feasible_space_v1.feedback --name feedback_envelope_${arm}_seed0 --checkpoint "$R/envelope_${arm}_seed0/last.pt"
  bash "$L" --id success_envelope_${arm}_seed0 -- "$P" -m research_feasible_space_v1.feedback --name success_envelope_${arm}_seed0 --pool feedback_envelope_${arm}_seed0
  bash "$L" --id eval_refreshed_envelope_${arm}_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_envelope_${arm}_seed0 --checkpoint "$R/envelope_${arm}_seed0/last.pt" --head "$R/success_envelope_${arm}_seed0/last.pt" --perturb
done
for kind in center projection; do
  bash "$L" --id eval_refreshed_envelope_${kind}_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_envelope_${kind}_seed0 --checkpoint "$R/envelope_xyz_seed0/last.pt" --head "$R/success_envelope_xyz_seed0/last.pt" --kind "$kind"
done
bash "$L" --id eval_refreshed_envelope_boundcenter_seed0 -- "$P" -m research_feasible_space_v1.evaluate --name refreshed_envelope_boundcenter_seed0 --checkpoint "$R/envelope_bounded_seed0/last.pt" --head "$R/success_envelope_bounded_seed0/last.pt" --kind center
