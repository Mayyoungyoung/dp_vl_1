#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
bash "$L" --id final_tests_v1 -- "$P" -m unittest discover -s tests -p test_feasible_space.py -v
bash "$L" --id resume_full_v1 -- "$P" -m research_feasible_space_v1.train --name resume_full_v1 --arm bounded --steps 100 --cell-loss
bash "$L" --id resume_part_v1 -- "$P" -m research_feasible_space_v1.train --name resume_part_v1 --arm bounded --steps 100 --cell-loss --stop-after 50
bash "$L" --id resume_cont_v1 -- "$P" -m research_feasible_space_v1.train --name resume_part_v1 --arm bounded --steps 100 --cell-loss --resume
bash "$L" --id resume_compare_v1 -- "$P" -m research_feasible_space_v1.resume_audit --names resume_full_v1 resume_part_v1 --output RESUME_EXACT.json
for arm in xyz bounded; do
  bash "$L" --id deployment_${arm}_v1 -- "$P" -m research_feasible_space_v1.check_deployment --name deployment_${arm}_v1 --checkpoint "$R/envelope_${arm}_seed0/last.pt" --head "$R/success_envelope_${arm}_seed0/last.pt" --evaluation refreshed_envelope_${arm}_seed0
done
bash "$L" --id fixed_witness_analysis_v1 -- "$P" -m research_feasible_space_v1.analyze --names refreshed_envelope_xyz_seed0:adaptive refreshed_envelope_bounded_seed0:adaptive refreshed_envelope_projection_seed0:adaptive refreshed_envelope_boundcenter_seed0:adaptive --output fixed_witness_v1 --figures
bash "$L" --id summary_v1 -- "$P" -m research_feasible_space_v1.summarize --name summary_v1
