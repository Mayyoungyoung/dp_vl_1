#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
R=/home/wzy/dpvlm/route_set_v1/runs/feasible_space_v1
bash "$L" --id final_tests_v1 -- "$P" -m unittest discover -s tests -p test_feasible_space.py -v
bash "$L" --id resume_full_v1 -- "$P" -m research_feasible_space_v1.train --name resume_full_v1 --arm bounded --steps 100 --cell-loss --tapered-cells
bash "$L" --id resume_part_v1 -- "$P" -m research_feasible_space_v1.train --name resume_part_v1 --arm bounded --steps 100 --cell-loss --tapered-cells --stop-after 50
bash "$L" --id resume_cont_v1 -- "$P" -m research_feasible_space_v1.train --name resume_part_v1 --arm bounded --steps 100 --cell-loss --tapered-cells --resume
bash "$L" --id resume_compare_v1 -- "$P" -m research_feasible_space_v1.resume_audit --names resume_full_v1 resume_part_v1 --output RESUME_EXACT.json
for arm in xyz bounded; do
  bash "$L" --id deployment_${arm}_v1 -- "$P" -m research_feasible_space_v1.check_deployment --name deployment_${arm}_v1 --checkpoint "$R/tapered_${arm}_seed0/last.pt" --head "$R/success_tapered_${arm}_seed0/last.pt" --evaluation refreshed_tapered_${arm}_seed0
done
bash "$L" --id fixed_witness_analysis_v1 -- "$P" -m research_feasible_space_v1.analyze --names refreshed_tapered_xyz_seed0:adaptive refreshed_tapered_bounded_seed0:adaptive refreshed_tapered_projection_seed0:adaptive refreshed_tapered_boundcenter_seed0:adaptive refreshed_tapered_bounded_seed1:adaptive refreshed_tapered_bounded_seed2:adaptive --output fixed_witness_v1 --figures
bash "$L" --id real_corridor_figures_v1 -- "$P" -m research_feasible_space_v1.figures --name real_corridor_figures_v1 --analysis fixed_witness_v1 --xyz refreshed_tapered_xyz_seed0 --bounded refreshed_tapered_bounded_seed0
bash "$L" --id saved_outcome_diagnostic_v1 -- "$P" -m research_feasible_space_v1.diagnose_outcomes --name saved_outcome_diagnostic_v1 --names refreshed_tapered_xyz_seed0 refreshed_tapered_bounded_seed0 refreshed_tapered_projection_seed0 refreshed_tapered_boundcenter_seed0
bash "$L" --id three_seed_statistics_v1 -- "$P" -m research_feasible_space_v1.statistics --name three_seed_statistics_v1
bash "$L" --id summary_v1 -- "$P" -m research_feasible_space_v1.summarize --name summary_v1
