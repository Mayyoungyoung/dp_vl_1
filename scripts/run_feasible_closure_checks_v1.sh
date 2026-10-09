#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_feasible_space_v1.sh"
P=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
names=()
for seed in 0 1 2; do
  for arm in xyz bounded; do names+=("refreshed_tapered_${arm}_seed${seed}:adaptive"); done
  for arm in projection center boundcenter; do names+=("refitted_control_${arm}_seed${seed}:adaptive"); done
done
bash "$L" --id fixed_witness_refitted_v2 -- "$P" -m research_feasible_space_v1.analyze --names "${names[@]}" --output fixed_witness_refitted_v2
bash "$L" --id fixed_witness_statistics_v2 -- "$P" -m research_feasible_space_v1.witness_statistics --name fixed_witness_statistics_v2
bash "$L" --id parameter_transfer_v1 -- "$P" -m research_feasible_space_v1.parameter_transfer --name parameter_transfer_v1
bash "$L" --id summary_v2 -- "$P" -m research_feasible_space_v1.summarize --name summary_v2
bash "$L" --id dependency_snapshot_v1 -- "$P" -m research_feasible_space_v1.dependency_snapshot --name dependencies_v1
