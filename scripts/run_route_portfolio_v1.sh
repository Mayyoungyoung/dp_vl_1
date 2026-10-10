#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
L="$S/scripts/launch_selective_repair_v1.sh"
P=/home/wzy/dpvlm/route_set_v1
M="$P/runs/mode_geometry_v1"
for seed in 0 1 2; do
  for arm in B0 Bset C; do
    bash "$L" --id portfolio_shift_${arm}_seed${seed}_v1 -- "$P/.venv/bin/python" -m research_route_portfolio_v1.evaluate --name shift_${arm}_seed${seed} --checkpoint "$M/canonical_${arm}_seed${seed}/last.pt" --sampling adaptive
  done
  for inference in 71239 71240 71241; do
    bash "$L" --id portfolio_shift_C_seed${seed}_ordinary_${inference}_v1 -- "$P/.venv/bin/python" -m research_route_portfolio_v1.evaluate --name shift_C_seed${seed}_ordinary_${inference} --checkpoint "$M/canonical_C_seed${seed}/last.pt" --sampling ordinary --inference-seed "$inference"
  done
done
for seed in 0 1 2 3 4; do
  bash "$L" --id portfolio_shift_C0_success_seed${seed}_v1 -- "$P/.venv/bin/python" -m research_route_portfolio_v1.evaluate --name shift_C0_success_seed${seed} --checkpoint "$M/canonical_C_seed0/last.pt" --head "$P/runs/realized_coverage_v1/success_context_fit${seed}/last.pt"
done
