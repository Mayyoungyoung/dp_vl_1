#!/usr/bin/env bash
set -euo pipefail
S="$(cd "$(dirname "$0")/.." && pwd)"
P=/home/wzy/dpvlm/route_set_v1
L="$S/scripts/launch_selective_repair_v1.sh"
# Frozen main study is the predecessor; no script or imported file is overwritten.
while kill -0 1363718 2>/dev/null; do sleep 5; done
test -f "$P/runs/route_portfolio_v1/shift_C0_success_seed4/RESULTS.json"
bash "$L" --id portfolio_preference_contract_tests_v1 -- "$P/.venv/bin/python" -m unittest discover -s tests -p test_route_portfolio_preference.py -q
for seed in 0 1 2; do
  bash "$L" --id portfolio_preference_C_seed${seed}_v1 -- "$P/.venv/bin/python" -m research_route_portfolio_v1.preference_evaluate --seed "$seed"
done
