#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
for seed in 1 2 3 4; do
  for arm in success dense; do
    launch "rc_rep_${arm}_${seed}_fit" -m research_realized_coverage_v1.train_allocation --kind "$arm" --name "${arm}_context_fit${seed}" --feedback feedback_C_train,feedback_C_success_train --seed "$seed" --steps 2400
    extra=()
    if [[ "$arm" = dense ]]; then extra=(--start-head "success_context_fit${seed}/last.pt"); fi
    launch "rc_rep_${arm}_${seed}_eval" -m research_realized_coverage_v1.evaluate --name "${arm}_rep_seed${seed}" --head "${arm}_context_fit${seed}/last.pt" "${extra[@]}"
  done
done
