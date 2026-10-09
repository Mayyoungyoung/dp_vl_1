#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
launch rc_net_diagnosis -m research_realized_coverage_v1.diagnose_allocator
launch rc_feedback_success_context -m research_realized_coverage_v1.feedback --name feedback_C_success_train --split TRAIN --proposal success_fit0/last.pt
for arm in success net; do
  launch "rc_${arm}_context_fit0" -m research_realized_coverage_v1.train_allocation --kind "$arm" --name "${arm}_context_fit0" --feedback feedback_C_train,feedback_C_success_train
  for step in 400 1200 2400; do
    extra=()
    if [[ "$arm" = net ]]; then extra=(--start-head success_context_fit0/step2400.pt); fi
    launch "rc_${arm}_context_eval${step}" -m research_realized_coverage_v1.evaluate --name "${arm}_context_step${step}" --head "${arm}_context_fit0/step${step}.pt" "${extra[@]}"
  done
done
