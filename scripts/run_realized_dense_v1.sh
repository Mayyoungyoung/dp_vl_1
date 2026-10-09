#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
launch rc_tests_dense -m pytest -q tests/test_realized_coverage.py tests/test_mode_geometry.py
for arm in dense no_peer; do
  launch "rc_${arm}_context_fit0" -m research_realized_coverage_v1.train_allocation --kind "$arm" --name "${arm}_context_fit0" --feedback feedback_C_train,feedback_C_success_train
  for step in 400 1200 2400; do
    launch "rc_${arm}_context_eval${step}" -m research_realized_coverage_v1.evaluate --name "${arm}_context_step${step}" --head "${arm}_context_fit0/step${step}.pt" --start-head success_context_fit0/step2400.pt
  done
done
