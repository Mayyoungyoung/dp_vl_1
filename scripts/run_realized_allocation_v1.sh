#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
launch rc_tests_v1 -m pytest -q tests/test_realized_coverage.py tests/test_mode_geometry.py tests/test_paired_modes_evaluation.py
for arm in success net; do
  launch "rc_${arm}_fit0" -m research_realized_coverage_v1.train_allocation --kind "$arm" --name "${arm}_fit0"
  for step in 400 1200 2400; do
    launch "rc_${arm}_eval${step}" -m research_realized_coverage_v1.evaluate --name "${arm}_step${step}" --head "${arm}_fit0/step${step}.pt"
  done
done
