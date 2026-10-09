#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
launch() { bash scripts/launch_realized_coverage_v1.sh --id "$1" -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python "${@:2}"; }
launch rc_final_tests -m pytest -q tests/test_realized_coverage.py tests/test_mode_geometry.py tests/test_analyze_verified_set.py
launch rc_final_statistics -m research_realized_coverage_v1.statistics --output FIVE_SEED_FINAL.json
launch rc_final_summary -m research_realized_coverage_v1.final_summary
launch rc_final_audit -m research_realized_coverage_v1.audit
