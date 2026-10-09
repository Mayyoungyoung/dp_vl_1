#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/launch_realized_coverage_v1.sh --id rc_reference_diagnosis -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_realized_coverage_v1.diagnose_references
bash scripts/run_realized_five_seed_v1.sh
bash scripts/run_realized_added_ablation_v1.sh
bash scripts/run_realized_checks_v1.sh
bash scripts/launch_realized_coverage_v1.sh --id rc_allocation_statistics -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_realized_coverage_v1.statistics
bash scripts/launch_realized_coverage_v1.sh --id rc_progress_report -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_realized_coverage_v1.summarize
