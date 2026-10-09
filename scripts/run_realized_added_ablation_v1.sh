#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/launch_realized_coverage_v1.sh --id rc_added_only_fit -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_realized_coverage_v1.train_allocation --kind added_only --name added_only_context_fit0 --feedback feedback_C_train,feedback_C_success_train
bash scripts/launch_realized_coverage_v1.sh --id rc_added_only_eval -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_realized_coverage_v1.evaluate --name added_only_context_step2400 --head added_only_context_fit0/last.pt --start-head success_context_fit0/step2400.pt
