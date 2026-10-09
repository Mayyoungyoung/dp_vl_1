#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
bash scripts/launch_realized_coverage_v1.sh --id rc_feedback_train -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_realized_coverage_v1.feedback --name feedback_C_train --split TRAIN
bash scripts/launch_realized_coverage_v1.sh --id rc_feedback_dev -- /home/wzy/dpvlm/route_set_v1/.venv/bin/python -m research_realized_coverage_v1.feedback --name diagnostic_C_dev --split DEV_MODEL
