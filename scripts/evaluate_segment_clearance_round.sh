#!/usr/bin/env bash
set -euo pipefail
launcher="$(dirname "$0")/launch_segment_clearance.sh"
python=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
# Evaluation-only continuation after training; never relaunch training here.
for split in old_dev dev32; do
  for seed in 0 1 2; do
    for arm in A B; do
      bash "$launcher" --id "eval_v2_${split}_${arm}_seed${seed}" -- "$python" -m scripts.evaluate_segment_clearance --split "$split" --seed "$seed" --arm "$arm"
    done
  done
done
