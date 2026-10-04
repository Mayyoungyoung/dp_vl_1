#!/usr/bin/env bash
set -euo pipefail
launcher="$(dirname "$0")/launch_segment_clearance.sh"
python=/home/wzy/dpvlm/route_set_v1/.venv/bin/python
for seed in 0 1 2; do
  bash "$launcher" --id "train_B_seed${seed}" -- "$python" -m scripts.train_segment_clearance --seed "$seed"
done
for split in old_dev dev32; do
  for seed in 0 1 2; do
    for arm in A B; do
      bash "$launcher" --id "eval_${split}_${arm}_seed${seed}" -- "$python" -m scripts.evaluate_segment_clearance --split "$split" --seed "$seed" --arm "$arm"
    done
  done
done
