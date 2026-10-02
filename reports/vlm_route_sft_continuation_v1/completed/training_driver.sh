#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV="${1:?Immutable release required}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
test "$#" -le 2
cd "$P/research_v2/releases/$REV"
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
test "$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)" = "$RESEARCH_GPU_UUID"
RESTORE=(--continue-from "$P/runs/vlm_route_sft_v1/seed0/last.pt")
if [ "${2:-}" = --resume ]; then
  RESTORE=(--resume)
elif [ "$#" -ne 1 ]; then
  exit 2
fi
exec "$P/.venv-qwen/bin/python" -m scripts.train_vlm_route_sft --observations "$P/data/observation_obstacle_reserved_development_v1/observations.jsonl" --supervision "$P/data/observation_obstacle_reserved_development_v1/supervision.jsonl" --model "$P/data/qwen3-vl-2b-instruct-89644892" --output "$P/runs/vlm_route_sft_continuation_v1/seed0" --steps 6000 --seed 0 --dev-plan-seed 200000 --lr 0.0001 --weight-decay 0 --batch-size 1 --eval-every 250 --checkpoint-every 25 --log-every 25 --chunk-size 64 "${RESTORE[@]}"
