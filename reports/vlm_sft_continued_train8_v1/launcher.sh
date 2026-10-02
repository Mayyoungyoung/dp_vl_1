#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV="${1:?Reviewed immutable inference release required}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
SOURCE="$P/research_v2/releases/$REV"
JOBS="$P/runs/vlm_sft_continued_train8_v1"
test ! -e "$JOBS"
mkdir "$JOBS"
cp "${BASH_SOURCE[0]}" "$JOBS/launcher.sh"
sha256sum "$JOBS/launcher.sh" > "$JOBS/launcher.sha256"
printf '%s\n' "$$" > "$JOBS/launcher.pid"
cd "$SOURCE"
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sha256sum scripts/evaluate_vlm_sft_continued_train8.py scripts/analyze_vlm_sft_continued_train8.py routeset/vlm_sft_continued_eval.py routeset/vlm_sft_continuation.py routeset/vlm_sft_train_probe.py routeset/vlm_route_grammar.py routeset/vlm_sft_evaluation.py routeset/vlm_sft_data.py routeset/vlm_route_serialization.py scripts/train_observed_lora.py tests/test_vlm_sft_continued_eval.py tests/test_vlm_sft_greedy.py tests/test_vlm_sft_train_probe.py configs/vlm_depth_interface_train8_v1.json > "$JOBS/source.sha256"
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id cpu_tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest -q tests/test_vlm_sft_continued_eval.py tests/test_vlm_sft_greedy.py tests/test_vlm_sft_train_probe.py
# This check opens no observations or model weights and refuses an inherited best.
"$P/.venv/bin/python" - "$P/runs/vlm_route_sft_continuation_v1/seed0" "$JOBS/lineage_preflight.json" <<'PY'
import json,sys
from pathlib import Path
from routeset.vlm_sft_continued_eval import inspect_completed_run
training,config,lineage,index,receipt=inspect_completed_run(Path(sys.argv[1]))
Path(sys.argv[2]).write_text(json.dumps(receipt,indent=2)+'\n')
print('completed6000_lineage_preflight=passed',flush=True)
PY
test "$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)" = GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id train8 --resume-strategy none -- "$P/.venv-qwen/bin/python" -m scripts.evaluate_vlm_sft_continued_train8 --run "$P/runs/vlm_route_sft_continuation_v1/seed0" --output "$JOBS/train8"
export CUDA_VISIBLE_DEVICES=-1
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id analyze --resume-strategy none -- "$P/.venv/bin/python" -m scripts.analyze_vlm_sft_continued_train8 --run "$P/runs/vlm_route_sft_continuation_v1/seed0" --original-greedy "$P/runs/vlm_route_greedy_train8_v1/train8" --continued-greedy "$JOBS/train8" --supervision-metadata "$P/data/observation_obstacle_reserved_development_v1/supervision.jsonl" --output "$JOBS/analysis"
