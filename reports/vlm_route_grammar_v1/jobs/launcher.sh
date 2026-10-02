#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV=0eeeecbe3d02adac413083a9edc0df674aa53be2
SOURCE="$P/research_v2/releases/$REV"
JOBS="$P/runs/vlm_route_grammar_v1"
test ! -e "$JOBS"
mkdir "$JOBS"
cp "${BASH_SOURCE[0]}" "$JOBS/launcher.sh"
sha256sum "$JOBS/launcher.sh" > "$JOBS/launcher.sha256"
cd "$SOURCE"
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sha256sum routeset/vlm_route_grammar.py routeset/vlm_sft_train_probe.py scripts/preflight_vlm_route_grammar.py scripts/evaluate_vlm_route_sft_constrained.py routeset/vlm_sft_data.py routeset/vlm_route_serialization.py configs/vlm_depth_interface_train8_v1.json > "$JOBS/source.sha256"
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest tests/test_vlm_route_grammar.py tests/test_vlm_route_grammar_torch.py tests/test_vlm_sft_train_probe.py -q
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id tokenizer_cpu --resume-strategy none -- "$P/.venv-qwen/bin/python" -m scripts.preflight_vlm_route_grammar --training "$P/runs/vlm_route_sft_v1/seed0" --output "$JOBS/tokenizer_cpu"
test "$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)" = GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id train8_k1 --resume-strategy none -- "$P/.venv-qwen/bin/python" -m scripts.evaluate_vlm_route_sft_constrained --checkpoint "$P/runs/vlm_route_sft_v1/seed0/best.pt" --scope train8_preflight --seed 0 --repeats 1 --output "$JOBS/train8_k1"
