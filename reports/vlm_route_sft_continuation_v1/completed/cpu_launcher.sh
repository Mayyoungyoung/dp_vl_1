#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV="${1:?Pass the reviewed immutable release commit}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
SOURCE="$P/research_v2/releases/$REV"
JOBS="$P/runs/vlm_route_sft_continuation_v1"
test -d "$SOURCE"
test ! -e "$JOBS"
mkdir "$JOBS"
cp "${BASH_SOURCE[0]}" "$JOBS/cpu_launcher.sh"
sha256sum "$JOBS/cpu_launcher.sh" > "$JOBS/cpu_launcher.sha256"
printf '%s\n' "$$" > "$JOBS/cpu_launcher.pid"
cd "$SOURCE"
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
sha256sum scripts/train_vlm_route_sft.py routeset/vlm_sft_continuation.py routeset/vlm_sft_data.py routeset/vlm_sft_loss.py routeset/vlm_route_serialization.py scripts/train_observed_lora.py routeset/observed_route_head.py routeset/train_v2.py routeset/common.py tests/test_vlm_sft_data.py tests/test_vlm_sft_loss.py tests/test_vlm_sft_training.py tests/test_vlm_sft_continuation.py tests/test_vlm_sft_continuation_torch.py > "$JOBS/cpu_source.sha256"
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id cpu_tests --resume-strategy none -- "$P/.venv/bin/python" -m pytest tests/test_vlm_sft_data.py tests/test_vlm_sft_loss.py tests/test_vlm_sft_training.py tests/test_vlm_sft_continuation.py tests/test_vlm_sft_continuation_torch.py -q
