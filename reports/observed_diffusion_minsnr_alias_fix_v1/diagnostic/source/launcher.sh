#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/wzy/dpvlm/route_set_v1
REV=75586c573d8d67b3a3a0d33ee2a027538a1bacad
SRC="$ROOT/research_v2/releases/$REV"
OUT="$ROOT/runs/observed_diffusion_minsnr_alias_diagnostic_v1"
SCRIPT="$ROOT/research_v2/incoming/minsnr_alias_diagnostic_75586c5_v1.py"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1 CODE_COMMIT="$REV" PYTHONPATH="$SRC:$SRC/scripts"
test ! -e "$OUT"
mkdir -p "$OUT/source"
cp -- "$0" "$OUT/source/launcher.sh"
cp -- "$SCRIPT" "$OUT/source/diagnostic.py"
sha256sum "$0" "$SCRIPT" "$SRC/tests/test_observed_diffusion_minsnr_probe.py" "$SRC/scripts/train_observed_diffusion_minsnr_probe.py" > "$OUT/source/source_sha256.txt"
cd "$SRC"
exec "$ROOT/.venv/bin/python" -m scripts.record_job --output "$OUT" --run-id diagnostic --resume-strategy none -- "$ROOT/.venv/bin/python" "$SCRIPT" "$OUT/analysis"
