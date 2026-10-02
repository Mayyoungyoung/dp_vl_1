#!/usr/bin/env bash
set -euo pipefail
root=/home/wzy/dpvlm/route_set_v1
sha=ce548f43ab22e804a8b70dea2f8bf297e20c8b84
release="$root/research_v2/releases/$sha"
run="$root/runs/observed_two_row_formal_train16_analysis_v1"
test ! -e "$run"
mkdir "$run"
cp "${BASH_SOURCE[0]}" "$run/frozen_launcher.sh"
printf '%s\n' "$release" > "$run/source_release"
printf '%s\n' "$$" > "$run/launcher_pid"
date -u +%FT%TZ > "$run/started_at"
finish() {
    code=$?
    printf '%s\n' "$code" > "$run/launcher_exit_code"
    date -u +%FT%TZ > "$run/finished_at"
}
trap finish EXIT
sha256sum "${BASH_SOURCE[0]}" "$release/scripts/analyze_two_row_formal_train.py" "$release/scripts/analyze_observed_two_row_pilot.py" "$release/scripts/collect_two_row_formal.py" "$release/scripts/record_job.py" > "$run/source_sha256.txt"
export CODE_COMMIT="$sha" CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$release:$release/scripts"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
cd "$release"
taskset -c 1 "$root/.venv/bin/python" scripts/record_job.py --output "$run" --run-id analysis --resume-strategy fresh-output -- \
    "$root/.venv/bin/python" -m scripts.analyze_two_row_formal_train \
    --corpus "$root/data/observed_two_row_formal116_v1" --output "$run/analysis"
