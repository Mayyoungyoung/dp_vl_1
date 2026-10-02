#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
run_id="${1:-observation_rlbench_pilot_20261002}"
shift || true
run="runs/$run_id"
mkdir -p "$run"
exec 9>"$run/lock"
flock -n 9 || exit 73
echo $$ > "$run/pid"
date -u +%FT%TZ > "$run/started_at"
echo running > "$run/status"
collector="${OBSERVATION_COLLECTOR:-scripts/observation_collect_rlbench.py}"
mkdir -p "$run/source"
cp "$collector" scripts/observation_rlbench_job.sh "$run/source/"
if [ "$collector" != scripts/observation_collect_rlbench.py ]; then
    cp scripts/observation_collect_rlbench.py "$run/source/"
fi
sha256sum "$run/source/"* > "$run/source_sha256.txt"
export COPPELIASIM_ROOT="$PWD/.observation-deps/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export CUDA_VISIBLE_DEVICES=""
export LIBGL_ALWAYS_SOFTWARE=1
export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
export LP_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export XDG_RUNTIME_DIR="$PWD/$run/xdg"
mkdir -p "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
.observation-deps/xorg/usr/bin/Xvfb -displayfd 3 -screen 0 640x480x24 -nolisten tcp 3>"$run/displaynum" >"$run/xvfb.log" 2>&1 &
xvfb_pid=$!
echo "$xvfb_pid" > "$run/xvfb_pid"
trap 'kill "$xvfb_pid" 2>/dev/null || true' EXIT
for i in $(seq 1 40); do
    [ -s "$run/displaynum" ] && break
    sleep 0.25
done
export DISPLAY=":$(cat "$run/displaynum")"
.venv-sim/bin/python -u "$run/source/$(basename "$collector")" --output "data/$run_id" "$@" > "$run/collection.log" 2>&1
result=$?
echo "$result" > "$run/exit_code"
date -u +%FT%TZ > "$run/finished_at"
if [ "$result" = 0 ]; then echo complete > "$run/status"; else echo failed > "$run/status"; fi
exit "$result"
