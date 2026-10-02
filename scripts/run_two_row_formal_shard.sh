#!/usr/bin/env bash
set -euo pipefail
sha="${1:?immutable SHA}"; shard="${2:?shard 0 or 1}"; session="${3:?fresh session directory}"; resume="${4:-false}"
[[ "$sha" =~ ^[0-9a-f]{40}$ ]] && [[ "$shard" =~ ^[01]$ ]] || { echo "Invalid immutable SHA or shard" >&2; exit 2; }
root=/home/wzy/dpvlm/route_set_v1
release="$root/research_v2/releases/$sha"
corpus="$root/data/observed_two_row_formal116_v1"
run="$root/runs/observed_two_row_formal116_v1"
work="$session/shard$shard"
test ! -e "$work"
mkdir "$work"
cp "${BASH_SOURCE[0]}" "$work/frozen_worker_wrapper.sh"
export CODE_COMMIT="$sha" CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$release:$release/scripts:${PYTHONPATH:-}"
export COPPELIASIM_ROOT="$root/.observation-deps/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}"
export QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe LP_NUM_THREADS=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_RUNTIME_DIR="$work/xdg"
mkdir "$XDG_RUNTIME_DIR"
chmod 700 "$XDG_RUNTIME_DIR"
xvfb_pid=''
cleanup() {
    code=$?
    printf '%s\n' "$code" > "$work/exit_code"
    if [ -n "$xvfb_pid" ] && [ "$(ps -o ppid= -p "$xvfb_pid" 2>/dev/null | tr -d ' ')" = "$$" ]; then kill "$xvfb_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT
"$root/.observation-deps/xorg/usr/bin/Xvfb" -displayfd 3 -screen 0 640x480x24 -nolisten tcp 3>"$work/displaynum" >"$work/xvfb.log" 2>&1 &
xvfb_pid=$!
echo "$xvfb_pid" > "$work/xvfb_pid"
for attempt in $(seq 1 40); do [ -s "$work/displaynum" ] && break; sleep .25; done
test -s "$work/displaynum"
export DISPLAY=":$(cat "$work/displaynum")"
cd "$release"
args=()
if [ "$resume" = true ]; then args+=(--resume); fi
"$root/.venv/bin/python" scripts/record_job.py --output "$work" --run-id coordinator --resume-strategy none -- \
    "$root/.venv-sim/bin/python" -u scripts/collect_two_row_formal.py shard \
    --corpus "$corpus" --run-root "$run/parents/shard$shard" --shard "$shard" \
    --sim-python "$root/.venv-sim/bin/python" "${args[@]}"
