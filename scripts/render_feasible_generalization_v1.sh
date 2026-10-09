#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
export PYTHONPATH="$PWD:$PWD/scripts" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
export COPPELIASIM_ROOT="$P/.observation-deps/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}" QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
D=$(mktemp -d "$P/runs/feasible_space_v1/generalization_display.XXXXXX")
export XDG_RUNTIME_DIR="$D/xdg"; mkdir "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
XVFB="$P/.observation-deps/xorg/usr/bin/Xvfb"
[[ -x "$XVFB" ]] || XVFB=$(command -v Xvfb)
taskset -c 0-3 "$XVFB" -displayfd 3 -screen 0 640x480x24 -nolisten tcp 3>"$D/displaynum" >"$D/xvfb.log" 2>&1 &
OWN_XVFB=$!
cleanup() { kill "$OWN_XVFB" 2>/dev/null || true; wait "$OWN_XVFB" 2>/dev/null || true; }
trap cleanup EXIT
for attempt in {1..40}; do [[ -s "$D/displaynum" ]] && break; sleep .25; done
export DISPLAY=":$(cat "$D/displaynum")"
exec_status=0
taskset -c 0-3 "$P/.venv/bin/python" -m research_feasible_space_v1.generalization_data collect || exec_status=$?
exit "$exec_status"
