#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
export CODE_COMMIT="$(basename "$PWD")" PYTHONPATH="$PWD:$PWD/scripts" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
export COPPELIASIM_ROOT="$P/.observation-deps/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}" QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
RUN="$P/runs/research_v3_v1/paired_score_render_v1"
mkdir -p "$RUN"
DISPLAY_ROOT=$(mktemp -d "$RUN/display.XXXXXX")
export XDG_RUNTIME_DIR="$DISPLAY_ROOT/xdg"
mkdir "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
XVFB="$P/.observation-deps/xorg/usr/bin/Xvfb"
[[ -x "$XVFB" ]] || XVFB=$(command -v Xvfb)
taskset -c 0-3 "$XVFB" -displayfd 3 -screen 0 640x480x24 -nolisten tcp 3>"$DISPLAY_ROOT/displaynum" >"$DISPLAY_ROOT/xvfb.log" 2>&1 &
OWN_XVFB=$!
cleanup() { kill "$OWN_XVFB" 2>/dev/null || true; wait "$OWN_XVFB" 2>/dev/null || true; }
trap cleanup EXIT
for attempt in {1..40}; do [[ -s "$DISPLAY_ROOT/displaynum" ]] && break; sleep .25; done
export DISPLAY=":$(cat "$DISPLAY_ROOT/displaynum")"
taskset -c 0-3 "$P/.venv/bin/python" -m scripts.research_v3_score_data "$@"
