#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
P=/home/wzy/dpvlm/route_set_v1
export PYTHONPATH="$PWD:$PWD/scripts" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export LC_ALL=C LANG=C PYTHONUTF8=1
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 LP_NUM_THREADS=1
export COPPELIASIM_ROOT="$P/.observation-deps/CoppeliaSim"
export LD_LIBRARY_PATH="$COPPELIASIM_ROOT:${LD_LIBRARY_PATH:-}" QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
export LIBGL_ALWAYS_SOFTWARE=1 MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
D=$(mktemp -d "$P/runs/selective_repair_v1/display.body.XXXXXX")
CPU_SET="${SELECTIVE_BODY_CPU_SET:-0-3}"
[[ "$CPU_SET" == 0-3 || "$CPU_SET" == 0 || "$CPU_SET" == 1 || "$CPU_SET" == 2 || "$CPU_SET" == 3 ]] || exit 1
export XDG_RUNTIME_DIR="$D/xdg"; mkdir "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"
XVFB="$P/.observation-deps/xorg/usr/bin/Xvfb"
[[ -x "$XVFB" ]] || XVFB=$(command -v Xvfb)
taskset -c "$CPU_SET" "$XVFB" -displayfd 3 -screen 0 640x480x24 -nolisten tcp 3>"$D/displaynum" >"$D/xvfb.log" 2>&1 &
OWN_XVFB=$!
cleanup() { kill "$OWN_XVFB" 2>/dev/null || true; wait "$OWN_XVFB" 2>/dev/null || true; }
trap cleanup EXIT
for attempt in {1..40}; do [[ -s "$D/displaynum" ]] && break; sleep .25; done
export DISPLAY=":$(cat "$D/displaynum")"
if [[ "${1:-}" == execution ]];then
 shift
 taskset -c "$CPU_SET" "$P/.venv-sim/bin/python" -m research_selective_repair_v1.body_execution_teacher "$@"
elif [[ "${1:-}" == suite ]];then
 shift
 taskset -c "$CPU_SET" "$P/.venv-sim/bin/python" -m research_selective_repair_v1.body_execution_suite "$@"
elif [[ "${1:-}" == amplitude_suite ]];then
 shift
 taskset -c "$CPU_SET" "$P/.venv-sim/bin/python" -m research_selective_repair_v1.amplitude_execution_suite "$@"
elif [[ "${1:-}" == repeat_suite ]];then
 shift
 taskset -c "$CPU_SET" "$P/.venv-sim/bin/python" -m research_selective_repair_v1.body_repeat_suite "$@"
elif [[ "${1:-}" == planner ]];then
 shift
 taskset -c "$CPU_SET" "$P/.venv-sim/bin/python" -m research_selective_repair_v1.planning_teacher "$@"
else
 taskset -c "$CPU_SET" "$P/.venv-sim/bin/python" -m research_selective_repair_v1.body_teacher "$@"
fi
