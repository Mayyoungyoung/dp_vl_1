#!/usr/bin/env bash
set -euo pipefail
stage=${1:?Pass assets or environment; stages never chain automatically}
case "$stage" in assets|environment) ;; *) exit 2 ;; esac
resume=${2:-}
if [[ -n "$resume" && "$resume" != --resume ]]; then exit 2; fi
release=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
root=/home/wzy/dpvlm/route_set_v1
revision=$(basename -- "$release")
[[ "$revision" =~ ^[0-9a-f]{40}$ ]] || { echo 'Use an immutable release directory'; exit 2; }
export CODE_COMMIT="$revision" CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HUB_DISABLE_XET=1 HF_HUB_DISABLE_TELEMETRY=1
export PYTHONPATH="$release"
run="$root/runs/hamster3d_preparation_v1"
stamp=$(date -u +%Y%m%dT%H%M%SZ)_${stage}_$$
session="$run/sessions/$stamp"
mkdir -p "$run/sessions"
wrapper_record="$run/sessions/${stamp}.wrapper.json"
python="$root/.venv-qwen/bin/python"
"$python" -c 'import hashlib,json,pathlib,sys; p=pathlib.Path(sys.argv[1]); pathlib.Path(sys.argv[2]).write_text(json.dumps({"launcher":str(p),"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"release":sys.argv[3]},indent=2)+"\n")' "$release/scripts/launch_hamster3d_preparation_v1.sh" "$wrapper_record" "$release"
args=()
if [[ -n "$resume" ]]; then args+=(--resume); fi
exec taskset -c 0 "$python" "$release/scripts/record_job.py" --output "$run/sessions" --run-id "$stamp" --resume-strategy none -- \
  "$python" "$release/scripts/prepare_hamster3d_external.py" \
  --stage "$stage" --manifest "$release/configs/hamster3d_assets_v1.json" \
  --requirements "$release/configs/hamster3d_environment_v1.txt" \
  --root "$root" --session "$session" "${args[@]}"
