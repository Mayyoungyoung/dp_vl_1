#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV="${1:?Pass the same reviewed immutable release used by successful CPU tests}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
SOURCE="$P/research_v2/releases/$REV"
JOBS="$P/runs/vlm_route_sft_continuation_v1"
ORIGINAL="$P/runs/vlm_route_sft_v1/seed0"
test -d "$JOBS"
test ! -e "$JOBS/seed0"
test ! -e "$JOBS/gpu_launcher.sh"
cp "${BASH_SOURCE[0]}" "$JOBS/gpu_launcher.sh"
sha256sum "$JOBS/gpu_launcher.sh" > "$JOBS/gpu_launcher.sha256"
printf '%s\n' "$$" > "$JOBS/gpu_launcher.pid"
cd "$SOURCE"
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=-1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$P/.venv/bin/python" - "$JOBS/cpu_tests.status.json" "$REV" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
assert s['status']=='completed' and s['exit_code']==0 and s['code_commit']==sys.argv[2], s
print('actual_same_release_cpu_tests=passed',flush=True)
PY
sha256sum -c "$JOBS/cpu_source.sha256" > "$JOBS/gpu_source_preflight.log"
"$P/.venv/bin/python" - "$ORIGINAL" <<'PY'
import hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);summary=json.loads((root/'summary.json').read_text());config=json.loads((root/'config.json').read_text())
assert summary['status']=='completed' and summary['step']==1500 and config['steps']==1500
assert summary['artifacts_sha256']['last.pt']=='d604b5a213bdf281e7976b460b7ceb2fc428488610b1b84670ddca04711cb00c'
assert hashlib.sha256((root/'last.pt').read_bytes()).hexdigest()==summary['artifacts_sha256']['last.pt']
assert config['lr']==.0001 and config['batch_size']==1 and config['eval_every']==250
assert config['seed']==0 and config['dev_plan_seed']==200000 and config['scheduler']=='constant'
assert config['data_fingerprint']=='40f33d663944402247667c425424b99474afaaf1234141910f5c3189e7c50da0'
print('actual_original1500_source=passed',flush=True)
PY
sha256sum "$ORIGINAL/last.pt" "$ORIGINAL/best.pt" "$ORIGINAL/initial_adapters.pt" "$ORIGINAL/config.json" "$ORIGINAL/summary.json" "$ORIGINAL/requests.jsonl" "$ORIGINAL/data_source_hashes.json" "$ORIGINAL/dataset_accounting.json" "$ORIGINAL/dev_plan.json" > "$JOBS/original_source.sha256"
test "$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)" = GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
# record_job's supported flag strategy appends --resume. A frozen driver
# switches modes so this never combines --resume with --continue-from.
cat > "$JOBS/training_driver.sh" <<'DRIVER'
#!/usr/bin/env bash
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REV="${1:?Immutable release required}"
[[ "$REV" =~ ^[0-9a-f]{40}$ ]]
test "$#" -le 2
cd "$P/research_v2/releases/$REV"
export CODE_COMMIT="$REV" PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
test "$(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader)" = "$RESEARCH_GPU_UUID"
RESTORE=(--continue-from "$P/runs/vlm_route_sft_v1/seed0/last.pt")
if [ "${2:-}" = --resume ]; then
  RESTORE=(--resume)
elif [ "$#" -ne 1 ]; then
  exit 2
fi
exec "$P/.venv-qwen/bin/python" -m scripts.train_vlm_route_sft --observations "$P/data/observation_obstacle_reserved_development_v1/observations.jsonl" --supervision "$P/data/observation_obstacle_reserved_development_v1/supervision.jsonl" --model "$P/data/qwen3-vl-2b-instruct-89644892" --output "$P/runs/vlm_route_sft_continuation_v1/seed0" --steps 6000 --seed 0 --dev-plan-seed 200000 --lr 0.0001 --weight-decay 0 --batch-size 1 --eval-every 250 --checkpoint-every 25 --log-every 25 --chunk-size 64 "${RESTORE[@]}"
DRIVER
sha256sum "$JOBS/training_driver.sh" > "$JOBS/training_driver.sha256"
"$P/.venv/bin/python" -m scripts.record_job --output "$JOBS" --run-id seed0_continue6000 --resume-strategy flag -- bash "$JOBS/training_driver.sh" "$REV"
sha256sum -c "$JOBS/original_source.sha256" > "$JOBS/original_unchanged.log"
