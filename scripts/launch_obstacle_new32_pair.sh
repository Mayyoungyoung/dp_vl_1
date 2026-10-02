#!/usr/bin/env bash
# Prepared only: the research lead schedules this on the authorized GPU1 slot.
# All model/training source remains fixed to the recorded immutable release.
set -euo pipefail

readonly ROOT=/home/wzy/dpvlm/route_set_v1
readonly COMMIT=9c19288b0e2fb86b6bea42ac23758314134872c0
readonly SOURCE="$ROOT/research_v2/releases/$COMMIT"
readonly DATA="$ROOT/data/obstacle_learning_curve_new32_v1"
readonly RUN="$ROOT/runs/observed_obstacle_new32_v1"
readonly BASE_PY="$ROOT/.venv/bin/python"
readonly QWEN_PY="$ROOT/.venv-qwen/bin/python"
readonly MODEL="$ROOT/data/qwen3-vl-2b-instruct-89644892"

test -f "$SOURCE/scripts/train_observed_geometry.py"
test -f "$SOURCE/scripts/observation_cache_qwen.py"
test -f "$DATA/snapshot_manifest.json"
test -f "$MODEL/provenance.json"
mkdir -p "$RUN"
exec 9>"$RUN/launcher.lock"
flock -n 9 || exit 73
if [[ -e "$RUN/status" ]]; then
    printf '%s\n' 'Refusing duplicate launch; use the recorded per-stage recovery commands.' >&2
    exit 64
fi
if [[ "$(readlink -f "$0")" != "$RUN/frozen_job.sh" ]]; then
    cp -- "$0" "$RUN/frozen_job.sh"
fi
printf '%s\n' "$COMMIT" > "$RUN/code_release"
printf '%s\n' "$$" > "$RUN/pid"
date -u +%FT%TZ > "$RUN/started_at"
printf '%s\n' running > "$RUN/status"
finish() {
    local code="$1"
    printf '%s\n' "$code" > "$RUN/exit_code"
    date -u +%FT%TZ > "$RUN/finished_at"
    if [[ "$code" -eq 0 ]]; then printf '%s\n' completed > "$RUN/status";
    else printf '%s\n' failed > "$RUN/status"; fi
}
trap 'finish "$?"' EXIT

export CUDA_VISIBLE_DEVICES=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CODE_COMMIT="$COMMIT" PYTHONPATH="$SOURCE"
export RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
cd "$SOURCE"
sha256sum "$RUN/frozen_job.sh" scripts/observation_cache_qwen.py scripts/train_observed_geometry.py \
    routeset/observed_geometry.py routeset/observed_route_head.py routeset/train_v2.py \
    "$DATA/observations.jsonl" "$DATA/supervision.jsonl" > "$RUN/source_input_sha256.txt"

# Resume metadata for training is recorded by record_job. Cache extraction
# validates and reuses existing immutable entries when its exact command is
# rerun after confirming the prior process has exited; no model training cache
# is reused after a backbone/LoRA update.
cat > "$RUN/RESUME.txt" <<'TEXT'
Read qwen_cache/plain_seed0/aux_seed0.status.json and logs before recovery.
Never restart a live PID or remove a live stage lock. The launcher refuses a
second full run. For failed training, execute the status file's resume_command
from its recorded cwd with the same CUDA_VISIBLE_DEVICES=1 and CPU1 environment.
For an interrupted Qwen cache, rerun that status file's exact command after the
old process exits; the frozen-cache script checks configuration and image hash.
After successful recovery continue only the missing stage, preserving logs.
TEXT

"$BASE_PY" scripts/record_job.py --output "$RUN" --run-id qwen_cache --resume-strategy none -- \
    "$QWEN_PY" -m scripts.observation_cache_qwen --model "$MODEL" \
    --manifest "$DATA/observations.jsonl" --output "$DATA/qwen_cache" \
    --device cuda --threads 1 --gpu-memory-fraction 0.35 --max-pixels 262144

"$BASE_PY" - "$DATA" <<'PY'
import hashlib,json,sys
from pathlib import Path
data=Path(sys.argv[1]); cache=data/'qwen_cache'
observations=[json.loads(line) for line in (data/'observations.jsonl').read_text().splitlines() if line.strip()]
records=[json.loads(line) for line in (cache/'samples.jsonl').read_text().splitlines() if line.strip()]
status=json.loads((cache/'status.json').read_text())
config=json.loads((cache/'cache_config.json').read_text())
assert status['status']=='frozen_rgb_language_feature_extraction_complete'
assert status['samples']==len(observations)==108
assert len(records)==len({r['id'] for r in records})==len(observations)
assert {r['id'] for r in records}=={r['id'] for r in observations}
assert config['manifest_sha256']==hashlib.sha256((data/'observations.jsonl').read_bytes()).hexdigest()
assert config['model_trainable_parameter_count']==0
for row in records:
    assert hashlib.sha256((cache/row['file']).read_bytes()).hexdigest()==row['sha256']
print(json.dumps({'cache_manifest_hash_verified':True,'samples':len(records)}),flush=True)
PY

for variant in plain aux; do
    weight=0.0
    if [[ "$variant" == aux ]]; then weight=0.02; fi
    "$BASE_PY" scripts/record_job.py --output "$RUN" --run-id "${variant}_seed0" -- \
        "$BASE_PY" -m scripts.train_observed_geometry \
        --observations "$DATA/observations.jsonl" --supervision "$DATA/supervision.jsonl" \
        --cache-dir "$DATA/qwen_cache" --output "$RUN/${variant}_seed0" \
        --steps 1000 --batch-size 32 --candidates 4 --horizon 24 --width 128 --depth 2 \
        --pooling both --geometry-pooling spatial --point-width 64 --pixel-stride 2 \
        --endpoint-residual-bound 0.05 --grounding-weight "$weight" --grounding-sigma 0.025 \
        --event-scale 0.2 --lr 0.0003 --seed 0 --eval-every 250 --threads 1 --device cuda
done
