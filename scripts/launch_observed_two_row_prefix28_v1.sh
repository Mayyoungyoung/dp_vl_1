#!/usr/bin/env bash
# Run only after the frozen CPU preparation and TRAIN capacity review pass.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Reviewed immutable source commit required}"
SRC="$P/research_v2/releases/$REVISION"
cd "$SRC"
export CODE_COMMIT="$REVISION" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export TWO_ROW_LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
"$P/.venv/bin/python" - <<'PY'
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
from routeset.common import sha256,write_json
from scripts.export_two_row_observations import verify_export,INPUT_KEYS
from scripts.observation_cache_qwen import REVISION

p=Path('/home/wzy/dpvlm/route_set_v1');src=Path.cwd()
assert src.name==os.environ['CODE_COMMIT'] and len(src.name)==40
assert len(os.sched_getaffinity(0))==1,'Launch with taskset on exactly one authorized CPU.'
gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
assert gpu==os.environ['RESEARCH_GPU_UUID']
d=p/'data/observation_two_row_prefix28_v1'
quality=p/'runs/observation_two_row_prefix28_preparation_v1/train_quality.json'
selection=src/'configs/observed_two_row_prefix28_selection_v1.json'
out=p/'runs/observed_two_row_prefix28_v1';cache=d/'qwen_cache'
manifest,gate=verify_export(d)
assert manifest['selection']==json.loads(selection.read_text())
q=json.loads(quality.read_text())
assert q['protocol']=='two_row_train_endpoint_capacity_and_reference_quality_v1'
assert q['capacity_gate_passed'] and not q['dev_raw_arrays_opened'] and not q['locked_raw_opened']
assert q['source_export_manifest_sha256']==sha256(d/'export_manifest.json')
rows=[json.loads(x) for x in (d/'observations.jsonl').read_text().splitlines()]
labels=[json.loads(x) for x in (d/'supervision.jsonl').read_text().splitlines()]
train=[r for r in labels if r['split']=='TRAIN']
assert q['train_input_ids']==sorted(r['id'] for r in train)
assert q['train_reference_counts']=={r['id']:len(r['routes']) for r in train}
assert 0<len(rows)<=84 and len(rows)==manifest['actual_inputs']
assert all(set(r)==INPUT_KEYS and r['split'] in ('TRAIN','DEV_MODEL') for r in rows)
assert any(r['split']=='DEV_MODEL' for r in rows) and any(r['routes'] for r in train)
for path,digest in q['source_files_sha256'].items():
    assert manifest['source_files_sha256'].get(path)==digest==sha256(path)
assert not out.exists() and not cache.exists(),'Fresh pipeline only. Inspect failed stages and use the exact recorded single-stage recovery command.'
out.mkdir(parents=True)
write_json(out/'pre_cache_gate.json',gate)
write_json(out/'preregistered_selection.json',manifest['selection'])
sources=[Path(os.environ['TWO_ROW_LAUNCHER']),selection,quality,d/'export_manifest.json',
    src/'scripts/train_observed_two_row.py',src/'scripts/train_observed_geometry.py',
    src/'scripts/train_observed_routes.py',src/'scripts/export_two_row_observations.py',
    src/'scripts/evaluate_observed_two_row.py',src/'scripts/evaluate_observed_obstacles.py',
    src/'scripts/observation_cache_qwen.py',src/'scripts/record_job.py',
    src/'routeset/observed_geometry.py',src/'routeset/observed_route_head.py',
    src/'routeset/observed_multitask.py',src/'routeset/observed_training_audit.py']
write_json(out/'source_hashes.json',{str(path):sha256(path) for path in sources})
write_json(out/'runtime.json',dict(code_commit=src.name,cpu_affinity=sorted(os.sched_getaffinity(0)),
    omp_threads=1,gpu_uuid=gpu,gpu_memory_fraction=.35,
    requested_parents=28,requested_inputs=84,actual_inputs=len(rows),
    source_export_manifest_sha256=sha256(d/'export_manifest.json'),
    quality_audit_sha256=sha256(quality),
    recovery='Never replay the whole launcher. Preserve failures; cache supports identical-config restart, training supports the recorded --resume command.',
    latency_scope='Cache and training cost are separate; cached head latency is not Qwen end-to-end request latency.'))
def record(run_id,python,module,arguments,resume):
    subprocess.run([sys.executable,'-m','scripts.record_job','--output',str(out),'--run-id',run_id,
        '--resume-strategy',resume,'--',str(p/python),'-m',module]+arguments,check=True)
record('qwen_cache','.venv-qwen/bin/python','scripts.observation_cache_qwen',[
    '--model',str(p/'data/qwen3-vl-2b-instruct-89644892'),'--manifest',str(d/'observations.jsonl'),
    '--output',str(cache),'--device','cuda','--threads','1','--gpu-memory-fraction','0.35',
    '--max-pixels','262144'],'none')
c=json.loads((cache/'cache_config.json').read_text());status=json.loads((cache/'status.json').read_text())
assert c['model']=='Qwen/Qwen3-VL-2B-Instruct' and c['revision']==c['processor']==REVISION
assert c['max_pixels']==262144 and c['model_trainable_parameter_count']==0 and set(c['input_contract'])==INPUT_KEYS
assert c['manifest_sha256']==sha256(d/'observations.jsonl') and c['transformers']=='4.57.1'
assert status['samples']==len(rows) and status['status']=='frozen_rgb_language_feature_extraction_complete'
records=[json.loads(x) for x in (cache/'samples.jsonl').read_text().splitlines()]
assert len(records)==len(rows) and {r['id'] for r in records}=={r['id'] for r in rows}
for r in records:
    assert sha256(cache/r['file'])==r['sha256']
    with np.load(cache/r['file'],allow_pickle=False) as a:
        assert a['mean_hidden'].shape==a['last_hidden'].shape==(2048,)
        assert np.isfinite(a['mean_hidden']).all() and np.isfinite(a['last_hidden']).all()
write_json(out/'actual_cache_receipt.json',dict(samples=len(rows),cache_config_sha256=sha256(cache/'cache_config.json'),
    samples_sha256=sha256(cache/'samples.jsonl'),pooling='both',pooled_feature_dimension=4096,
    cache_status=status,scope='One real RGB+language Qwen encoding per actual input; all requests and load cost retained.'))
verify_export(d)
record('peak_seed0','.venv/bin/python','scripts.train_observed_two_row',[
    '--data',str(d),'--selection',str(selection),'--quality-audit',str(quality),
    '--output',str(out/'peak_seed0'),'--device','cuda'],'flag')
verify_export(d)
print('Fixed prefix28 ordinary baseline completed. Preserve best, last1500, full evaluation pools and all denominators.',flush=True)
PY
