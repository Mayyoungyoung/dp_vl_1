#!/usr/bin/env bash
# Root-reviewed immutable source; four sequential runs, never a parallel GPU queue.
set -euo pipefail
P=/home/wzy/dpvlm/route_set_v1
REVISION="${1:?Full immutable reviewed training revision required}"
CONFIG="${2:?Reviewed replication configuration path required}"
[[ "$REVISION" =~ ^[0-9a-f]{40}$ ]]
SRC="$P/research_v2/releases/$REVISION"
cd "$SRC"
export CUDA_VISIBLE_DEVICES=1 RESEARCH_GPU_UUID=GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CODE_COMMIT="$REVISION" PYTHONUNBUFFERED=1
export REPLICATION_CONFIG="$CONFIG" REPLICATION_LAUNCHER="$(readlink -f "${BASH_SOURCE[0]}")"
"$P/.venv/bin/python" - <<'PY'
import hashlib,json,os,subprocess,sys
from pathlib import Path
from routeset.observed_multitask import check_multitask_model_gate
from scripts.analyze_multitask_landmark_pair import analyze
p=Path('/home/wzy/dpvlm/route_set_v1');src=Path.cwd();cp=Path(os.environ['REPLICATION_CONFIG']).resolve()
config=json.loads(cp.read_text());assert config['protocol']=='multitask_landmark_three_seed_replication_v1'
assert src.name==os.environ['CODE_COMMIT'] and config['seeds']==[1,2]
assert config['output_family']=='observed_multitask_landmark_replication_v1'
assert config['arms']=={'ordinary':{'grounding_weight':0.,'grounding_target':'endpoint'},'event_supported':{'grounding_weight':.02,'grounding_target':'event_supported'}}
sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
d=p/'data'/config['dataset'];out=p/'runs'/config['output_family'];out.mkdir(exist_ok=True)
for filename,key in [('snapshot_manifest.json','snapshot_manifest_sha256'),('observations.jsonl','observations_sha256'),('supervision.jsonl','supervision_sha256')]:
    assert sha(d/filename)==config[key],filename
gate=check_multitask_model_gate(d/'observations.jsonl',d/'supervision.jsonl')
assert gate['collection_denominators']['successful_reference_routes']==321
cache=json.loads((d/'qwen_cache/cache_config.json').read_text())
assert cache['revision']==cache['processor']=='89644892e4d85e24eaac8bacfd4f463576704203'
assert cache['manifest_sha256']==config['observations_sha256'] and cache['model_trainable_parameter_count']==0
old=p/'research_v2/releases'/config['seed0_auxiliary_source']
common=('routeset/observed_geometry.py','routeset/observed_route_head.py','routeset/observed_multitask.py',
    'routeset/observed_grounding_targets.py','routeset/common.py','routeset/models.py','routeset/train_v2.py','scripts/train_observed_routes.py')
assert all(sha(src/name)==sha(old/name) for name in common),'No model/target/sampler implementation change authorized'
args=config['shared_training_arguments']
assert (args['steps'],args['batch_size'],args['candidates'],args['eval_every'])==(1500,32,4,250)
assert args['sample_stream_audit'] is True
baseline=json.loads((Path(config['seed0_controls']['ordinary'])/'config.json').read_text())
for key,value in args.items():
    if key=='sample_stream_audit':continue
    assert baseline['checkpoint_selection' if key=='selection_metric' else key]==value,key
provenance=dict(config_sha256=sha(cp),source_commit=src.name,
    sources={str(path):sha(path) for path in [Path(os.environ['REPLICATION_LAUNCHER']),cp]+[src/name for name in common]+[src/'scripts/train_observed_geometry.py',src/'routeset/observed_training_audit.py',src/'scripts/analyze_multitask_landmark_pair.py']})
receipt=out/'source_hashes.json'
if receipt.exists():assert json.loads(receipt.read_text())==provenance,'Resume must preserve exact source/config/launcher'
else:receipt.write_text(json.dumps(provenance,indent=2)+'\n')
(out/'preregistered_replication.json').write_text(json.dumps(config,indent=2)+'\n')
(out/'current_model_use_gate.json').write_text(json.dumps(gate,indent=2)+'\n')
def options(values):
    result=[]
    for name,value in values.items():
        flag='--'+name.replace('_','-')
        if isinstance(value,bool):
            if value:result.append(flag)
        else:result.extend([flag,str(value)])
    return result
def checked_complete(run,expected):
    actual=json.loads((run/'config.json').read_text())
    assert actual['code_commit']==src.name and actual['dataset_fingerprint']==baseline['dataset_fingerprint']
    for name,value in expected.items():assert actual['checkpoint_selection' if name=='selection_metric' else name]==value,name
    summary=json.loads((run/'summary.json').read_text());assert summary['last_step']==1500
    for name,field in [('best.pt','best_checkpoint_sha256'),('last.pt','last_checkpoint_sha256'),('dev_model/predictions.npz','prediction_sha256'),('last_dev_model/predictions.npz','last_prediction_sha256')]:assert sha(run/name)==summary[field],name
    audit=summary['sample_stream_audit'];assert audit['batches']==1500 and audit['observation_draws']==48000
    if expected['grounding_target']=='event_supported':assert summary['grounding_target_fingerprint']==config['grounding_target_fingerprint']
for seed in config['seeds']:
    for arm,overrides in config['arms'].items():
        name=arm+'_seed'+str(seed);run=out/name;values=dict(args,**overrides,seed=seed)
        status=run/'status.json'
        if status.exists() and json.loads(status.read_text()).get('status')=='completed':
            checked_complete(run,values);print('Already completed and verified: '+name,flush=True);continue
        assert not (run/'active.lock').exists(),'Inspect active run before attempting resume'
        command=[sys.executable,'-m','scripts.train_observed_geometry','--observations',str(d/'observations.jsonl'),
            '--supervision',str(d/'supervision.jsonl'),'--cache-dir',str(d/'qwen_cache'),'--output',str(run),
            '--multitask-snapshot-manifest',str(d/'snapshot_manifest.json')]+options(values)
        if (run/'last.pt').exists():command.append('--resume')
        print('Starting/resuming '+name+' from '+src.name,flush=True)
        subprocess.run([sys.executable,'-m','scripts.record_job','--output',str(out),'--run-id',name,'--resume-strategy','flag','--']+command,check=True)
        checked_complete(run,values)
    result=analyze(out/('ordinary_seed'+str(seed)),out/('event_supported_seed'+str(seed)))
    (out/('paired_seed'+str(seed)+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('Verified actual same initialization, sample stream and exposure: seed '+str(seed),flush=True)
print('All four fixed replication runs completed. Retain every best/last/task/parent result.',flush=True)
PY
