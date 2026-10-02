"""Read only registered old DEV and new TRAIN/DEV mechanical summaries."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time

started = time.perf_counter()
source = Path('/home/wzy/dpvlm/route_set_v1/research_v2/releases/5bb9087c9ca511c2a68a4da08798e95e8c6d4cdc')
sys.path.insert(0, str(source))
from routeset.multitask_fingerprints import physical_layout, json_hash, audit_fingerprints

root = Path('/home/wzy/dpvlm/route_set_v1')
old = root/'data/observation_multitask_resume_probe_v1'
new = root/'data/observation_multitask_validated_five_v1'
hashes = {}
def read(path):
    raw = path.read_bytes()
    hashes[str(path)] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)

plans = {str(p): read(p/'partition_manifest.json') for p in (old,new)}
assert hashes[str(old/'partition_manifest.json')] == 'c7fd569dbdfa27b6794e8c941b4115a20fd8eb4fd05125407243fe49bac60126'
assert hashes[str(new/'partition_manifest.json')] == 'b4406619f45cb4ea9c85a3c4479593d97b0c82138d52c782c6a66020f4202cd7'
selected = [s for s in plans[str(old)]['parents'] if s['split']=='DEV_MODEL']
assert len(selected)==12 and all(s['parent_index'] in (16,17) for s in selected)
wanted = {s['parent_id'] for s in selected}
rows, closures = [], []
for p in (old,new):
    for spec in plans[str(p)]['parents']:
        if spec['split'] not in (('DEV_MODEL',) if p==old else ('TRAIN','DEV_MODEL')):
            continue
        folder=p/spec['split']/'parents'/spec['parent_id']
        if p==old:
            closure=read(folder/'closed.json')
            assert not (folder/'worker.lock').exists()
            assert closure['status']=='complete' and closure['completed_attempts']==3 and closure['requested_attempts']==3
            pointer=read(folder/'reference_pointer.json')
            reference=(folder/pointer['directory']/'reference.json').resolve()
            reference.relative_to(folder.resolve())
            metadata=read(reference)
            assert hashes[str(reference)]==pointer['reference_sha256']
            row={k:spec[k] for k in ('parent_id','task','split')}
            row.update(physical_layout_sha256=json_hash(physical_layout(metadata['world'])),
                physical_layout_quantized_sha256=json_hash(physical_layout(metadata['world'],True)),
                rgb_file_sha256=pointer['image_sha256'])
            closures.append(dict(spec,closure=closure,original_mechanical_fingerprint_exists=(folder/'mechanical_fingerprint.json').exists(),
                derivation_scope='Existing physical_layout and json_hash functions on hash-verified DEV reference world only; no original data written'))
        else:
            row=read(folder/'mechanical_fingerprint.json')
            assert all(row[k]==spec[k] for k in ('parent_id','task','split'))
        rows.append(dict(row,source_root=str(p)))
gate=audit_fingerprints(rows)
configs, manifests, usage_hits = [], {}, []
for path in sorted((root/'runs').glob('**/config.json')):
    config=json.loads(path.read_text())
    value=config.get('observations') or config.get('observation_manifest')
    if not isinstance(value,str):continue
    manifest=Path(value)
    if not manifest.is_file() or manifest.suffix!='.jsonl':continue
    configs.append(dict(config=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),observations=str(manifest)))
    if str(manifest) in manifests:continue
    raw=manifest.read_bytes();inputs=[json.loads(line) for line in raw.splitlines() if line.strip()]
    assert all(row.get('split') in ('TRAIN','DEV_MODEL','DEV_SCORE','CALIBRATION','TEST_LOCKED','OOD_LOCKED') for row in inputs)
    # Registry manifests contain only observation metadata; no raw fields or labels opened.
    ids={row['parent_id'] for row in inputs}
    manifests[str(manifest)]=dict(sha256=hashlib.sha256(raw).hexdigest(),examples=len(inputs),parents=len(ids),old_dev_intersection=sorted(ids&wanted))
    usage_hits.extend(dict(manifest=str(manifest),parent_id=parent) for parent in sorted(ids&wanted))
metric_files=[];metric_hits=[]
for family in sorted((root/'runs').iterdir()):
    if not family.is_dir() or not ('multitask' in family.name or family.name.startswith('observation_retrieval')):continue
    for path in sorted(family.glob('**/per_scene.json')):
        raw=path.read_bytes();saved=json.loads(raw)
        data=saved if isinstance(saved,list) else saved.get('rows',saved.get('per_scene',[]))
        if not isinstance(data,list):raise ValueError('Unrecognized per-scene metadata')
        ids={row.get('parent_id') for row in data}
        identities={row.get('scene_id',row.get('id','')) for row in data}
        matches=[parent for parent in sorted(wanted) if parent in ids or any(str(value).startswith(parent+'_lang') for value in identities)]
        metric_files.append(dict(path=str(path),sha256=hashlib.sha256(raw).hexdigest(),rows=len(data)))
        metric_hits.extend(dict(file=str(path),parent_id=parent) for parent in matches)
for path,digest in hashes.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest
out=dict(protocol='legacy_multitask_dev_transfer_eligibility_v1',checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),elapsed_seconds=time.perf_counter()-started,
    source_commit=source.name,existing_helper_sha256=hashlib.sha256((source/'routeset/multitask_fingerprints.py').read_bytes()).hexdigest(),
    selected_requested_parents=selected,closures=closures,mechanical_rows=rows,mechanical_gate=gate,
    model_config_index=configs,model_input_manifests=manifests,model_input_usage_hits=usage_hits,
    per_scene_metric_files=metric_files,per_scene_metric_usage_hits=metric_hits,source_files_sha256=hashes,
    all_source_files_unchanged=True,closed_dev_parents=len(closures),comparison_new_train_dev_parents=len(rows)-len(closures),
    eligible_for_prospective_frozen_transfer=not gate['duplicate_groups'] and not usage_hits and not metric_hits,
    scope='Legacy DEV_MODEL closure counts and reference world only; new TRAIN/DEV saved mechanical metadata only; existing model input IDs and saved model per-scene IDs only. No locked per-parent metadata/raw, RGB/depth/path files opened. Registrations contain role identities only. No model metrics calculated. No original data written.',
    limitation='No detected physical/RGB duplicate and no usage in indexed runs; different hashes are not a proof of statistical independence. Same six tasks/variation policy, not new-task, OOD or locked-test evidence.')
print(json.dumps(out,indent=2,allow_nan=False))
