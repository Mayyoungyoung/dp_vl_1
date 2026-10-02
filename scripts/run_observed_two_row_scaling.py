"""Separate frozen CPU preparation and GPU stages for fixed prefix44/prefix76.

The whole stage is fresh-only. Interrupted individual jobs retain record_job
commands and state; no stage automatically retries or chooses a new parent.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts import export_two_row_observations as exporter
from scripts.observation_cache_qwen import REVISION as QWEN_REVISION

PROTOCOL='observed_two_row_fixed_exposure_scaling_v1'
GPU_UUID='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab'
EXPECTED_INITIAL_MODEL_SHA256='7f81eba70dfcef2a3df19ebd5661ace3881abd552a1f2d4614cf490bf153bc13'


def stage_paths(project,source,train_parents):
    if type(train_parents) is not int or train_parents not in (32,64):
        raise ValueError('Only registered32/64 TRAIN scaling stages are permitted')
    project,source=Path(project),Path(source);prefix=train_parents+12
    return dict(data=project/f'data/observation_two_row_prefix{prefix}_v1',
        preparation=project/f'runs/observation_two_row_prefix{prefix}_preparation_v1',
        training=project/f'runs/observed_two_row_prefix{prefix}_v1',
        selection=source/f'configs/observed_two_row_prefix{prefix}_selection_v1.json',
        corpus=project/'data/observed_two_row_formal116_v1')


def validate_quality(data,selection,quality_path):
    data,quality_path=Path(data),Path(quality_path)
    manifest,gate=exporter.verify_export(data)
    count=exporter.selection_guard(selection)
    quality=json.loads(quality_path.read_text())
    rows=[json.loads(line) for line in (data/'observations.jsonl').read_text().splitlines()]
    labels=[json.loads(line) for line in (data/'supervision.jsonl').read_text().splitlines()]
    train=[row for row in labels if row['split']=='TRAIN']
    if (manifest['selection']!=selection or quality.get('protocol')!='two_row_train_endpoint_capacity_and_reference_quality_v1'
            or not quality.get('capacity_gate_passed') or quality.get('dev_raw_arrays_opened') is not False
            or quality.get('locked_raw_opened') is not False
            or quality.get('registered_train_parents')!=count or quality.get('registered_train_indices')!=list(range(count))
            or quality.get('source_export_manifest_sha256')!=exporter.digest(data/'export_manifest.json')
            or quality.get('train_input_ids')!=sorted(row['id'] for row in train)
            or quality.get('train_reference_counts')!={row['id']:len(row['routes']) for row in train}):
        raise ValueError('Complete fixed TRAIN quality gate required before any GPU work')
    if (not 0<len(rows)<=selection['requested_inputs'] or len(rows)!=manifest['actual_inputs']
            or any(set(row)!=exporter.INPUT_KEYS or row['split'] not in ('TRAIN','DEV_MODEL') for row in rows)
            or not any(row['split']=='DEV_MODEL' for row in rows) or not any(row['routes'] for row in train)):
        raise ValueError('Actual observation-only TRAIN/DEV inputs required')
    for path,value in quality['source_files_sha256'].items():
        if manifest['source_files_sha256'].get(path)!=value or exporter.digest(path)!=value:
            raise ValueError('TRAIN quality source changed')
    return manifest,gate,rows


def verify_fixed_dev(data,old_data):
    """Bind the exact old36 DEV conditions, without reading any other raw role."""
    data,old_data=Path(data),Path(old_data)
    manifests=[json.loads((root/'export_manifest.json').read_text()) for root in (old_data,data)]
    selected=[];source_hashes={}
    for root,manifest in zip((old_data,data),manifests):
        exporter.selection_guard(manifest['selection'])
        group={}
        for name in ('observations.jsonl','supervision.jsonl'):
            if exporter.digest(root/name)!=manifest['output_files_sha256'][name]:raise ValueError('DEV metadata hash changed')
            rows=[json.loads(line) for line in (root/name).read_text().splitlines()]
            rows=[row for row in rows if row['split']=='DEV_MODEL']
            group[name]={row['id']:row for row in rows}
            if len(group[name])!=36 or len(rows)!=36:raise ValueError('Exactly the original36 DEV inputs are required')
        selected.append(group)
    if manifests[0]['selection']['selected_indices']!=exporter.SELECTED or selected[0]!=selected[1]:
        raise ValueError('Old DEV observation/language/label identity differs')
    expected_ids={f'two_row_reach_{283200+i}_target{target}' for i in range(64,76) for target in range(3)}
    if any(set(group[name])!=expected_ids for group in selected for name in group):raise ValueError('DEV registration changed')
    for identifier in sorted(expected_ids):
        row=selected[0]['observations.jsonl'][identifier];label=selected[0]['supervision.jsonl'][identifier]
        paths=[row['image'],label['observation'],label['verification_only'],label['route_config']]+label['routes']
        for path in paths:
            value=manifests[0]['source_files_sha256'].get(path)
            if value is None or manifests[1]['source_files_sha256'].get(path)!=value or exporter.digest(path)!=value:
                raise ValueError('Fixed DEV raw input/label/reference SHA changed')
            source_hashes[path]=value
    # Also preserve every exported DEV receipt/artifact index source, not only arrays.
    for path,value in manifests[0]['source_files_sha256'].items():
        if '/parents/DEV_MODEL/' in path.replace('\\','/'):
            if manifests[1]['source_files_sha256'].get(path)!=value or exporter.digest(path)!=value:
                raise ValueError('Fixed DEV parent source SHA changed')
            source_hashes[path]=value
    return dict(protocol='fixed_original36_dev_identity_v1',conditions=36,input_ids=sorted(expected_ids),
        old_export_manifest_sha256=exporter.digest(old_data/'export_manifest.json'),
        new_export_manifest_sha256=exporter.digest(data/'export_manifest.json'),source_files_sha256=source_hashes,
        observation_and_supervision_rows_equal=True,raw_source_hashes_equal=True,locked_raw_opened=False)


def compare_dev_features(cache,old_cache,ids):
    cache,old_cache=Path(cache),Path(old_cache)
    records=[]
    for root in (old_cache,cache):
        rows=[json.loads(line) for line in (root/'samples.jsonl').read_text().splitlines()]
        chosen={r['id']:r for r in rows if r['id'] in ids}
        if len(chosen)!=len(ids) or sum(r['id'] in ids for r in rows)!=len(ids):raise ValueError('Fixed DEV cache identities missing or duplicated')
        records.append(chosen)
    results=[]
    for identifier in sorted(ids):
        arrays=[]
        for root,index in zip((old_cache,cache),records):
            row=index[identifier]
            if Path(row['file']).name!=row['file'] or exporter.digest(root/row['file'])!=row['sha256']:
                raise ValueError('Fixed DEV cache source hash changed')
            with np.load(root/row['file'],allow_pickle=False) as a:
                arrays.append({key:a[key].copy() for key in ('mean_hidden','last_hidden','id','parent_id','split','image_sha256','input_tokens')})
        if any(not np.array_equal(arrays[0][key],arrays[1][key]) for key in ('id','parent_id','split','image_sha256','input_tokens')):
            raise ValueError('Fixed DEV cache observation/token identity changed')
        delta={key:float(np.max(np.abs(arrays[0][key].astype(float)-arrays[1][key].astype(float)))) for key in ('mean_hidden','last_hidden')}
        results.append(dict(id=identifier,old_sha256=records[0][identifier]['sha256'],new_sha256=records[1][identifier]['sha256'],
            feature_max_abs_difference=delta,feature_arrays_equal=all(np.array_equal(arrays[0][key],arrays[1][key]) for key in delta)))
    return dict(protocol='fixed_original36_dev_qwen_identity_v1',conditions=len(ids),per_condition=results,
        passed=all(row['feature_arrays_equal'] for row in results),
        comparison='Feature arrays and input metadata by ID; distinct manifest/NPZ container hashes need not match.')


def training_audit(summary):
    stream=summary['sample_stream_audit']
    result=dict(initial_model_sha256=stream['initial_model_sha256'],expected_initial_model_sha256=EXPECTED_INITIAL_MODEL_SHA256,
        actual_last_step=summary['last_step'],actual_batches=stream['batches'],actual_observation_draws=stream['observation_draws'],
        actual_training_candidate_path_states=summary['trajectory_exposures'],batch_draw_stream_comparison_required=False)
    result['passed']=(result['initial_model_sha256']==EXPECTED_INITIAL_MODEL_SHA256 and result['actual_last_step']==1500
        and result['actual_batches']==1500 and result['actual_observation_draws']==48000 and result['actual_training_candidate_path_states']==192000)
    return result


def verify_cache(cache,data,rows):
    cache,data=Path(cache),Path(data)
    config=json.loads((cache/'cache_config.json').read_text());status=json.loads((cache/'status.json').read_text())
    if (config['model']!='Qwen/Qwen3-VL-2B-Instruct' or config['revision']!=QWEN_REVISION
            or config['processor']!=QWEN_REVISION or config['max_pixels']!=262144
            or config['model_trainable_parameter_count']!=0 or set(config['input_contract'])!=exporter.INPUT_KEYS
            or config['manifest_sha256']!=exporter.digest(data/'observations.jsonl') or config['transformers']!='4.57.1'
            or status['samples']!=len(rows) or status['status']!='frozen_rgb_language_feature_extraction_complete'):
        raise ValueError('Pinned actual Qwen cache contract changed')
    records=[json.loads(line) for line in (cache/'samples.jsonl').read_text().splitlines()]
    expected={row['id']:row for row in rows}
    if len(expected)!=len(rows) or len(records)!=len(rows) or {r['id'] for r in records}!=set(expected):
        raise ValueError('Cache does not contain exactly every actual input')
    for record in records:
        row=expected[record['id']];path=cache/record['file']
        if Path(record['file']).name!=record['file'] or exporter.digest(path)!=record['sha256']:
            raise ValueError('Cache file or hash changed')
        with np.load(path,allow_pickle=False) as a:
            if (a['mean_hidden'].shape!=(2048,) or a['last_hidden'].shape!=(2048,)
                    or not np.isfinite(a['mean_hidden']).all() or not np.isfinite(a['last_hidden']).all()
                    or str(a['id'])!=row['id'] or str(a['parent_id'])!=row['parent_id'] or str(a['split'])!=row['split']
                    or str(a['image_sha256'])!=exporter.digest(row['image'])):
                raise ValueError('Cached feature identity, current image or shape changed')
    return dict(samples=len(rows),cache_config_sha256=exporter.digest(cache/'cache_config.json'),
        samples_sha256=exporter.digest(cache/'samples.jsonl'),pooling='both',pooled_feature_dimension=4096,
        cache_status=status,cache_reused=False,
        scope='Every actual TRAIN and fixed DEV input re-encoded once; model loading and repeated DEV encoding cost retained.')


def source_record(source,launcher,selection):
    files=[Path(launcher),Path(selection),Path(__file__)]+[source/name for name in (
        'scripts/export_two_row_observations.py','scripts/audit_two_row_train_reference_quality.py',
        'scripts/train_observed_two_row.py','scripts/train_observed_geometry.py','scripts/train_observed_routes.py',
        'scripts/evaluate_observed_two_row.py','scripts/evaluate_observed_obstacles.py',
        'scripts/observation_cache_qwen.py','scripts/record_job.py','routeset/observed_geometry.py',
        'routeset/observed_route_head.py','routeset/observed_multitask.py','routeset/observed_training_audit.py')]
    return {str(path.resolve()):exporter.digest(path) for path in files}


def run_record(project,output,run_id,environment,module,arguments,resume):
    command=[str(project/environment),'-m',module]+list(map(str,arguments))
    subprocess.run([sys.executable,'-m','scripts.record_job','--output',str(output),'--run-id',run_id,
        '--resume-strategy',resume,'--']+command,check=True)


def run(project,source,train_parents,stage,launcher):
    project,source=Path(project).resolve(),Path(source).resolve()
    paths=stage_paths(project,source,train_parents)
    selection=json.loads(paths['selection'].read_text());exporter.selection_guard(selection)
    if (source.name!=os.environ.get('CODE_COMMIT') or len(source.name)!=40
            or any(c not in '0123456789abcdef' for c in source.name) or Path.cwd().resolve()!=source):
        raise ValueError('Run from the declared immutable release directory')
    if len(os.sched_getaffinity(0))!=1:raise ValueError('Exactly one authorized CPU is required')
    quality=paths['preparation']/'train_quality.json'
    if stage=='prepare':
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('Preparation must hide all GPUs')
        if paths['preparation'].exists() or paths['data'].exists() or paths['data'].with_name(paths['data'].name+'.staging').exists():
            raise FileExistsError('Fresh preparation only; inspect and recover individual recorded stages')
        # Mechanical readiness precedes any selected raw read or export creation.
        *_,missing=exporter.closed_prefix(paths['corpus'],selection)
        if missing:raise ValueError('Fixed requested prefix remains open: '+','.join(missing))
        paths['preparation'].mkdir(parents=True)
        exporter.write_json(paths['preparation']/'source_hashes.json',source_record(source,launcher,paths['selection']))
        exporter.write_json(paths['preparation']/'runtime.json',dict(protocol=PROTOCOL,stage=stage,code_commit=source.name,
            cpu_affinity=sorted(os.sched_getaffinity(0)),gpu_visible=False,threads=1,selection=selection,
            recovery='Never replay the whole stage; export/audit have fresh outputs and preserve failure records.'))
        run_record(project,paths['preparation'],'export','.venv/bin/python','scripts.export_two_row_observations',[
            '--source',paths['corpus'],'--selection',paths['selection'],'--output',paths['data']],'fresh-output')
        dev=verify_fixed_dev(paths['data'],project/'data/observation_two_row_prefix28_v1')
        exporter.write_json(paths['preparation']/'fixed_dev_identity.json',dev)
        run_record(project,paths['preparation'],'train_quality','.venv/bin/python','scripts.audit_two_row_train_reference_quality',[
            '--data-root',paths['data'],'--output',quality],'fresh-output')
        result=json.loads(quality.read_text())
        exporter.write_json(paths['preparation']/'preparation_receipt.json',dict(protocol=PROTOCOL,
            source_export_manifest_sha256=exporter.digest(paths['data']/'export_manifest.json'),
            quality_audit_sha256=exporter.digest(quality),selection_sha256=exporter.digest(paths['selection']),
            fixed_dev_identity_sha256=exporter.digest(paths['preparation']/'fixed_dev_identity.json'),
            capacity_gate_passed=result['capacity_gate_passed'],gpu_stage_started=False,
            next_action='Root reviews the completed TRAIN-only audit before independently launching the GPU stage.'))
        print(json.dumps(dict(stage=stage,train_parents=train_parents,capacity_gate_passed=result['capacity_gate_passed'],gpu_started=False)))
        return
    if stage!='gpu':raise ValueError('Separate prepare or gpu stage required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Only the authorized GPU1 may be visible')
    gpu=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
    if gpu!=GPU_UUID:raise ValueError('Authorized GPU identity changed')
    manifest,gate,rows=validate_quality(paths['data'],selection,quality)
    receipt=json.loads((paths['preparation']/'preparation_receipt.json').read_text())
    if (receipt['source_export_manifest_sha256']!=exporter.digest(paths['data']/'export_manifest.json')
            or receipt['quality_audit_sha256']!=exporter.digest(quality)
            or receipt['fixed_dev_identity_sha256']!=exporter.digest(paths['preparation']/'fixed_dev_identity.json')
            or receipt['selection_sha256']!=exporter.digest(paths['selection']) or not receipt['capacity_gate_passed']):
        raise ValueError('Preparation receipt differs from reviewed export/quality/selection')
    dev=verify_fixed_dev(paths['data'],project/'data/observation_two_row_prefix28_v1')
    if dev!=json.loads((paths['preparation']/'fixed_dev_identity.json').read_text()):raise ValueError('Fixed DEV identity receipt changed')
    cache=paths['data']/'qwen_cache';output=paths['training']
    if output.exists() or cache.exists():raise FileExistsError('Fresh GPU stage only; recover recorded cache or training job individually')
    output.mkdir(parents=True)
    exporter.write_json(output/'pre_cache_gate.json',gate)
    exporter.write_json(output/'preregistered_selection.json',selection)
    hashes=source_record(source,launcher,paths['selection'])
    hashes.update({str(path):exporter.digest(path) for path in (quality,paths['data']/'export_manifest.json',paths['preparation']/'preparation_receipt.json')})
    exporter.write_json(output/'source_hashes.json',hashes)
    exporter.write_json(output/'runtime.json',dict(protocol=PROTOCOL,stage=stage,code_commit=source.name,
        cpu_affinity=sorted(os.sched_getaffinity(0)),threads=1,gpu_uuid=gpu,gpu_memory_fraction=.35,
        requested_parents=len(selection['selected_indices']),requested_inputs=selection['requested_inputs'],actual_inputs=len(rows),
        training_steps=1500,batch_size=32,training_observation_draws=48000,training_candidate_path_states=192000,
        expected_initial_model_sha256=EXPECTED_INITIAL_MODEL_SHA256,batch_draw_stream_comparison_required=False,
        cache_reused=False,quality_audit_sha256=exporter.digest(quality),
        recovery='Never replay the whole stage. Cache may restart with identical config; training resumes the recorded command with --resume.',
        latency_scope='Actual cache/load and training costs separate; cached head latency is not an end-to-end Qwen request.'))
    run_record(project,output,'qwen_cache','.venv-qwen/bin/python','scripts.observation_cache_qwen',[
        '--model',project/'data/qwen3-vl-2b-instruct-89644892','--manifest',paths['data']/'observations.jsonl',
        '--output',cache,'--device','cuda','--threads','1','--gpu-memory-fraction','0.35','--max-pixels','262144'],'none')
    exporter.write_json(output/'actual_cache_receipt.json',verify_cache(cache,paths['data'],rows))
    features=compare_dev_features(cache,project/'data/observation_two_row_prefix28_v1/qwen_cache',dev['input_ids'])
    exporter.write_json(output/'fixed_dev_feature_identity.json',features)
    if not features['passed']:raise ValueError('Re-encoded fixed DEV feature arrays differ; preserve evidence and stop before training')
    exporter.verify_export(paths['data'])
    run_record(project,output,'peak_seed0','.venv/bin/python','scripts.train_observed_two_row',[
        '--data',paths['data'],'--selection',paths['selection'],'--quality-audit',quality,
        '--output',output/'peak_seed0','--device','cuda'],'flag')
    exporter.verify_export(paths['data'])
    audit=training_audit(json.loads((output/'peak_seed0/summary.json').read_text()))
    exporter.write_json(output/'initialization_and_exposure_audit.json',audit)
    if not audit['passed']:raise ValueError('Initial model or fixed training exposure differs from preregistration')
    print(json.dumps(dict(stage=stage,train_parents=train_parents,status='completed',ordinary_baseline=True)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',default='/home/wzy/dpvlm/route_set_v1')
    parser.add_argument('--train-parents',type=int,choices=(32,64),required=True)
    parser.add_argument('--stage',choices=('prepare','gpu'),required=True)
    parser.add_argument('--launcher',type=Path,required=True)
    args=parser.parse_args();run(args.project,Path.cwd(),args.train_parents,args.stage,args.launcher)


if __name__=='__main__':main()
