"""Separate quality/cache/train stages for the fixed composite108 ordinary control.

No stage launches the next. Historical sources and model/optimizer loops stay
unchanged. New DEV is never a selected input. This is data scaling, not a method.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

from scripts import export_two_row_composite_observations as exporter
from scripts import audit_two_row_train_reference_quality as quality_core
from scripts import observation_cache_qwen as qwen

PROTOCOL='ordinary_two_row_composite108_constant12000_v1'
QUALITY_PROTOCOL='two_row_composite108_train_capacity_v1'
CACHE_PROTOCOL='two_row_composite108_old225_cache_reuse_v1'
GPU_UUID='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab'
INITIAL=dict(initial_model_sha256='7f81eba70dfcef2a3df19ebd5661ace3881abd552a1f2d4614cf490bf153bc13',
    initial_torch_cpu_rng_sha256='c35b449a11db8b8af5620fb2377adf1709091f5d451783a0894523b68370a154',
    initial_sampler_state_sha256='48a64e771100689dadc6faec20d1a9d312e0e35cbd828ef0c10e20de352a56b6')
OLD_CACHE_HASHES=dict(cache_config_json='a70af2f4b16eb833f466cfc8cbc7ba34dbf965797244858f581d121d41656dbb',
    samples_jsonl='cbc17b382cddc076f40a62a3992ab097504b33521638984521738b66bbdf0989',
    status_json='abac30b3f6ef90e242219a800b3595f44038ff1eb66b9a72e1c6ebd953f111cc')
EXTRACTOR_SHA='9b8121ec8c95e4e95d89e191c0fae3b9251db120c7b621bed18dad7fc820d4bf'
TRAIN_PARENTS=[f'two_row_reach_{283200+i}' for i in range(64)]+[f'two_row_reach_{400000+i}' for i in range(32)]
MODEL_OPTIONS=dict(steps=12000,batch_size=32,candidates=4,horizon=24,width=128,depth=2,point_width=64,
    lr=.0003,seed=0,eval_every=250,pooling='both',pixel_stride=2,event_scale=.2,
    anchor_mode='straight_through_peak',endpoint_mode='surface_anchor',endpoint_residual_bound=.05,
    grounding_target='endpoint',grounding_weight=.02,grounding_sigma=.025,sampling_mode='uniform',
    sample_stream_audit=True,refinement_mode='none',objective='saturation')
digest=exporter.digest


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def lines(path):return [json.loads(r) for r in Path(path).read_text().splitlines() if r.strip()]


def validate_policy(p):
    fixed=dict(protocol=PROTOCOL,requested_train_parents=96,requested_reused_dev_parents=12,
        requested_train_inputs=288,fixed_dev_inputs=36,steps=12000,batch_size=32,candidates=4,
        observation_draws=384000,candidate_path_states=1536000,dev_selection_opportunities=48,
        seed=0,eval_every=250,lr=.0003,old_cache_inputs=225,new_qwen_inputs_max=96,
        new_dev_raw_allowed=False,automatic_stage_chaining=False)
    if any(p.get(k)!=v for k,v in fixed.items()) or p.get('model_options')!=MODEL_OPTIONS:
        raise ValueError('Fixed composite ordinary12000 policy changed')
    if p.get('initial_state')!=INITIAL or p.get('old_cache_metadata_sha256')!=OLD_CACHE_HASHES or p.get('extractor_sha256')!=EXTRACTOR_SHA:
        raise ValueError('Historical initialization/cache source identity changed')
    return p


def selected_train(data):
    """Metadata selects all recorded TRAIN before any TRAIN raw array opens."""
    observations=lines(Path(data)/'observations.jsonl');labels=lines(Path(data)/'supervision.jsonl')
    inputs={r['id']:r for r in observations if r['split']=='TRAIN'}
    selected=[r for r in labels if r['split']=='TRAIN']
    if (len(inputs)!=sum(r['split']=='TRAIN' for r in observations) or len(selected)!=len(inputs) or
            {r['id'] for r in selected}!=set(inputs) or any(r['parent_id'] not in TRAIN_PARENTS for r in selected)):
        raise ValueError('All actual composite TRAIN identities required')
    return inputs,selected


def quality(data,output):
    from PIL import Image
    from routeset.observed_geometry import backproject_rgbd
    from routeset.observed_route_head import resample_event_segments
    from scripts.evaluate_observed_two_row import scene_metrics
    data,output=Path(data),Path(output)
    if output.exists():raise FileExistsError('Fresh quality output required')
    started=time.perf_counter();manifest,gate=exporter.verify_export(data)
    inputs,labels=selected_train(data);hashes={};clouds={};refs=[];conditions=[]
    def checked(name):
        path=Path(name);value=digest(path)
        if manifest['source_files_sha256'].get(str(path))!=value:raise ValueError('TRAIN source changed')
        hashes[str(path)]=value;return path
    for label in labels:
        row=inputs[label['id']];parent=row['parent_id']
        if parent not in clouds:
            image=np.asarray(Image.open(checked(row['image'])).convert('RGB'))
            with np.load(checked(label['observation']),allow_pickle=False) as a:current={k:a[k].copy() for k in a.files}
            cloud=backproject_rgbd(image,current['depth'],current['camera_intrinsics'],current['camera_extrinsics'],pixel_stride=2)
            clouds[parent]=(cloud,current)
        cloud,current=clouds[parent]
        with np.load(checked(label['verification_only']),allow_pickle=False) as a:
            geometry={k:a[k] for k in ('obstacle_centers','obstacle_halfsizes')}
        cfg=json.loads(checked(label['route_config']).read_text())
        conditions.append(dict(id=label['id'],parent_id=parent,positive_references=len(label['routes']),
                               unknown_types=sum(t is None for t in label['route_types'])))
        for filename in label['routes']:
            with np.load(checked(filename),allow_pickle=False) as a:
                pose,events=a['gripper_pose'],a['gripper_open'];h24,h24_events=resample_event_segments(pose,events,24)
            metric,candidates=scene_metrics(h24[None],h24_events[None],current,geometry,label['semantic_targets'],label['route_types'],cfg)
            refs.append(dict(id=label['id'],parent_id=parent,route=filename,raw=quality_core.trajectory_statistics(pose[:,:3]),
                model_H24=quality_core.trajectory_statistics(h24),raw_event_transitions=int(np.count_nonzero(np.diff(events>.5))),
                h24_event_transitions=int(np.count_nonzero(np.diff(h24_events>.5))),
                endpoint_support=quality_core.endpoint_support(pose[-1,:3],cloud['world_xyz'],cloud['valid_mask']),
                model_H24_tip_valid=metric['TipValidAtK']==1,model_H24_type=candidates[0]['declared_passage_type']))
    exporter._sources_unchanged(hashes);errors=[r for r in refs if not r['endpoint_support']['axis_bound_representable']]
    result=dict(protocol=QUALITY_PROTOCOL,registered_train_parents=96,registered_train_parent_ids=TRAIN_PARENTS,
        requested_train_inputs=288,requested_train_route_proposals=2592,observed_train_parents=len(clouds),
        observed_train_inputs=len(labels),unobserved_requested_train_inputs=288-len(labels),
        source_export_manifest_sha256=digest(data/'export_manifest.json'),train_input_ids=sorted(inputs),
        train_reference_counts={r['id']:len(r['routes']) for r in labels},positive_references=len(refs),
        conditions=conditions,references=refs,axis_bound_m=.05,pixel_stride=2,horizon=24,
        capacity_gate_passed=bool(refs) and not errors,capacity_failures=errors,
        model_h24_tip_valid_references=sum(r['model_H24_tip_valid'] for r in refs),source_files_sha256=hashes,
        audit_source_sha256=digest(__file__),quality_core_sha256=digest(quality_core.__file__),
        elapsed_seconds=time.perf_counter()-started,data_filtering=False,dev_raw_arrays_opened=False,
        new_dev_raw_opened=False,locked_raw_opened=False,mechanical_gate=gate,
        scope='Endpoint representation capacity only, not learnability; H24 geometry diagnostics are not a filtering rule.')
    write(output,result);return result


def validate_quality(data,path):
    data,path=Path(data),Path(path);m,gate=exporter.verify_export(data);q=json.loads(path.read_text())
    inputs,labels=selected_train(data)
    required=dict(protocol=QUALITY_PROTOCOL,registered_train_parents=96,registered_train_parent_ids=TRAIN_PARENTS,
        requested_train_inputs=288,requested_train_route_proposals=2592,train_input_ids=sorted(inputs),
        train_reference_counts={r['id']:len(r['routes']) for r in labels},axis_bound_m=.05,pixel_stride=2,horizon=24,
        source_export_manifest_sha256=digest(data/'export_manifest.json'),capacity_gate_passed=True,
        data_filtering=False,dev_raw_arrays_opened=False,new_dev_raw_opened=False,locked_raw_opened=False,
        audit_source_sha256=digest(__file__),quality_core_sha256=digest(quality_core.__file__))
    if any(q.get(k)!=v for k,v in required.items()) or not q.get('references') or q.get('capacity_failures'):
        raise ValueError('Complete passing actual composite TRAIN capacity audit required')
    expected={(r['id'],p) for r in labels for p in r['routes']}
    actual=[(r['id'],r['route']) for r in q['references']]
    if len(actual)!=len(expected) or set(actual)!=expected or any(not r['endpoint_support']['axis_bound_representable'] for r in q['references']):
        raise ValueError('Capacity receipt omits or fails a TRAIN positive')
    for path,value in q['source_files_sha256'].items():
        if m['source_files_sha256'].get(path)!=value or digest(path)!=value:raise ValueError('TRAIN capacity source changed')
    return m,gate,q


def cache_key(row):return hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]+'.npz'


def inspect_cache(cache,manifest,rows,pin_old=False):
    cache,manifest=Path(cache),Path(manifest)
    if pin_old:
        for name in ('cache_config.json','samples.jsonl','status.json'):
            if digest(cache/name)!=OLD_CACHE_HASHES[name.replace('.','_')]:raise ValueError('Historical cache metadata changed')
        if digest(qwen.__file__)!=EXTRACTOR_SHA:raise ValueError('Historical cache extractor source changed')
    cfg=json.loads((cache/'cache_config.json').read_text());status=json.loads((cache/'status.json').read_text())
    required=dict(model='Qwen/Qwen3-VL-2B-Instruct',revision=qwen.REVISION,processor=qwen.REVISION,
        manifest_sha256=digest(manifest),max_pixels=262144,model_trainable_parameter_count=0,
        transformers='4.57.1',dtype='torch.bfloat16',model_parameter_count=2127532032)
    if any(cfg.get(k)!=v for k,v in required.items()) or set(cfg.get('input_contract',[]))!=qwen.ALLOWED:
        raise ValueError('Pinned Qwen cache configuration changed')
    records=lines(cache/'samples.jsonl');lookup={r['id']:r for r in records}
    if len(lookup)!=len(records) or set(lookup)!={r['id'] for r in rows} or status.get('samples')!=len(rows) or status.get('status')!='frozen_rgb_language_feature_extraction_complete':
        raise ValueError('Complete exact cache identities/status required')
    expected_fields={'mean_hidden','last_hidden','id','parent_id','split','image_sha256','input_tokens'}
    for row in rows:
        record=lookup[row['id']];path=cache/record['file']
        if record['file']!=cache_key(row) or digest(path)!=record['sha256']:raise ValueError('Cache source filename/hash changed')
        with np.load(path,allow_pickle=False) as a:
            if (set(a.files)!=expected_fields or any(str(a[k])!=row[k] for k in ('id','parent_id','split')) or
                    str(a['image_sha256'])!=digest(row['image']) or int(a['input_tokens'])!=record['input_tokens'] or
                    record['image_sha256']!=str(a['image_sha256']) or any(a[k].shape!=(2048,) or a[k].dtype!=np.float32 or
                    not np.isfinite(a[k]).all() for k in ('mean_hidden','last_hidden'))):
                raise ValueError('Cache feature arrays or observation identity changed')
    return cfg,status,lookup


def same_cache_contract(old,new):
    if {k:v for k,v in old.items() if k!='manifest_sha256'}!={k:v for k,v in new.items() if k!='manifest_sha256'}:
        raise ValueError('New cache differs from the actual historical encoder configuration')


def merge_cache(data,old_cache,new_cache,new_manifest,output):
    started=time.perf_counter();data,old_cache,new_cache,output=map(Path,(data,old_cache,new_cache,output))
    m,_=exporter.verify_export(data);rows=qwen.read_manifest(data/'observations.jsonl');new_rows=rows[225:]
    if qwen.read_manifest(Path(new_manifest))!=new_rows or any(r['split']!='TRAIN' for r in new_rows):
        raise ValueError('Only newly exported TRAIN inputs may be encoded')
    old_manifest=Path(m['historical_export'])/'observations.jsonl'
    if qwen.read_manifest(old_manifest)!=rows[:225]:raise ValueError('Historical225 observation identity changed')
    old_cfg,old_status,old_idx=inspect_cache(old_cache,old_manifest,rows[:225],pin_old=True)
    new_cfg,new_status,new_idx=inspect_cache(new_cache,new_manifest,new_rows)
    same_cache_contract(old_cfg,new_cfg)
    staging=output.with_name(output.name+'.staging')
    if output.exists() or staging.exists():raise FileExistsError('Fresh composite cache merge required')
    staging.mkdir(parents=True);index=[];reused=[]
    for row in rows:
        old=row['id'] in old_idx;record=(old_idx if old else new_idx)[row['id']]
        origin=(old_cache if old else new_cache)/record['file'];destination=staging/record['file']
        shutil.copyfile(origin,destination)
        if digest(destination)!=record['sha256']:raise ValueError('Copied NPZ bytes differ')
        index.append(record)
        if old:reused.append(dict(id=row['id'],split=row['split'],file=record['file'],sha256=record['sha256'],
                                npz_bytes_equal=True,feature_arrays_equal=True))
    (staging/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in index),encoding='utf-8')
    write(staging/'cache_config.json',dict(old_cfg,manifest_sha256=digest(data/'observations.jsonl')))
    write(staging/'status.json',dict(status='frozen_rgb_language_feature_extraction_complete',samples=len(rows),
        reused_samples=225,newly_encoded_samples=len(new_rows),new_encoding_status=new_status,
        historical_cost_not_added=old_status,merge_elapsed_seconds=time.perf_counter()-started))
    receipt=dict(protocol=CACHE_PROTOCOL,source_export_manifest_sha256=digest(data/'export_manifest.json'),
        old_cache=str(old_cache),new_cache=str(new_cache),new_manifest=str(new_manifest),
        new_manifest_sha256=digest(new_manifest),new_encoding_count=len(new_rows),old_reused_count=225,
        reused_dev_count=36,old225_by_id=reused,new_input_ids=[r['id'] for r in new_rows],
        extractor_sha256=digest(qwen.__file__),old_config_sha256=digest(old_cache/'cache_config.json'),
        old_index_sha256=digest(old_cache/'samples.jsonl'),new_config_sha256=digest(new_cache/'cache_config.json'),
        new_index_sha256=digest(new_cache/'samples.jsonl'),new_status_sha256=digest(new_cache/'status.json'),
        artifact_sha256={p.name:digest(p) for p in staging.iterdir() if p.is_file()},
        cost_scope='Historical encoding cost retained, not charged again. New Qwen load/encoding and copy/validation are separate; no E2E latency claim.')
    write(staging/'composite_cache_receipt.json',receipt);staging.rename(output);return receipt


def validate_cache(data):
    data=Path(data);m,_=exporter.verify_export(data);cache=data/'qwen_cache'
    r=json.loads((cache/'composite_cache_receipt.json').read_text());rows=qwen.read_manifest(data/'observations.jsonl')
    if r.get('protocol')!=CACHE_PROTOCOL or r['source_export_manifest_sha256']!=digest(data/'export_manifest.json'):
        raise ValueError('Composite cache lineage differs')
    merged_cfg,_,_=inspect_cache(cache,data/'observations.jsonl',rows)
    old=Path(r['old_cache']);old_rows=qwen.read_manifest(Path(m['historical_export'])/'observations.jsonl')
    old_cfg,_,index=inspect_cache(old,Path(m['historical_export'])/'observations.jsonl',old_rows,pin_old=True)
    new_cache=Path(r['new_cache']);new_manifest=Path(r['new_manifest'])
    if (r['extractor_sha256']!=digest(qwen.__file__) or r['new_manifest_sha256']!=digest(new_manifest) or
            qwen.read_manifest(new_manifest)!=rows[225:]):raise ValueError('New encoding source/manifest changed')
    for key,path in [('old_config_sha256',old/'cache_config.json'),('old_index_sha256',old/'samples.jsonl'),
        ('new_config_sha256',new_cache/'cache_config.json'),('new_index_sha256',new_cache/'samples.jsonl'),
        ('new_status_sha256',new_cache/'status.json')]:
        if r[key]!=digest(path):raise ValueError('Encoding provenance changed')
    new_cfg,_,new_index=inspect_cache(new_cache,new_manifest,rows[225:])
    same_cache_contract(old_cfg,new_cfg);same_cache_contract(old_cfg,merged_cfg)
    for record in new_index.values():
        if digest(cache/record['file'])!=record['sha256']:raise ValueError('New encoded cache bytes differ after merge')
    if (r['old_reused_count']!=225 or r['reused_dev_count']!=36 or r['new_encoding_count']!=len(rows)-225 or
            r['new_input_ids']!=[v['id'] for v in rows[225:]] or len(r['old225_by_id'])!=225):
        raise ValueError('Composite cache exposure/identity differs')
    for row,saved in zip(old_rows,r['old225_by_id']):
        record=index[row['id']]
        if saved!=dict(id=row['id'],split=row['split'],file=record['file'],sha256=record['sha256'],npz_bytes_equal=True,feature_arrays_equal=True) or digest(cache/record['file'])!=record['sha256']:
            raise ValueError('Historical feature bytes changed')
    for name,value in r['artifact_sha256'].items():
        if Path(name).name!=name or digest(cache/name)!=value:raise ValueError('Merged cache artifact changed')
    return r


def cache(data,quality_path,output,model):
    data,output=Path(data),Path(output);validate_quality(data,quality_path)
    if output.exists() or (data/'qwen_cache').exists() or (data/'qwen_cache.staging').exists():raise FileExistsError('Fresh cache stage only; no partial encoding replay')
    m,_=exporter.verify_export(data);rows=qwen.read_manifest(data/'observations.jsonl');new=rows[225:]
    if not new or len(new)>96 or any(r['split']!='TRAIN' for r in new):raise ValueError('New actual TRAIN cache scope is empty or invalid')
    old=Path(m['historical_export'])/'qwen_cache'
    inspect_cache(old,Path(m['historical_export'])/'observations.jsonl',rows[:225],pin_old=True)
    output.mkdir(parents=True);manifest=output/'new_train_observations.jsonl'
    manifest.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new),encoding='utf-8')
    command=[sys.executable,'-m','scripts.observation_cache_qwen','--model',str(model),'--manifest',str(manifest),
        '--output',str(output/'new_qwen_cache'),'--device','cuda','--threads','1','--gpu-memory-fraction','.35','--max-pixels','262144']
    write(output/'requested_encoding.json',dict(new_request_count=len(new),new_input_ids=[r['id'] for r in new],command=command,
        failure_policy='No automatic retry. All scheduled requests are retained as the interruption upper bound.'))
    started=time.perf_counter()
    with (output/'new_encoding.log').open('w',encoding='utf-8') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=False)
    write(output/'encoding_process.json',dict(command=command,exit_code=result.returncode,elapsed_seconds=time.perf_counter()-started))
    if result.returncode:raise RuntimeError('New Qwen extraction failed; preserve partial cache without replay')
    result=merge_cache(data,old,output/'new_qwen_cache',manifest,data/'qwen_cache')
    write(output/'cache_stage_receipt.json',result);return result


def source_hashes():
    root=Path(__file__).resolve().parents[1]
    names=['scripts/run_observed_two_row_composite.py','scripts/export_two_row_composite_observations.py',
        'scripts/audit_two_row_train_reference_quality.py','scripts/observation_cache_qwen.py',
        'scripts/train_observed_geometry.py','scripts/train_observed_two_row.py','scripts/train_observed_routes.py',
        'scripts/evaluate_observed_two_row.py','scripts/evaluate_observed_obstacles.py',
        'routeset/observed_geometry.py','routeset/observed_route_head.py','routeset/observed_multitask.py',
        'routeset/observed_path_refinement.py','routeset/observed_training_audit.py']
    return {name:digest(root/name) for name in names}


@contextmanager
def training_adapter(ordinary,sources,initial=INITIAL):
    old_verify=ordinary.verify_export;old_audit=ordinary.base.new_stream_audit
    def checked(*args,**kwargs):
        value=old_audit(*args,**kwargs)
        if any(value.get(k)!=v for k,v in initial.items()):raise ValueError('Initial model/RNG/sampler differs from ordinary control')
        return value
    ordinary.verify_export=exporter.verify_export;ordinary.base.new_stream_audit=checked
    try:
        with ordinary.evaluation_adapter(sources):yield
    finally:
        ordinary.verify_export=old_verify;ordinary.base.new_stream_audit=old_audit


def validate_resume(current,saved):
    ignored={'resume','stop_after','device','threads','code_commit'}
    for key,value in current.items():
        if key not in ignored and (key not in saved or saved[key]!=value):raise ValueError('Composite resume differs: '+key)


def validate_budget(summary,checkpoint):
    audit=checkpoint['sample_stream_audit']
    if (checkpoint['step']!=12000 or checkpoint['trajectory_exposures']!=1536000 or summary['trajectory_exposures']!=1536000 or
            summary['last_step']!=12000 or audit['batches']!=12000 or audit['observation_draws']!=384000 or
            [r['step'] for r in checkpoint['history']]!=list(range(250,12001,250)) or
            any(audit.get(k)!=v for k,v in INITIAL.items()) or summary['sample_stream_audit']!=audit):
        raise ValueError('Composite actual initialization/exposure/selection budget differs')
    return dict(observation_draws=384000,training_candidate_path_states=1536000,dev_selection_opportunities=48,
        initial_state=INITIAL,actual_index_chain_sha256=audit['index_chain_sha256'],same_sample_stream_as_old=False)


def train(data,quality_path,policy_path,output,resume=False,stop_after=None,device='cuda'):
    import torch
    from scripts import train_observed_two_row as ordinary
    data,output=Path(data).resolve(),Path(output).resolve();policy_path=Path(policy_path).resolve()
    p=validate_policy(json.loads(policy_path.read_text()));m,gate,q=validate_quality(data,quality_path);validate_cache(data)
    if stop_after is not None and (type(stop_after)is not int or stop_after<=0 or stop_after>=12000 or stop_after%250):
        raise ValueError('Resume checkpoints only at predeclared250-step selections')
    inputs,labels=selected_train(data)
    options=dict(MODEL_OPTIONS,observations=str(data/'observations.jsonl'),supervision=str(data/'supervision.jsonl'),
        cache_dir=str(data/'qwen_cache'),output=str(output),checkpoint_selection='tip_unique_valid',geometry_pooling='spatial',
        metric_aggregation='instruction',multitask_snapshot_manifest=None,refinement_sigma=None,refinement_prefix_fraction=None,
        refinement_bound=None,resume=resume,stop_after=stop_after,device=device,threads=1,
        composite_protocol=PROTOCOL,composite_policy_sha256=digest(policy_path),composite_source_sha256=source_hashes(),
        composite_export_sha256=digest(data/'export_manifest.json'),composite_quality_sha256=digest(quality_path),
        composite_cache_sha256=digest(data/'qwen_cache/composite_cache_receipt.json'),
        two_row_tip_protocol=ordinary.PROTOCOL,model_use_gate_source='scripts.export_two_row_composite_observations.verify_export')
    if not resume and output.exists():raise FileExistsError('Fresh training output required')
    if resume and not (output/'last.pt').exists():raise ValueError('Actual last checkpoint required for resume')
    current=torch.load(output/'last.pt',map_location='cpu',weights_only=False) if resume else None
    if current is not None:validate_resume(options,current['config'])
    output.mkdir(parents=True,exist_ok=True)
    lock=output/'composite_driver.lock';fd=os.open(str(lock),os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    started=time.perf_counter();event=dict(resume=resume,start_step=0 if current is None else current['step'],status='running')
    try:
        if current is not None and current['step']==12000:
            receipt=verify_completion(output,ordinary)
            event['optimization_and_inference_skipped']=True
            return receipt
        with training_adapter(ordinary,(options['observations'],options['supervision'])):
            ordinary.base.train(SimpleNamespace(**options))
        checkpoint=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
        if checkpoint['step']<12000:return dict(status='stopped_at_registered_checkpoint',step=checkpoint['step'])
        summary=json.loads((output/'summary.json').read_text());budget=validate_budget(summary,checkpoint)
        receipt=dict(protocol=PROTOCOL,export_sha256=options['composite_export_sha256'],
            quality_sha256=options['composite_quality_sha256'],cache_receipt_sha256=options['composite_cache_sha256'],
            source_sha256=options['composite_source_sha256'],policy_sha256=options['composite_policy_sha256'],
            summary_sha256=digest(output/'summary.json'),checkpoint_sha256={n:digest(output/n) for n in ('best.pt','last.pt')},
            prediction_artifacts=ordinary.prediction_artifact_index(output),budget=budget,
            final_mechanical_gate=exporter.verify_export(data)[1],actual_train_inputs=len(inputs),
            eligible_positive_train_inputs=sum(bool(r['routes']) for r in labels),requested_train_inputs=288,
            average_draws_per_eligible_input=384000/sum(bool(r['routes']) for r in labels),
            main_evaluation_requests=sum(r['dev_model']['examples'] for r in checkpoint['history'])+
                sum(summary[k]['examples'] for k in ('metrics','last_metrics','train_metrics')),
            cached_latency_extra_requests=24,cached_latency_extra_complete_path_states=96,
            base_elapsed_seconds=summary['elapsed_s'],base_gpu_hours_reserved=summary['gpu_hours_reserved'],
            scope='Same total training budget and ordinary model, larger data; not equal per-parent exposure or core-method evidence')
        write(output/'composite_training_receipt.json',receipt);return receipt
    finally:
        event['elapsed_seconds']=time.perf_counter()-started;event['gpu_hours_reserved']=event['elapsed_seconds']/3600 if device.startswith('cuda') else 0.
        event['status']='completed' if sys.exc_info()[0] is None else 'failed'
        if sys.exc_info()[0] is not None:event['exception']=repr(sys.exc_info()[1])
        with (output/'driver_events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(event)+'\n')
        lock.unlink()


def verify_completion(output,ordinary):
    import torch
    output=Path(output);path=output/'composite_training_receipt.json'
    if not path.exists():raise ValueError('Unsealed completed training: preserve all pools, no automatic inference replay')
    r=json.loads(path.read_text());checkpoint=torch.load(output/'last.pt',map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];summary=json.loads((output/'summary.json').read_text())
    if (r.get('protocol')!=PROTOCOL or r['summary_sha256']!=digest(output/'summary.json') or
            r['source_sha256']!=source_hashes() or r['export_sha256']!=cfg['composite_export_sha256'] or
            r['quality_sha256']!=cfg['composite_quality_sha256'] or r['policy_sha256']!=cfg['composite_policy_sha256'] or
            r['cache_receipt_sha256']!=cfg['composite_cache_sha256'] or r['budget']!=validate_budget(summary,checkpoint) or
            r['prediction_artifacts']!=ordinary.prediction_artifact_index(output) or
            any(digest(output/name)!=value for name,value in r['checkpoint_sha256'].items())):
        raise ValueError('Completed composite seal changed')
    return r


def fixed_last_train(data,quality_path,output,model_folder):
    import torch
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from routeset.observed_route_head import load_observed_dataset
    from scripts import train_observed_two_row as ordinary
    from scripts.evaluate_observed_two_row_online import head_options
    data,output,model_folder=map(Path,(data,output,model_folder));validate_quality(data,quality_path);validate_cache(data)
    verify_completion(model_folder,ordinary);checkpoint=torch.load(model_folder/'last.pt',map_location='cpu',weights_only=False)
    cfg=checkpoint['config'];ckptsha=digest(model_folder/'last.pt');exportsha=digest(data/'export_manifest.json')
    if cfg['composite_export_sha256']!=exportsha:raise ValueError('Fixed-last diagnostic source differs')
    recovered=recover_diagnostic(output,ckptsha,exportsha)
    if recovered is not None:return recovered
    started=time.perf_counter();torch.set_num_threads(1);inputs,_=selected_train(data)
    loaded=load_observed_dataset(cfg['observations'],cfg['supervision'],cfg['cache_dir'],24,'both')
    geometry=ordinary.base.load_geometry(loaded,cfg['observations'],cfg['supervision'],2)
    ids=np.flatnonzero(loaded['splits']=='TRAIN')
    if set(map(str,loaded['scene_ids'][ids]))!=set(inputs) or len(ids)!=len(inputs):raise ValueError('All actual TRAIN diagnostic inputs required')
    model=ObservedGeometryRouteHead(**head_options(cfg));model.load_state_dict(checkpoint['model']);model.eval()
    versions=tuple(p._version for p in model.parameters());staging=output.with_name(output.name+'.staging');staging.mkdir(parents=True)
    write(staging/'request_receipt.json',dict(checkpoint_sha256=ckptsha,source_export_manifest_sha256=exportsha,
        reserved_forward_requests=len(ids),reserved_complete_path_states=4*len(ids),
        interruption_attempted_lower=0,interruption_attempted_upper=len(ids),no_automatic_replay=True))
    with training_adapter(ordinary,(cfg['observations'],cfg['supervision'])):
        metrics=ordinary.evaluate(model,loaded,geometry,ids,'cpu',staging,selection_metric='tip_unique_valid',
                                  evaluation_sources=(cfg['observations'],cfg['supervision']))
    if versions!=tuple(p._version for p in model.parameters()) or digest(model_folder/'last.pt')!=ckptsha:raise ValueError('Diagnostic changed weights')
    exporter.verify_export(data)
    r=dict(checkpoint_sha256=ckptsha,source_export_manifest_sha256=exportsha,fixed_last_step=12000,
        requested_train_inputs=288,actual_train_inputs=len(ids),unavailable_train_inputs=288-len(ids),
        new_forward_requests=len(ids),new_complete_path_states=4*len(ids),new_qwen_encodings=0,new_dev_predictions=0,
        optimizer_updates=0,metrics=metrics,elapsed_seconds=time.perf_counter()-started,
        artifact_sha256={p.name:digest(p) for p in staging.iterdir() if p.is_file()})
    write(staging/'diagnostic_receipt.json',r);staging.rename(output);return r


def recover_diagnostic(output,checkpoint_sha,export_sha):
    output=Path(output);staging=output.with_name(output.name+'.staging')
    existing=output if output.exists() else staging if staging.exists() else None
    if existing is None:return None
    if not (existing/'diagnostic_receipt.json').exists():
        raise ValueError('Unsealed TRAIN diagnostic remains; preserve its requested upper bound, no automatic forward replay')
    r=json.loads((existing/'diagnostic_receipt.json').read_text())
    if r['checkpoint_sha256']!=checkpoint_sha or r['source_export_manifest_sha256']!=export_sha:
        raise ValueError('Sealed TRAIN diagnostic source changed')
    for name,value in r['artifact_sha256'].items():
        if Path(name).name!=name or digest(existing/name)!=value:raise ValueError('Sealed TRAIN diagnostic pool changed')
    if existing==staging:staging.rename(output)
    return r


def resource_guard(stage):
    if hasattr(os,'sched_getaffinity') and len(os.sched_getaffinity(0))!=1:raise ValueError('Exactly one CPU required')
    if stage in ('quality','fixed-last-train'):
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('CPU stage must hide GPUs')
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Only GPU1 authorized')
        uuid=subprocess.check_output(['nvidia-smi','--id=1','--query-gpu=uuid','--format=csv,noheader'],text=True).strip()
        if uuid!=GPU_UUID:raise ValueError('GPU UUID changed')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('quality','cache','train','fixed-last-train'),required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--policy',type=Path,required=True);p.add_argument('--quality-audit',type=Path)
    p.add_argument('--model',type=Path);p.add_argument('--model-folder',type=Path)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args(argv);validate_policy(json.loads(a.policy.read_text()));resource_guard(a.stage)
    if a.stage!='train' and (a.resume or a.stop_after is not None):p.error('Only training supports checkpoint resume')
    if a.stage!='quality' and a.quality_audit is None:p.error('--quality-audit required')
    if a.stage=='quality':r=quality(a.data,a.output)
    elif a.stage=='cache':
        if a.model is None:p.error('--model required for actual new Qwen encoding')
        r=cache(a.data,a.quality_audit,a.output,a.model)
    elif a.stage=='train':r=train(a.data,a.quality_audit,a.policy,a.output,a.resume,a.stop_after)
    else:
        if a.model_folder is None:p.error('--model-folder required')
        r=fixed_last_train(a.data,a.quality_audit,a.output,a.model_folder)
    print(json.dumps(dict(stage=a.stage,protocol=r.get('protocol'),output=str(a.output),capacity_gate_passed=r.get('capacity_gate_passed'))))


if __name__=='__main__':main()
