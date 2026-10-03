"""Two bounded eight-token technical partials, never a route-quality experiment."""
import argparse
import datetime
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from scripts import probe_hamster3d_train6 as original
from routeset import hamster_layer_transport as transport

ROOT=original.ROOT;REPO=Path(__file__).resolve().parents[1]
PROTOCOL='hamster3d_transport_exact8_v1'
FIRST='two_row_reach_283200_target0'
OLD_ROOT=ROOT/'runs/hamster3d_train6_probe_v1'
OLD_SOURCE_SHA='63a561d8c12fafe2fbe07eafbf3d72bf69db6d507203f14fa316ae10f60fc76d'
EXPECTED=dict(protocol=PROTOCOL,id=FIRST,split='TRAIN',arm_order=['original','pinned_layer'],
    maximum_generate_calls=2,candidates_per_call=1,max_new_tokens=8,maximum_model_forwards=16,
    maximum_generated_tokens=16,official_dtype='bfloat16',resident_layers=[0,1,2,3],
    offloaded_layers=list(range(4,36)),gpu_cap_bytes=8864694272,gpu_fraction=.35,
    rss_cap_bytes=32*2**30,workspace_bytes=2*2**30,cpu_affinity=[0],cpu_threads=1,
    asynchronous_streams=False,non_blocking=False,clear_cache_per_tensor=False,
    equality='all_actual_output_logits_shapes_dtypes_values_and_tokens_exact_finite',
    maximum_decode_median_ratio=.75,load_timeout_seconds=600,arm_timeout_seconds=180,
    conversion_timeout_seconds=180,total_timeout_seconds=1200,retry=False,resume=False,
    automatic_full_request=False,quality_metrics=None,full_request_authorized=False)


def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def sha(path):return original.sha(path)
def write(path,value):return original.write(path,value)


def remaining_seconds(started,phase_limit):
    remaining=EXPECTED['total_timeout_seconds']-(time.perf_counter()-started)
    if remaining<=0:raise TimeoutError('Whole technical body budget exhausted')
    return min(float(phase_limit),remaining)


def input_fingerprint(value):
    import torch
    if torch.is_tensor(value):
        return dict(shape=list(value.shape),dtype=str(value.dtype),sha256=transport.tensor_hash(value))
    if isinstance(value,dict):return {k:input_fingerprint(v) for k,v in sorted(value.items())}
    if isinstance(value,(list,tuple)):return [input_fingerprint(v) for v in value]
    raise ValueError('Unexpected non-observation input payload')


def validate_policy(value):
    if set(value)!={'policy','old_failure_files','installed_source_sha256'} or value['policy']!=EXPECTED:
        raise ValueError('Fixed two-partial scientific and runtime policy changed')
    files=value['old_failure_files']
    required={'probe/call_ledger.json','probe/model_loaded.json','probe/preflight.json','probe/status.json',
        'probe/'+FIRST+'/frame_0_640.npz','probe/'+FIRST+'/frame_0_640.png',
        'probe/'+FIRST+'/input.json','probe/'+FIRST+'/prompt_token_ids.npy',
        'probe/'+FIRST+'/streamed_generated_token_ids.npy','probe/'+FIRST+'/streamed_raw_output.txt',
        'probe.log','probe.status.json','probe_recipe_identity.json','probe_recipe_receipt.json','registry.jsonl'}
    if set(files)!=required:raise ValueError('Complete original failed request evidence required')
    if set(value['installed_source_sha256'])!={'big_modeling.py','hooks.py','utils/modeling.py','utils/memory.py'}:
        raise ValueError('Actual Accelerate implementation dependencies required')
    for name,item in files.items():
        if Path(name).is_absolute() or '..' in Path(name).parts or item['bytes']<0 or len(item['sha256'])!=64:
            raise ValueError('Invalid old evidence identity')
    return value


def preserve_failure(config,output):
    folder=output/'original_failure';folder.mkdir()
    for name,expected in config['old_failure_files'].items():
        src=original.confined(OLD_ROOT/name,OLD_ROOT)
        if src.stat().st_size!=expected['bytes'] or sha(src)!=expected['sha256']:
            raise ValueError('Sealed original failure changed: '+name)
        dest=folder/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest)
        if sha(dest)!=expected['sha256']:raise ValueError('Original failure copy changed bytes')
    old=read(folder/'probe/status.json');ledger=read(folder/'probe/call_ledger.json')
    if (old['status']!='failed' or old['exception_type']!='TimeoutError' or old['forward_calls']!=88
            or old['generated_tokens']!=87 or old['generate_calls']!=1 or old['candidates_unattempted']!=5
            or len(ledger)!=1 or ledger[0]['id']!=FIRST):
        raise ValueError('Historical failure identity/budget changed')
    tokens=np.load(folder/'probe'/FIRST/'streamed_generated_token_ids.npy',allow_pickle=False)
    if tokens.shape!=(87,) or not np.issubdtype(tokens.dtype,np.integer):raise ValueError('Historical87 prefix changed')
    write(output/'original_failure_index.json',dict(root=str(OLD_ROOT),files=config['old_failure_files'],
        copied_original_bytes=True,original_cost_not_recharged=True,historical_prefix_tokens=87))
    return tokens


def load_first():
    from PIL import Image
    data=ROOT/'data/observation_two_row_composite108_v1'
    if sha(data/'export_manifest.json')!=original.EXPORT_SHA:raise ValueError('Fixed composite export changed')
    manifest=read(data/'export_manifest.json')
    if sha(data/'observations.jsonl')!=manifest['output_files_sha256']['observations.jsonl']:
        raise ValueError('Fixed observed-input file changed')
    with (data/'observations.jsonl').open(encoding='utf8') as stream:row=json.loads(next(stream))
    if (set(row)!={'id','parent_id','split','image','instruction'} or row['id']!=FIRST
            or row['parent_id']!='two_row_reach_283200' or row['split']!='TRAIN'):
        raise ValueError('Only the exact first TRAIN observation is allowed')
    parent=Path(manifest['sources']['old']['source_dataset'])/'parents/TRAIN'/row['parent_id']
    image=original.confined(row['image'],parent);observation=original.confined(image.parent/'observation.npz',parent)
    hashes={str(p):sha(p) for p in (image,observation)}
    if any(manifest['source_files_sha256'][name]!=value for name,value in hashes.items()):
        raise ValueError('Current observation bytes changed')
    with np.load(observation,allow_pickle=False) as z:
        if set(z.files)!={'depth','gripper_pose','gripper_open','camera_intrinsics','camera_extrinsics'}:
            raise ValueError('Observed NPZ answer whitelist violation')
        observed={k:z[k].copy() for k in z.files}
    rgb=Image.open(image).convert('RGB')
    if np.asarray(rgb).shape!=(224,224,3) or observed['depth'].shape!=(224,224) or any(not np.isfinite(v).all() for v in observed.values()):
        raise ValueError('Invalid frozen first RGB-D')
    return dict(row=row,rgb=rgb,observed=observed),hashes


def prepare_inputs(processor,sample,folder):
    import torch
    from PIL import Image
    from hamster3d.inference.preprocessing import prepare_inputs as official_prepare,build_geometry_inputs,build_v5_messages
    folder.mkdir()
    prepared=official_prepare(sample['rgb'],sample['observed']['depth'],tmp_dir=str(folder))
    geometry=build_geometry_inputs(prepared['rgb_resized'],prepared['depth_resized'],device='cuda:0')
    text=processor.apply_chat_template(build_v5_messages(sample['row']['instruction']),tokenize=False,add_generation_prompt=True)
    inputs=processor(text=[text],images=[Image.fromarray(prepared['rgb_resized'])],padding=True,return_tensors='pt').to('cuda:0')
    for key,value in list(inputs.items()):
        if torch.is_tensor(value) and torch.is_floating_point(value):inputs[key]=value.to(torch.bfloat16)
    inputs['geometry_encoder_inputs']=[v.to(torch.bfloat16) for v in geometry['geometry_encoder_inputs']]
    inputs['depth_maps']=[v.to(torch.bfloat16) for v in geometry['depth_maps']]
    expected=read(OLD_ROOT/'probe'/FIRST/'input.json')
    ids=inputs['input_ids'].detach().cpu().numpy()
    if (text!=expected['prompt'] or sorted(inputs)!=expected['model_keys']
            or not np.array_equal(ids,np.load(OLD_ROOT/'probe'/FIRST/'prompt_token_ids.npy',allow_pickle=False))):
        raise ValueError('Same official prompt/input tokens required')
    # Official intermediate RGB and float16 depth are directly byte-bound.
    for name in ('frame_0_640.png','frame_0_640.npz'):
        if sha(folder/name)!=sha(OLD_ROOT/'probe'/FIRST/name):raise ValueError('Official resized RGB/depth bytes differ')
    np.save(folder/'prompt_token_ids.npy',ids,allow_pickle=False)
    write(folder/'input.json',dict(id=FIRST,prompt=text,keys=sorted(inputs),prompt_tokens=int(ids.shape[1]),
        official_preprocessing_exact=True,read_supervision=False))
    return inputs


class ExactEightStream(original.TokenAudit):
    def put(self,value):
        super().put(value)
        if len(self.generated_ids)>8:raise RuntimeError('Eight-token technical limit exceeded')


def compare_arrays(left,right):
    if len(left)!=8 or len(right)!=8:raise ValueError('Eight full-vocabulary logit arrays per arm required')
    rows=[]
    for a,b in zip(left,right):
        finite=bool(np.isfinite(a).all() and np.isfinite(b).all())
        rows.append(dict(shape_original=list(a.shape),shape_new=list(b.shape),dtype_original=str(a.dtype),
            dtype_new=str(b.dtype),finite=finite,exact=bool(finite and a.shape==b.shape and a.dtype==b.dtype and np.array_equal(a,b)),
            maximum_abs_difference=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if finite and a.shape==b.shape else None))
    return rows


def technical_gate(tokens_original,tokens_new,historical,logits,old_times,new_times,memory_passed):
    if len(old_times)!=8 or len(new_times)!=8 or any(not np.isfinite(v) or v<=0 for v in old_times+new_times):
        raise ValueError('Eight finite positive observed forward timings required')
    old_median=float(np.median(old_times[1:]));new_median=float(np.median(new_times[1:]))
    exact=(len(tokens_original)==len(tokens_new)==8 and list(tokens_original)==list(tokens_new)==list(historical[:8])
           and len(logits)==8 and all(r['finite'] and r['exact'] for r in logits))
    ratio=new_median/old_median
    return dict(passed=bool(exact and memory_passed and ratio<=.75),tokens_and_logits_exact=bool(exact),
        memory_passed=bool(memory_passed),original_decode_median_seconds=old_median,
        pinned_decode_median_seconds=new_median,decode_median_ratio=ratio,
        threshold=.75,includes_matched_logit_CPU_capture=True,sequential_order_confounded=True,
        quality_result=False,automatic_full_request=False)


def run_arm(model,processor,inputs,arm,folder,ledger,persist,seconds):
    import torch
    folder.mkdir();row=dict(arm=arm,id=FIRST,state='issued',partial_candidates=1,generate_calls=1,
                           forwards=[],generated_tokens=0)
    ledger.append(row);persist()
    prompt=inputs['input_ids'][0].detach().cpu().tolist();stream=ExactEightStream(prompt)
    captured=[];pending=[];started=time.perf_counter()
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    # This model cache is normally refreshed at prefill. Explicitly reset it for
    # the same first-request state in both arms; no KV cache is passed to generate.
    model.model.rope_deltas=None
    def pre(module,args,kwargs):
        if len(row['forwards'])>=8:raise RuntimeError('Ninth model forward refused before execution')
        item=dict(index=len(row['forwards']),state='issued',input_tokens=int(kwargs['input_ids'].shape[1]))
        row['forwards'].append(item);persist();torch.cuda.synchronize()
        pending.append(time.perf_counter())
    def post(module,args,kwargs,result):
        torch.cuda.synchronize();copy_start=time.perf_counter()
        values=result.logits.detach().cpu().contiguous().clone()
        cpu_seconds=time.perf_counter()-copy_start
        finite=bool(torch.isfinite(values).all())
        captured.append(values)
        item=row['forwards'][-1]
        item.update(state='completed',seconds_with_CPU_capture=time.perf_counter()-pending.pop(),
                    logits_CPU_capture_seconds=cpu_seconds,logits_shape=list(values.shape),logits_dtype=str(values.dtype),
                    logits_finite=finite,rss_bytes=transport.check_rss())
        persist()
        if not finite:raise ValueError('Nonfinite actual output logits')
        if torch.cuda.max_memory_reserved()>transport.CAP_BYTES:raise RuntimeError('VRAM cap exceeded')
    before=model.register_forward_pre_hook(pre,with_kwargs=True)
    after=model.register_forward_hook(post,with_kwargs=True)
    try:
        with original.deadline(seconds),torch.inference_mode():
            result=model.generate(**inputs,max_new_tokens=8,do_sample=False,temperature=None,top_p=None,streamer=stream)
        torch.cuda.synchronize()
        generated=result[:,len(prompt):].detach().cpu().tolist()[0]
        if generated!=stream.generated_ids:raise ValueError('Streamer/result token mismatch')
        if len(generated)!=8 or len(captured)!=8:raise ValueError('Eight-token matched diagnostic did not complete')
        row['state']='completed'
    except BaseException as error:
        row.update(state='failed',error_type=type(error).__name__,error=str(error));raise
    finally:
        before.remove();after.remove()
        row.update(elapsed_seconds=time.perf_counter()-started,generated_tokens=len(stream.generated_ids),
            forward_calls=len(row['forwards']),completed_forwards=len(captured),
            max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        np.save(folder/'generated_token_ids.npy',np.asarray(stream.generated_ids,dtype=np.int64),allow_pickle=False)
        (folder/'partial_output.txt').write_text(processor.decode(stream.generated_ids,skip_special_tokens=True),encoding='utf8')
        # Preserve the actual dtype (including bf16) with CPU tensor serialization.
        torch.save(captured,folder/'full_output_logits.pt');write(folder/'result.json',row);persist()
    return stream.generated_ids,captured,row


def run(config_path,output):
    config=validate_policy(read(config_path));output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter();ledger=[];torch=None;status=dict(protocol=PROTOCOL,status='running',pid=os.getpid(),
        code_commit=os.environ.get('CODE_COMMIT'),source_sha256=sha(__file__),config_sha256=sha(config_path),
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),quality_metrics=None,optimizer_steps=0)
    def persist():write(output/'call_ledger.json',ledger)
    write(output/'status.json',status);persist()
    try:
        if (sys.platform!='linux' or os.sched_getaffinity(0)!={0} or os.environ.get('CUDA_VISIBLE_DEVICES')!=original.GPU_UUID
                or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))):
            raise ValueError('Private Linux CPU0/GPU1 and single-thread environment required')
        if sha(original.__file__)!=OLD_SOURCE_SHA:raise ValueError('Original b127 baseline source bytes changed')
        historical=preserve_failure(config,output)
        import accelerate
        directory=Path(accelerate.__file__).parent
        if importlib.metadata.version('accelerate')!='1.10.1':raise ValueError('Accelerate version changed')
        for name,expected in config['installed_source_sha256'].items():
            if sha(directory/name)!=expected:raise ValueError('Installed Accelerate source differs')
        for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_IMPLICIT_TOKEN','XFORMERS_DISABLED'):os.environ[key]='1'
        with original.offline_network():
            assets,receipts=original.verify_prepared_assets();sample,input_hashes=load_first()
            write(output/'preflight.json',dict(policy=EXPECTED,input_sha256=input_hashes,preparation_receipts=receipts,
                old_source_sha256=OLD_SOURCE_SHA,hook_source_sha256=sha(transport.__file__),
                installed_source_sha256=config['installed_source_sha256'],historical_failure_files=config['old_failure_files'],
                no_supervision_or_geometry_labels=True))
            import torch as torch_module
            torch=torch_module;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0)
            torch.cuda.set_device(0)
            cap=min(int(torch.cuda.get_device_properties(0).total_memory*.35),int(8.4*2**30))
            if cap!=transport.CAP_BYTES:raise ValueError('Exact original35% VRAM cap differs')
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
            if available<transport.RSS_CAP_BYTES:raise RuntimeError('Less than32GiB currently available RAM')
            torch.cuda.set_per_process_memory_fraction(cap/torch.cuda.get_device_properties(0).total_memory,0)
            status.update(cuda_cap_bytes=cap,available_CPU_before_load_bytes=available)
            with original.deadline(remaining_seconds(started,600)):
                model,processor=original.load_offloaded(ROOT/assets['model_relative'],ROOT/assets['code_relative'],output,cap)
            transport.check_rss();status['model_load_seconds_from_start']=time.perf_counter()-started
            with original.deadline(remaining_seconds(started,180)):
                inputs=prepare_inputs(processor,sample,output/'prepared_input')
            input_identity=input_fingerprint(dict(inputs));write(output/'model_input_fingerprint.json',input_identity)
            status['load_preprocess_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            status['load_preprocess_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            old_tokens,old_logits,old_row=run_arm(model,processor,inputs,'original',output/'original',ledger,persist,remaining_seconds(started,180))
            if input_fingerprint(dict(inputs))!=input_identity:raise ValueError('Original call mutated common inputs')
            if old_tokens!=historical[:8].tolist():raise ValueError('Baseline differs from saved historical prefix')
            gc.collect();torch.cuda.synchronize();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
            conversion=time.perf_counter()
            with original.deadline(remaining_seconds(started,180)):
                hooks,audit=transport.install_transport(model,lambda a:write(output/'transport_audit.json',a))
            status['transport_conversion_seconds']=time.perf_counter()-conversion
            status['conversion_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            status['conversion_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            new_tokens,new_logits,new_row=run_arm(model,processor,inputs,'pinned_layer',output/'pinned_layer',ledger,persist,remaining_seconds(started,180))
            if input_fingerprint(dict(inputs))!=input_identity:raise ValueError('New call mutated common inputs')
            # NumPy cannot encode bf16, so numerical reporting uses lossless fp32
            # conversion; actual dtype and bitwise torch.equal remain authoritative.
            differences=compare_arrays([x.float().numpy() for x in old_logits],[x.float().numpy() for x in new_logits])
            for a,b,row in zip(old_logits,new_logits,differences):
                row.update(dtype_original=str(a.dtype),dtype_new=str(b.dtype),
                           exact=bool(a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)))
            audit=transport.verify_after(model,hooks,audit,8);write(output/'transport_audit.json',audit)
            remaining_seconds(started,1)
            result=technical_gate(old_tokens,new_tokens,historical,differences,
                [r['seconds_with_CPU_capture'] for r in old_row['forwards']],
                [r['seconds_with_CPU_capture'] for r in new_row['forwards']],True)
            result.update(logit_comparisons=differences,original_failure_prefix_sha256=config['old_failure_files']['probe/'+FIRST+'/streamed_generated_token_ids.npy']['sha256'],
                all_partial_candidates=2,actual_generate_calls=2,actual_forward_calls=16,
                elapsed_seconds=time.perf_counter()-started,cost_scope='Whole technical body; arm timings and conversion are nested',
                limits=['Fixed original then new order confounds allocator/OS warmth.','CPU copying all actual logits is matched instrumentation, not deployment throughput.',
                        'Only eight conditional output steps checked; no full-route success or unobserved-input equivalence claim.'])
            write(output/'report.json',result)
            if any(sha(Path(name))!=value for name,value in input_hashes.items()):raise ValueError('Observed input changed during run')
            for name,item in config['old_failure_files'].items():
                if sha(OLD_ROOT/name)!=item['sha256']:raise ValueError('Original failure evidence changed during run')
            status.update(status='completed',exit_code=0,gate_passed=result['passed'])
    except BaseException as error:
        import traceback
        status.update(status='failed',exit_code=1,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        status.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-started,
            generate_calls=len(ledger),partial_candidates_issued=len(ledger),partial_candidates_unattempted=2-len(ledger),
            forward_calls=sum(len(r['forwards']) for r in ledger),generated_tokens=sum(r.get('generated_tokens',0) for r in ledger),
            gpu_reservation_hours=(time.perf_counter()-started)/3600 if torch is not None else 0)
        if torch is not None and torch.cuda.is_initialized():
            import resource
            status.update(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                          final_cuda_allocated_bytes=torch.cuda.memory_allocated(),final_cuda_reserved_bytes=torch.cuda.memory_reserved())
            status['whole_max_memory_reserved_bytes']=max([torch.cuda.max_memory_reserved(),status.get('load_preprocess_peak_reserved_bytes',0),
                status.get('conversion_peak_reserved_bytes',0)]+[r.get('max_memory_reserved_bytes',0) for r in ledger])
            status['whole_max_memory_allocated_bytes']=max([torch.cuda.max_memory_allocated(),status.get('load_preprocess_peak_allocated_bytes',0),
                status.get('conversion_peak_allocated_bytes',0)]+[r.get('max_memory_allocated_bytes',0) for r in ledger])
        write(output/'status.json',status);persist()
        write(output/'artifact_index.json',{p.relative_to(output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size)
              for p in output.rglob('*') if p.is_file() and p.name!='artifact_index.json'})
    return status


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();raise SystemExit(run(args.config,args.output)['exit_code'])
