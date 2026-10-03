"""One separately authorized TRAIN0 request after sealed exact8 transport gates.

No warmup, alternate hook generation, quality labels, retries or automatic stages.
The old probes and installed libraries are never modified.
"""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np
from scripts import probe_hamster3d_transport as prior
from routeset import hamster_layer_transport as transport

original=prior.original;ROOT=original.ROOT;REPO=Path(__file__).resolve().parents[1]
PROTOCOL='hamster3d_pinned_full1_train0_v1'
PRIOR_COMMIT='cbcd8129c967db6f0995a824754752cc7a6f0f5b'
PRIOR_ROOT=ROOT/'runs/hamster3d_transport_exact8_v1'
OUTPUT_ROOT=ROOT/'runs/hamster3d_pinned_full1_v1/probe'
EXPECTED=dict(protocol=PROTOCOL,id=prior.FIRST,split='TRAIN',generate_calls=1,candidate_budget=1,
    maximum_model_forwards=1024,max_new_tokens=1024,request_timeout_seconds=900,
    model_load_timeout_seconds=600,conversion_timeout_seconds=180,total_timeout_seconds=1800,
    seed=0,do_sample=False,dtype='bfloat16',resident_layers=[0,1,2,3],offloaded_layers=list(range(4,36)),
    gpu_cap_bytes=8864694272,gpu_fraction=.35,rss_cap_bytes=32*2**30,workspace_bytes=2*2**30,
    cpu_affinity=[0],cpu_threads=1,prefix_logit_comparisons=8,historical_token_prefix=87,
    warmup_calls=0,original_hook_generate_calls=0,retry=False,resume=False,
    read_supervision=False,quality_metrics=None,task_success=None,automatic_next_stage=False)
# Filled from a read-only hash inventory of the completed actual exact8 run.
EVIDENCE_CANONICAL_SHA='a971ef49e87cd9dd9afb61ce49c74907dad5b082e0f2c6dd0da91747e6839f9f'
SOURCE_CANONICAL_SHA='959bd68858c7d9439a3aa8b3993141139daf82ff789c352715bfb1f59dc9999c'
read=prior.read;sha=prior.sha;write=prior.write


def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_policy(value):
    if set(value)!={'policy','exact8_files','dependency_sources'} or value['policy']!=EXPECTED:
        raise ValueError('Fixed one-request policy changed')
    if canonical(value['exact8_files'])!=EVIDENCE_CANONICAL_SHA or len(value['exact8_files'])!=41:
        raise ValueError('All actual sealed exact8 evidence required')
    if canonical(value['dependency_sources'])!=SOURCE_CANONICAL_SHA:
        raise ValueError('Frozen original and transport source identities changed')
    return value


def check_completed_gate(report,status,outer,receipt):
    if (status.get('status')!='completed' or status.get('exit_code')!=0 or not status.get('gate_passed')
            or status.get('code_commit')!=PRIOR_COMMIT or status.get('generate_calls')!=2
            or status.get('forward_calls')!=16 or status.get('generated_tokens')!=16
            or status.get('partial_candidates_unattempted')!=0):
        raise ValueError('Actual completed exact8 budget gate failed')
    if (outer.get('status')!='completed' or outer.get('exit_code')!=0 or not outer.get('end_utc')
            or outer.get('code_commit')!=PRIOR_COMMIT or receipt.get('status')!='completed'
            or receipt.get('child_exit_code')!=0 or not receipt.get('source_unchanged')
            or not receipt.get('gate_passed') or receipt.get('code_commit')!=PRIOR_COMMIT):
        raise ValueError('Exact8 outer job and source receipt must have completed')
    rows=report.get('logit_comparisons',[])
    if (not report.get('passed') or not report.get('tokens_and_logits_exact') or not report.get('memory_passed')
            or report.get('actual_forward_calls')!=16 or len(rows)!=8
            or not 0<report.get('decode_median_ratio',float('inf'))<=.75):
        raise ValueError('Exact8 numerical, resource or speed gate failed')
    for row in rows:
        if (not row.get('exact') or not row.get('finite') or row.get('maximum_abs_difference')!=0
                or row.get('shape_original')!=[1,1,151936] or row.get('shape_new')!=[1,1,151936]
                or row.get('dtype_original')!='torch.bfloat16' or row.get('dtype_new')!='torch.bfloat16'):
            raise ValueError('Every full-vocabulary actual logit must match exactly')
    if status.get('whole_max_memory_reserved_bytes',float('inf'))>transport.CAP_BYTES or status.get('process_peak_rss_bytes',float('inf'))>transport.RSS_CAP_BYTES:
        raise ValueError('Exact8 resource limits failed')


def verify_exact8(config,root=PRIOR_ROOT):
    root=Path(root)
    for name,item in config['exact8_files'].items():
        path=original.confined(root/name,root)
        if path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            raise ValueError('Actual exact8 artifact changed: '+name)
    index=read(root/'probe/artifact_index.json')
    expected={name[len('probe/'):]:item for name,item in config['exact8_files'].items()
              if name.startswith('probe/') and name!='probe/artifact_index.json'}
    if index!=expected:raise ValueError('Sealed artifact index does not bind the complete prior pool')
    report,status,outer,receipt=[read(root/name) for name in ('probe/report.json','probe/status.json','probe.status.json','probe_receipt.json')]
    check_completed_gate(report,status,outer,receipt)
    for name,key in [('probe/status.json','inner_status_sha256'),('probe/report.json','report_sha256'),('probe/artifact_index.json','artifact_index_sha256')]:
        if receipt[key]!=config['exact8_files'][name]['sha256']:raise ValueError('Completed receipt binding changed')
    return dict(prior_commit=PRIOR_COMMIT,verified_files=len(expected)+6,files=config['exact8_files'],
                original_cost_not_recharged=True,actual_gate=report,outer=outer)


def remaining(started,limit):
    # Shared elapsed-time allowance for the guarded major phases, not a global
    # hard alarm. Initial evidence checks and final parsing/hash/index writes
    # may run outside these phase guards; record_job captures their actual wall.
    value=EXPECTED['total_timeout_seconds']-(time.perf_counter()-started)
    if value<=0:raise TimeoutError('Whole full1 body budget exhausted')
    return min(float(limit),value)


def prefix_check(tokens,historical):
    n=min(len(tokens),len(historical))
    equal=list(tokens[:n])==list(historical[:n])
    return dict(compared_tokens=n,all_compared_equal=equal,complete_87_token_prefix=len(tokens)>=87,
                first_mismatch=next((i for i in range(n) if tokens[i]!=historical[i]),None))


def verify_transport_after(model,hooks,audit,count):
    """Same tensor/placement invariants, with this separate full1 call budget."""
    if len(hooks)!=32 or not 1<=count<=1024:raise ValueError('Invalid completed full1 forward count')
    layers=model.model.language_model.layers
    if len(layers)!=36:raise ValueError('Official layer inventory changed')
    for i,layer in enumerate(layers):
        row=audit['layers'][i]
        if i<4:
            actual=transport.all_tensors(layer)
            if any(v.device.type!='cuda' for v in actual.values()):raise ValueError('Resident placement changed')
        else:
            hook=hooks[i-4];actual=hook.weights
            if hook.active or any(p.device.type!='meta' for p in layer.parameters()):raise ValueError('Incomplete decoder offload')
            if any(row[k]!=count for k in ('pre_issued','pre_completed','post_completed')) or row['failed_pre']:
                raise ValueError('Layer and model call accounting disagree')
            if any(v.device.type!='cpu' or not v.is_pinned() for v in actual.values()):raise ValueError('Pinned state moved')
        if {name:transport.tensor_hash(v) for name,v in actual.items()}!=row['original_sha256']:
            raise ValueError('Official tensor bytes changed')
    audit.update(status='verified',expected_forwards=count,official_tensor_bytes_unchanged=True,rss_final_bytes=transport.check_rss())
    return audit


def strict_v5_format(raw):
    """Audit the whole response, never extract a substring or repair its text."""
    result=dict(whole_output_closed=False,strict_json_syntax=False,strict_v5_structure=False,
                wrapper='plain_json',waypoint_count=0)
    text=raw.strip()
    if text.startswith('```'):
        result['wrapper']='json_fence'
        match=re.fullmatch(r'```json\s*(.*?)\s*```',text,re.DOTALL)
        if match is None:
            result['error']='Unclosed or non-whole official JSON fence';return result
        text=match.group(1)
    def reject_constant(value):raise ValueError('Nonfinite JSON constant: '+value)
    def unique_keys(pairs):
        obj={}
        for key,value in pairs:
            if key in obj:raise ValueError('Duplicate JSON key: '+key)
            obj[key]=value
        return obj
    try:
        entries=json.loads(text,parse_constant=reject_constant,object_pairs_hook=unique_keys)
        result.update(whole_output_closed=True,strict_json_syntax=True)
        if not isinstance(entries,list) or not entries:raise ValueError('Nonempty waypoint array required')
        for entry in entries:
            if not isinstance(entry,dict):raise ValueError('Every entry must be a waypoint object')
            point=entry.get('point_3d')
            if (not isinstance(point,list) or len(point)!=3 or
                    any(type(x) not in (int,float) or not np.isfinite(x) for x in point)):
                raise ValueError('Every point_3d must contain exactly three finite JSON numbers')
            if entry.get('gripper') not in ('close','open','none'):raise ValueError('Explicit official gripper state required')
            if 'label' in entry and not isinstance(entry['label'],str):raise ValueError('Optional label must be a string')
        result.update(strict_v5_structure=True,waypoint_count=len(entries))
    except (ValueError,TypeError,OverflowError,RecursionError) as error:
        result['error']=str(error)
    return result


def parse_output(raw,sample,folder,ended_eos,limited):
    from hamster3d.inference.postprocessing import parse_trajectory
    strict=strict_v5_format(raw)
    result=dict(ended_with_eos=bool(ended_eos),limit_reached=bool(limited),quality_metrics=None,task_success=None,
                strict_format=strict)
    try:
        points,actions=parse_trajectory(raw,'v5')
        camera,world=original.uvd_to_world(points,sample['observed']['camera_intrinsics'],sample['observed']['camera_extrinsics'],*sample['rgb'].size)
        result.update(official_waypoints=points,official_actions=actions,official_parse_nonempty=bool(points),
            coordinate_projection_completed=True,complete_parsed_output=bool(ended_eos and not limited and
                strict['strict_v5_structure'] and len(points)==strict['waypoint_count']))
        np.savez_compressed(folder/'predicted_coordinates.npz',uvd=np.asarray(points),camera_xyz=camera,world_xyz=world,
            camera_intrinsics=sample['observed']['camera_intrinsics'],camera_extrinsics=sample['observed']['camera_extrinsics'])
    except Exception as error:
        result.update(official_parse_nonempty=False,complete_parsed_output=False,coordinate_projection_completed=False,
                      parse_exception=type(error).__name__,parse_message=str(error))
    # The official parser may fall back to v3; retain its output but never repair,
    # append JSON syntax, or label a time/token-truncated response complete.
    return result


def generate_one(model,processor,inputs,sample,folder,historical,reference_logits,ledger,persist,seconds):
    import torch
    folder.mkdir();row=dict(id=prior.FIRST,state='issued',generate_calls=1,candidates_issued=1,
        forwards=[],generated_tokens=0,logit_prefix_comparisons=[])
    ledger.append(row);persist();prompt=inputs['input_ids'][0].detach().cpu().tolist()
    class Stream(original.TokenAudit):
        def put(self,value):
            super().put(value)
            if self.prompt_seen:
                np.save(folder/'streamed_generated_token_ids.npy',np.asarray(self.generated_ids,dtype=np.int64),allow_pickle=False)
                (folder/'streamed_raw_output.txt').write_text(processor.decode(self.generated_ids,skip_special_tokens=True),encoding='utf8')
                row['generated_tokens']=len(self.generated_ids);persist()
    stream=Stream(prompt);started=time.perf_counter();pending=[];error=None;ended_eos=False;captured=[]
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();model.model.rope_deltas=None
    def pre(module,args,kwargs):
        if len(row['forwards'])>=1024:raise RuntimeError('1025th model forward refused before execution')
        item=dict(index=len(row['forwards']),state='issued',input_tokens=int(kwargs['input_ids'].shape[1]))
        row['forwards'].append(item);persist();torch.cuda.synchronize();pending.append(time.perf_counter())
    def post(module,args,kwargs,result):
        torch.cuda.synchronize();index=len(row['forwards'])-1;copy_start=time.perf_counter()
        if index<8:
            values=result.logits.detach().cpu().contiguous().clone();captured.append(values)
            other=reference_logits[index]
            same=bool(values.dtype==other.dtype and values.shape==other.shape and torch.equal(values,other))
            finite=bool(torch.isfinite(values).all())
            row['logit_prefix_comparisons'].append(dict(index=index,exact=same,finite=finite,
                shape=list(values.shape),dtype=str(values.dtype),sha256=transport.tensor_hash(values)))
            if not (same and finite):raise ValueError('Full request first8 logits changed from sealed exact8')
        else:
            if not bool(torch.isfinite(result.logits).all()):raise ValueError('Nonfinite output logits')
        row['forwards'][-1].update(state='completed',seconds=time.perf_counter()-pending.pop(),
            check_and_CPU_capture_seconds=time.perf_counter()-copy_start,prefix_logit_CPU_capture=index<8,
            rss_bytes=transport.check_rss())
        persist()
        if torch.cuda.max_memory_reserved()>transport.CAP_BYTES:raise RuntimeError('Fixed VRAM cap exceeded')
    prehook=model.register_forward_pre_hook(pre,with_kwargs=True);posthook=model.register_forward_hook(post,with_kwargs=True)
    try:
        with original.deadline(seconds),torch.inference_mode():
            outputs=model.generate(**inputs,max_new_tokens=1024,do_sample=False,temperature=None,top_p=None,streamer=stream)
        torch.cuda.synchronize();tokens=outputs[:,len(prompt):].detach().cpu().tolist()[0]
        if tokens!=stream.generated_ids:raise ValueError('Stream/final token identity mismatch')
        eos=model.generation_config.eos_token_id
        eos=eos if isinstance(eos,(list,tuple)) else [eos]
        ended_eos=bool(tokens and tokens[-1] in eos)
        if len(row['forwards'])!=len(tokens):raise ValueError('One greedy token per actual forward required')
        if len(captured)!=8:raise ValueError('Known unchanged first8 prefix was not completed')
        check=prefix_check(tokens,historical)
        if not check['all_compared_equal']:raise ValueError('Historical87 prefix changed')
        row['state']='completed'
    except BaseException as caught:
        error=caught;row.update(state='failed',error_type=type(caught).__name__,error=str(caught))
    finally:
        prehook.remove();posthook.remove()
        tokens=stream.generated_ids;raw=processor.decode(tokens,skip_special_tokens=True)
        np.save(folder/'streamed_generated_token_ids.npy',np.asarray(tokens,dtype=np.int64),allow_pickle=False)
        (folder/'streamed_raw_output.txt').write_text(raw,encoding='utf8')
        torch.save(captured,folder/'first8_output_logits.pt')
        limited=error is not None or len(tokens)>=1024
        parsed=parse_output(raw,sample,folder,ended_eos,limited)
        row.update(generated_tokens=len(tokens),forward_calls=len(row['forwards']),
            completed_forwards=sum(r['state']=='completed' for r in row['forwards']),
            elapsed_seconds=time.perf_counter()-started,request_deadline_seconds=seconds,
            historical_prefix=prefix_check(tokens,historical),parse=parsed,
            max_memory_allocated_bytes=torch.cuda.max_memory_allocated(),max_memory_reserved_bytes=torch.cuda.max_memory_reserved())
        write(folder/'result.json',row);persist()
    if error is not None:raise error
    return row


def run(config_path,output):
    config=validate_policy(read(config_path));output=Path(output)
    if output!=OUTPUT_ROOT:raise ValueError('Only the independent registered fresh output is allowed')
    output.mkdir(parents=True,exist_ok=False);started=time.perf_counter();ledger=[];torch=None;hooks=[];audit={}
    status=dict(protocol=PROTOCOL,status='running',pid=os.getpid(),code_commit=os.environ.get('CODE_COMMIT'),
        source_sha256=sha(__file__),config_sha256=sha(config_path),start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        optimizer_steps=0,quality_metrics=None,task_success=None)
    persist=lambda:write(output/'call_ledger.json',ledger)
    write(output/'status.json',status);persist()
    try:
        if (sys.platform!='linux' or os.sched_getaffinity(0)!={0} or os.environ.get('CUDA_VISIBLE_DEVICES')!=original.GPU_UUID
                or any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))):
            raise ValueError('Registered CPU0/GPU1 and single-thread runtime required')
        for name,item in config['dependency_sources'].items():
            if sha(REPO/name)!=item['sha256'] or (REPO/name).stat().st_size!=item['bytes']:raise ValueError('Immutable dependency changed: '+name)
        proof=verify_exact8(config);write(output/'exact8_gate.json',proof)
        prior_config=prior.validate_policy(read(REPO/'configs/hamster3d_transport_exact8_v1.json'))
        historical=prior.preserve_failure(prior_config,output)
        import accelerate
        if importlib.metadata.version('accelerate')!='1.10.1':raise ValueError('Accelerate revision changed')
        for name,digest in prior_config['installed_source_sha256'].items():
            if sha(Path(accelerate.__file__).parent/name)!=digest:raise ValueError('Installed Accelerate source changed')
        for key in ('HF_HUB_OFFLINE','TRANSFORMERS_OFFLINE','HF_HUB_DISABLE_IMPLICIT_TOKEN','XFORMERS_DISABLED'):os.environ[key]='1'
        with original.offline_network():
            assets,receipts=original.verify_prepared_assets();sample,input_hashes=prior.load_first()
            write(output/'preflight.json',dict(policy=EXPECTED,input_sha256=input_hashes,preparation_receipts=receipts,
                exact8_gate_sha256=sha(output/'exact8_gate.json'),dependency_sources=config['dependency_sources'],read_labels=False))
            import torch as torch_module
            torch=torch_module;torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(0);torch.cuda.set_device(0)
            properties=torch.cuda.get_device_properties(0)
            cap=min(int(properties.total_memory*.35),int(8.4*2**30))
            if cap!=transport.CAP_BYTES:raise ValueError('Actual original35% GPU cap differs')
            available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
            if available<transport.RSS_CAP_BYTES:raise RuntimeError('Less than32GiB available CPU RAM')
            torch.cuda.set_per_process_memory_fraction(cap/properties.total_memory,0)
            status.update(cuda_cap_bytes=cap,available_CPU_before_load_bytes=available)
            reference_logits=torch.load(PRIOR_ROOT/'probe/pinned_layer/full_output_logits.pt',map_location='cpu',weights_only=True)
            if len(reference_logits)!=8 or any(x.shape!=(1,1,151936) or x.dtype!=torch.bfloat16 or not bool(torch.isfinite(x).all()) for x in reference_logits):
                raise ValueError('Sealed first8 actual logits malformed')
            load_start=time.perf_counter()
            with original.deadline(remaining(started,600)):
                model,processor=original.load_offloaded(ROOT/assets['model_relative'],ROOT/assets['code_relative'],output,cap)
            status['model_load_seconds']=time.perf_counter()-load_start;transport.check_rss()
            prepare_start=time.perf_counter()
            with original.deadline(remaining(started,180)):
                inputs=prior.prepare_inputs(processor,sample,output/'prepared_input')
            status['input_prepare_seconds']=time.perf_counter()-prepare_start
            identity=prior.input_fingerprint(dict(inputs));write(output/'model_input_fingerprint.json',identity)
            if identity!=read(PRIOR_ROOT/'probe/model_input_fingerprint.json'):raise ValueError('True model input bytes changed')
            status['load_prepare_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
            status['load_prepare_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();conversion=time.perf_counter()
            with original.deadline(remaining(started,180)):
                hooks,audit=transport.install_transport(model,lambda a:write(output/'transport_audit.json',a))
            status['transport_conversion_seconds']=time.perf_counter()-conversion
            status['conversion_peak_reserved_bytes']=torch.cuda.max_memory_reserved();status['conversion_peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            result=generate_one(model,processor,inputs,sample,output/prior.FIRST,historical.tolist(),reference_logits,ledger,persist,remaining(started,900))
            if prior.input_fingerprint(dict(inputs))!=identity:raise ValueError('Generation mutated observed inputs')
            verify_start=time.perf_counter()
            with original.deadline(remaining(started,180)):
                audit=verify_transport_after(model,hooks,audit,result['forward_calls']);write(output/'transport_audit.json',audit)
            status['post_tensor_verification_seconds']=time.perf_counter()-verify_start
            if any(sha(Path(p))!=v for p,v in input_hashes.items()):raise ValueError('Original inputs changed')
            verify_exact8(config)
            status.update(status='completed',exit_code=0,complete_parsed_output=result['parse']['complete_parsed_output'])
    except BaseException as error:
        import traceback
        status.update(status='failed',exit_code=1,error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc())
    finally:
        elapsed=time.perf_counter()-started
        status.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=elapsed,
            generate_calls=len(ledger),candidates_issued=len(ledger),candidates_unattempted=1-len(ledger),
            forward_calls=sum(len(r['forwards']) for r in ledger),generated_tokens=sum(r.get('generated_tokens',0) for r in ledger),
            gpu_reservation_hours=elapsed/3600 if torch is not None else 0,cost_scope='Body through exit-status preparation includes nested evidence/load/conversion/request/capture/postcheck; final status/index serialization is outside this timer; record_job outer wall includes all',
            active_offload_layers_at_exit=sum(bool(h.active) for h in hooks))
        if torch is not None and torch.cuda.is_initialized():
            import resource
            status.update(process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                whole_max_memory_reserved_bytes=max([torch.cuda.max_memory_reserved(),status.get('load_prepare_peak_reserved_bytes',0),status.get('conversion_peak_reserved_bytes',0)]+[r.get('max_memory_reserved_bytes',0) for r in ledger]),
                whole_max_memory_allocated_bytes=max([torch.cuda.max_memory_allocated(),status.get('load_prepare_peak_allocated_bytes',0),status.get('conversion_peak_allocated_bytes',0)]+[r.get('max_memory_allocated_bytes',0) for r in ledger]))
        if audit:write(output/'transport_audit.json',audit)
        write(output/'status.json',status);persist()
        write(output/'artifact_index.json',{p.relative_to(output).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size)
            for p in output.rglob('*') if p.is_file() and p.name!='artifact_index.json'})
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();raise SystemExit(run(args.config,args.output)['exit_code'])
