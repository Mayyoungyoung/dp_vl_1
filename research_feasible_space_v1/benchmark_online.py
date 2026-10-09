"""Actual RGB-D/language → frozen VLM → eight paths → frozen four-return timing."""
import argparse,hashlib,json,time
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
from scripts.observation_cache_qwen import REVISION,ALLOWED
from scripts.run_observed_probability import ROOT,read,write,sha,lines,torch_setup
from scripts.train_observed_geometry import read_geometry
from research_realized_coverage_v1.core import DATA,Q
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.deployment import load_planner


def main(name):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
    model_path=ROOT/'data/qwen3-vl-2b-instruct-89644892';provenance=read(model_path/'provenance.json')
    assert provenance['revision']==REVISION and provenance['all_hashes_verified']
    vlm=Qwen3VLForConditionalGeneration.from_pretrained(model_path,torch_dtype=torch.bfloat16,local_files_only=True,attn_implementation='sdpa',device_map={'':'cuda'})
    vlm.requires_grad_(False).eval();processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,min_pixels=64*32*32,max_pixels=256*32*32)
    ck=RUN/'tapered_bounded_seed0/last.pt';head=RUN/'success_tapered_bounded_seed0/last.pt';planner=load_planner(ck,Q,head,'cuda')
    torch.cuda.synchronize();load_seconds=time.monotonic()-tic
    rows=[r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL'][:8]
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    inputs_sha={};records=[];calls=[];hook=planner.generator.relative_output.register_forward_hook(lambda module,args,result:calls.append(tuple(result.shape)))

    def run(row):
        assert set(row)==ALLOWED
        torch.cuda.synchronize();start=time.monotonic();rgb=Image.open(row['image']).convert('RGB')
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':row['instruction']}]}]
        text=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        tokens=processor(text=[text],images=[rgb],return_tensors='pt').to('cuda')
        with torch.inference_mode():
            states=vlm.model(**tokens,use_cache=False,return_dict=True).last_hidden_state[0]
            selected=states[tokens.attention_mask[0].bool()].float()
            features=torch.cat((selected.mean(0),selected[-1]))[None]
            geo=read_geometry(row['image'],labels[row['id']]['observation'],2)
            with np.load(labels[row['id']]['observation']) as a:current=np.r_[a['gripper_pose'],np.asarray(a['gripper_open']).reshape(1)].astype(np.float32)
            inp=dict(features=features,current=torch.as_tensor(current[None],device='cuda'),**{k:torch.as_tensor(v[None],device='cuda') for k,v in geo.items()})
            result=planner(**inp)
        torch.cuda.synchronize();elapsed=time.monotonic()-start
        assert result['paths'].shape==(1,8,24,3) and result['selected_paths'].shape==(1,4,24,3)
        return elapsed,features[0].cpu().numpy(),result['selected_indices'][0].cpu().tolist()

    run(rows[0]) # one explicitly excluded warmup, still charged to the job ledger
    torch.cuda.reset_peak_memory_stats()
    for row in rows:
        elapsed,features,chosen=run(row)
        key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
        cache=DATA/'export/qwen_cache'/(key+'.npz')
        with np.load(cache) as a:cached=np.r_[a['mean_hidden'].reshape(-1),a['last_hidden'].reshape(-1)]
        error=float(np.abs(features-cached).max());assert error<1e-5,'Frozen online feature extraction changed'
        for p in (row['image'],labels[row['id']]['observation'],cache):inputs_sha[str(p)]=sha(p)
        records.append(dict(id=row['id'],online_seconds=elapsed,maximum_cached_feature_error=error,selected_indices=chosen))
    hook.remove();assert len(calls)==9 and all(s[:2]==(1,8) for s in calls)
    report=dict(requests=8,warmup_requests=1,load_seconds=load_seconds,online_ms_mean=1000*float(np.mean([r['online_seconds'] for r in records])),
                online_ms_median=1000*float(np.median([r['online_seconds'] for r in records])),peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                one_decode_per_request=True,records=records,input_sha256=inputs_sha,generator_sha256=sha(ck),head_sha256=sha(head),scorer_sha256=sha(Q),
                model_provenance_sha256=sha(model_path/'provenance.json'),scope='Actual RGB disk read/tokenization/frozen Qwen+point backprojection+two observation encoders+8route decode+full q+4return; excludes checker and model load',locked_access=False)
    write(out/'RESULTS.json',report);print(json.dumps({k:v for k,v in report.items() if k!='input_sha256'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);main(**vars(p.parse_args()))
