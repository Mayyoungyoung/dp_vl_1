import argparse
import json
import time
import numpy as np
import torch
from research_realized_coverage_v1.core import RUN,BASE,Q,DATA,sha,lines,write,torch_setup
from research_realized_coverage_v1.deployment import load_planner
from scripts.evaluate_paired_modes import inputs_for

def main(name,head,starter=None):
    torch_setup();model=load_planner(BASE,Q,RUN/head,RUN/starter if starter else None,'cuda')
    with np.load(RUN/name/'eval_adaptive/pool.npz') as z:d={k:z[k] for k in z.files}
    ident=str(d['ids'][0]);row=next(r for r in lines(DATA/'export/observations.jsonl') if r['id']==ident and r['split']=='DEV_MODEL')
    label=next(r for r in lines(DATA/'export/supervision.jsonl') if r['id']==ident and r['split']=='DEV_MODEL')
    inp=inputs_for(row,label,DATA/'export/qwen_cache',torch,{})
    calls=[];hook=model.generator.head.output.register_forward_hook(lambda *args:calls.append(1))
    with torch.inference_mode():out=model(**inp)
    hook.remove();assert len(calls)==1
    error={}
    for key in ('paths','events','q'):
        value=out[key][0].cpu().numpy();error[key]=float(np.max(np.abs(value-d[key][0])));np.testing.assert_allclose(value,d[key][0],rtol=0,atol=1e-7)
    from scripts.analyze_paired_selection import select
    expected=select(d['paths'][0],d['q'][0],4);assert out['selected_indices'][0].tolist()==expected
    elapsed=[]
    with torch.inference_mode():
        for _ in range(20):
            torch.cuda.synchronize();tic=time.monotonic();model(**inp);torch.cuda.synchronize();elapsed.append(1000*(time.monotonic()-tic))
    result=dict(generator_sha256=sha(BASE),head_sha256=sha(RUN/head),errors=error,selected_exact=True,decoder_calls=1,input_keys=sorted(inp),
        additional_benchmark_decodes=20,pipeline_ms_median=float(np.median(elapsed)),
        latency_scope='One fixed DEV input,20 repeats, complete observed encoder+allocation+decode+frozenq; excludes external Qwen extraction and disk input loading')
    write(RUN/(name+'_DEPLOYMENT.json'),result);print(json.dumps(result))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--head',required=True);p.add_argument('--starter');a=p.parse_args();main(**vars(a))
