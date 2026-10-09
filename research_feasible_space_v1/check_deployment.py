"""Actual full-API replay and timing; inputs whitelist remains observation-only."""
import argparse,json,time
import numpy as np
import torch
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.deployment import load_planner
from research_realized_coverage_v1.core import Q,DATA,lines,read,write,torch_setup
from scripts.evaluate_paired_modes import inputs_for

def main(name,checkpoint,head,evaluation):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False);model=load_planner(checkpoint,Q,head,'cuda')
    with np.load(RUN/evaluation/'eval_adaptive/pool.npz') as z:pool={k:z[k] for k in z.files}
    expected={r['id']:r['selected_indices'] for r in read(RUN/evaluation/'eval_adaptive/rows.json')}
    rows={r['id']:r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL'}
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'};hashes={};errors=[];calls=[];times=[]
    hook=model.generator.relative_output.register_forward_hook(lambda module,args,result:calls.append(result.shape))
    for i in range(16):
        inp=inputs_for(rows[str(pool['ids'][i])],labels[str(pool['ids'][i])],DATA/'export/qwen_cache',torch,hashes)
        with torch.inference_mode():
            torch.cuda.synchronize();tic=time.monotonic();result=model(**inp);torch.cuda.synchronize();times.append(time.monotonic()-tic)
        for k in ('paths','events','q'):errors.append(float(np.max(np.abs(result[k][0].cpu().numpy()-pool[k][i]))))
        assert np.array_equal(result['mode_ids'][0].cpu().numpy(),pool['mode_ids'][i])
        assert np.array_equal(result['selected_indices'][0].cpu().numpy(),expected[str(pool['ids'][i])])
        assert result['selected_paths'].shape==(1,4,24,3)
    hook.remove();assert len(calls)==16 and max(errors)<1e-6
    report=dict(maximum_path_event_q_error=max(errors),decoded_sets=len(calls),requests=16,one_decode_per_request=True,selected_indices_exact=True,api_cached_feature_ms_mean=1000*float(np.mean(times)),
        inputs=list(inp),certificate='predicted cell membership only',locked_access=False)
    write(out/'RESULTS.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=type(RUN),required=True);p.add_argument('--head',type=type(RUN));p.add_argument('--evaluation',required=True);main(**vars(p.parse_args()))
