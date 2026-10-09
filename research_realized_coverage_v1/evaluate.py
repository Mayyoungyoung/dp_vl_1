"""Deployable one-decode evaluation, with original checker and frozen return rule."""
import argparse
import json
import time
import numpy as np
from research_realized_coverage_v1.core import *
from research_realized_coverage_v1.allocator import SuccessHead,SetUtility,allocate
from scripts.evaluate_paired_modes import inputs_for,references,check_candidates
from scripts.research_v3_audit import coverage,average,plain

def evaluate(name,head=None,checkpoint=BASE,start_head=None):
    torch_setup();model,ck=load_generator(checkpoint);model.requires_grad_(False)
    scorer=load_scored_planner(Q,'cuda');scorer.requires_grad_(False)
    proposal=None;kind=None
    if head:
        saved=torch.load(RUN/head,map_location='cpu',weights_only=False);kind=saved['settings']['kind']
        assert saved['settings']['generator_sha256']==sha(checkpoint),'Stale feedback/head generator binding'
        proposal=(SuccessHead() if kind=='success' else SetUtility()).cuda();proposal.load_state_dict(saved['model']);proposal.eval()
    starter=None
    if start_head:
        initial=torch.load(RUN/start_head,map_location='cpu',weights_only=False)
        assert initial['settings']['kind']=='success' and initial['settings']['generator_sha256']==sha(checkpoint)
        starter=SuccessHead().cuda();starter.load_state_dict(initial['model']);starter.eval()
    out=RUN/name/'eval_adaptive';out.mkdir(parents=True,exist_ok=False)
    rows=[r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL'];assert len(rows)==288
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    with np.load(old.SUPPORT/'support.npz') as z:known={str(k):set(z['modes'][i,z['mask'][i]]) for i,k in enumerate(z['ids']) if z['splits'][i]=='DEV_MODEL'}
    results=[];pools={k:[] for k in ('paths','events','q','ids','mode_ids','variant_ids')};hashes={};times=[];alloc_time=[];cost=dict(replacements=0,query_sets_evaluated=0)
    for row in rows:
        inp=inputs_for(row,labels[row['id']],DATA/'export/qwen_cache',torch,hashes)
        runner=SceneRunner(model,scorer,inp)
        with torch.inference_mode():
            m=torch.as_tensor(runner.base[None],device='cuda');v=torch.as_tensor(runner.base_variants[None],device='cuda')
            torch.cuda.synchronize();tic=time.monotonic()
            if starter:m,v,_=allocate('success',starter,runner.context,m,v)
            if proposal:
                m,v,c=allocate(kind,proposal,runner.context,m,v)
                for k in cost:cost[k]+=c[k]
            torch.cuda.synchronize();alloc_time.append(time.monotonic()-tic)
            tic=time.monotonic();p,e,q=runner.run(m.cpu().numpy(),v.cpu().numpy());torch.cuda.synchronize();times.append(time.monotonic()-tic)
        ref=references(labels[row['id']]);a=assess(p,e,q,ref);p,e,q=p[0],e[0],q[0]
        _,cc=check_candidates(p,e,ref['label'],ref['current'],ref['truth'],ref['config'])
        assert a['valid'][0].tolist()==[c['TipValid'] for c in cc]
        words=[VOCAB[w] if w>=0 else None for w in a['words'][0]];valid=a['valid'][0];chosen=a['selected'][0]
        results.append(dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],variant=row['parent_id'].rsplit('_',1)[1],
            raw=coverage(words,valid,known[row['id']],list(range(8))),selected=coverage(words,valid,known[row['id']],chosen),
            words=words,valid=valid.tolist(),q=q.tolist(),assigned_modes=[VOCAB[j] for j in m[0].tolist()],
            raw_words=[VOCAB[j] if j>=0 else None for j in a['raw_words'][0]],selected_indices=chosen.tolist(),candidates=cc,
            condition_hit=float(np.mean(valid&(a['words'][0]==m[0].cpu().numpy())))))
        for k,val in dict(paths=p,events=e,q=q,ids=row['id'],mode_ids=m[0].cpu().numpy(),variant_ids=v[0].cpu().numpy()).items():pools[k].append(val)
    np.savez_compressed(out/'pool.npz',**pools)
    report=dict(raw=average([r['raw'] for r in results]),selected=average([r['selected'] for r in results]),
        condition_hit=float(np.mean([r['condition_hit'] for r in results])),generator_sha256=sha(checkpoint),head_sha256=sha(RUN/head) if head else None,
        start_head_sha256=sha(RUN/start_head) if start_head else None,
        scorer_sha256=sha(Q),pool_sha256=sha(out/'pool.npz'),cost=dict(cost,decoded_sets=288,generated_routes=2304,
            proposal_ms_mean=1000*np.mean(alloc_time),decode_and_q_ms_mean=1000*np.mean(times)),input_hashes=hashes,locked_access=False)
    write(out/'rows.json',plain(results));write(out/'metrics.json',plain(report));print(json.dumps(plain({k:v for k,v in report.items() if k!='input_hashes'})),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--head');p.add_argument('--start-head');p.add_argument('--checkpoint',type=type(BASE),default=BASE);a=p.parse_args();evaluate(**vars(a))
