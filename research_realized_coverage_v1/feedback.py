"""Single-slot frozen-snapshot feedback; TRAIN fitting and DEV diagnostic separate."""
import argparse
from collections import Counter
import json
import os
import time
import numpy as np
from research_realized_coverage_v1.core import *
from scripts.evaluate_paired_modes import inputs_for,references
from scripts.research_v3_audit import plain

def collect(name,split,resume=False,limit=None,checkpoint=BASE):
    torch_setup();model,_=load_generator(checkpoint);model.requires_grad_(False)
    scorer=load_scored_planner(Q,'cuda');scorer.requires_grad_(False)
    out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    manifest=dict(generator_sha256=sha(checkpoint),scorer_sha256=sha(Q),split=split,seed=91009,
        query_design='16 modes x two rotating slots; all seven companion pairs unchanged',source_commit=os.environ.get('CODE_COMMIT'))
    if resume:
        before=read(out/'manifest.json');assert {k:v for k,v in before.items() if k!='source_commit'}=={k:v for k,v in manifest.items() if k!='source_commit'}
        write(out/('resume_'+manifest['source_commit']+'.json'),dict(original=before,current=manifest))
    else:write(out/'manifest.json',manifest)
    rows=[r for r in lines(DATA/'export/observations.jsonl') if r['split']==split]
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']==split}
    if limit:rows=rows[:limit]
    counts=Counter();omissions=Counter();deltas=[];hashes={};tic=time.monotonic();records=[]
    for index,row in enumerate(rows):
        path=out/(row['id']+'.npz')
        if not path.exists():
            inp=inputs_for(row,labels[row['id']],DATA/'export/qwen_cache',torch,hashes)
            runner=SceneRunner(model,scorer,inp);queries=interventions(runner.base,runner.base_variants,index)
            m=np.array([r[0] for r in queries]);v=np.array([r[1] for r in queries]);slot=np.array([r[2] for r in queries]);target=np.array([r[3] for r in queries])
            p,e,q=runner.run(m,v);ref=references(labels[row['id']]);a=assess(p,e,q,ref)
            np.savez_compressed(path,context=runner.context[0].cpu().numpy(),modes=m,variants=v,slot=slot,target=target,
                paths=p,events=e,q=q,**a,gain=a['utility']-a['utility'][0],
                added=a['present']&~a['present'][0],lost=a['present'][0]&~a['present'])
        with np.load(path) as z:d={k:z[k] for k in z.files}
        counts['requests']+=1;counts['decoded_sets']+=len(d['modes']);counts['checked_routes']+=8*len(d['modes'])
        valid=d['valid'][0];words=d['words'][0];raw=d['raw_words'][0];m=d['modes'][0];present=d['present'][0]
        counts['B_requested_wrong_raw_mode']+=int((raw!=m).sum())
        counts['B_valid_wrong_mode']+=int((valid&(words!=m)).sum())
        counts['C_raw_same_mode_invalid']+=int((~valid&(raw==m)).sum())
        counts['D_valid_duplicate_slots']+=int(valid.sum()-present.sum())
        counts['unrequested_incidental_valid_modes']+=sum(present[j] and j not in m for j in range(16))
        recovered=[]
        for word in range(16):
            if word in m or present[word]:continue
            hit=(d['target']==word)&d['present'][:,word]
            if not hit.any():continue
            counts['A_unrequested_recoverable_mode']+=1
            net=hit&(d['gain'][:,0]>0);safe=net&(d['gain'][:,1]>=0)&(d['gain'][:,2]>=0)&(d['gain'][:,3]>=0)
            counts['A_net_positive']+=int(net.any());counts['A_quality_nondecreasing']+=int(safe.any())
            recovered.append(dict(mode=VOCAB[word],net_positive=bool(net.any()),quality_nondecreasing=bool(safe.any())))
        counts['added_but_net_nonpositive']+=int((d['added'].any(-1)&(d['gain'][:,0]<=0)).sum())
        counts['lost_any_mode']+=int(d['lost'].any(-1).sum())
        diff=np.linalg.norm(d['paths']-d['paths'][0],axis=-1).mean(-1)
        counts['changed_companion_routes']+=sum(int((np.delete(diff[j],int(d['slot'][j]))>1e-6).sum()) for j in range(1,len(diff)))
        deltas.extend(d['gain'][1:].tolist())
        records.append(dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],baseline_utility=d['utility'][0].tolist(),recoverable=recovered,
            best_local_gain=float(d['gain'][:,0].max()),file_sha256=sha(path)))
        if (index+1)%64==0:print(json.dumps(plain(dict(requests=index+1,counts=dict(counts)))),flush=True)
    summary=dict(counts=dict(counts),baseline_mean=np.mean([r['baseline_utility'] for r in records],0).tolist(),
        sampled_best_local_gain=float(np.mean([r['best_local_gain'] for r in records])),mean_replacement_delta=np.mean(deltas,0).tolist(),
        elapsed_seconds=time.monotonic()-tic,scope='Actual single-slot diagnostic, not deployment or global reachable bound',
        input_hashes=hashes,manifest=manifest,files=records,locked_access=False)
    summary=plain(summary)
    write(out/'SUMMARY.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('files','input_hashes')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--split',choices=['TRAIN','DEV_MODEL'],required=True);p.add_argument('--resume',action='store_true');p.add_argument('--limit',type=int);p.add_argument('--checkpoint',type=type(BASE),default=BASE);a=p.parse_args()
    collect(a.name,a.split,a.resume,a.limit,a.checkpoint)
