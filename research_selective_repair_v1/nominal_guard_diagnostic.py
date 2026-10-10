"""Does nominal-word protection exclude predicted realized-space repairs?

Uses only sealed current-observation fields and predicted probabilities. No
truth or native outcomes read. This is a policy-support diagnostic, not gain.
"""
import argparse,time
import numpy as np
from research_selective_repair_v1.io import RUN,SOURCE,read,write,sha
from research_selective_repair_v1.body_options import options
from research_selective_repair_v1.body_allocation import allocate,objectives
from research_selective_repair_v1.body_feedback_data import WORDS
from research_selective_repair_v1.execution_semantics import word

def run(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);tic=time.monotonic();records=[];hashes={}
    settings=read(RUN/'constraints_seed0_v1/config.json');base=settings['post_base']
    for label,kind in [('actual','actual'),('planned','planned'),('categorical','actual'),('binary','planned')]:
        folder=RUN/('body_full_%s_DEV_screen_seed0_v1'%label);file=folder/'sealed_predictions.npz'
        seal=read(folder/'SEAL.json');assert seal['current_observation_only'] and seal['prediction_sha256']==sha(file)
        hashes[str(file)]=sha(file)
        with np.load(file) as z:d={k:z[k] for k in ('ids','drafts','completed','probabilities','choices')}
        rows=[]
        for i,ident in enumerate(d['ids']):
            completed=d['completed'][i];h=np.clip(completed[:,2]-base,.025,.4)
            cfg=dict(row_x=[float(completed[j:j+2,0].mean()) for j in (0,2)],post_y=[completed[j:j+2,1].tolist() for j in (0,2)],post_heights=[float(h[j:j+2].max()) for j in (0,2)],post_base_z=base,tip_clearance_m=.02)
            candidate=options(d['drafts'][i],completed);words=[word(p,cfg) for p in candidate.reshape(24,24,3)]
            planned=np.asarray([WORDS.index(w)+1 if w in WORDS else 0 for w in words]).reshape(8,3)
            allowed=planned==planned[:,:1];p=d['probabilities'][i]
            constrained,_=allocate(p,kind,True,planned,allowed);relaxed,_=allocate(p,kind,True,planned,None)
            assert np.array_equal(constrained,d['choices'][i])
            nominal_changed=int((planned[np.arange(8),relaxed]!=planned[:,0]).sum())
            rows.append(dict(id=str(ident),changed_choices=int((constrained!=relaxed).sum()),nominal_changed=nominal_changed,
                excluded_alternatives=int((~allowed).sum()),relaxed_choices=relaxed.tolist()))
        records.append(dict(method=label,requests=len(rows),changed_requests=sum(r['changed_choices']>0 for r in rows),
            changed_choices=sum(r['changed_choices'] for r in rows),nominal_changed=sum(r['nominal_changed'] for r in rows),
            excluded_alternatives=sum(r['excluded_alternatives'] for r in rows),rows=rows))
    write(out/'SUMMARY.json',dict(records=records,seconds=time.monotonic()-tic,source_commit=SOURCE.name,input_sha256=hashes,
        scope='Current NN fields/forecast probabilities only; relaxing nominal word guard retains exactly the existing .01 per-word expected coverage and per-route predicted failure guards. No geometry/body outcome or success claim.',locked_access=False))
    print([{k:v for k,v in r.items() if k!='rows'} for r in records],flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
