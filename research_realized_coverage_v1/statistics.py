"""Paired allocation-head seeds and layout-family summaries, never seed selection."""
import argparse
import json
import numpy as np
from research_realized_coverage_v1.core import RUN,read,write

def main(output):
    arms={'success':['success_context_step2400']+['success_rep_seed%d'%s for s in range(1,5)],
          'dense':['dense_context_step2400']+['dense_rep_seed%d'%s for s in range(1,5)]}
    rows={};summary={};seed_results={}
    for arm,names in arms.items():
        rows[arm]=[];seed_results[arm]=[]
        for seed,name in enumerate(names):
            data=read(RUN/name/'eval_adaptive/rows.json');families=sorted({r['family'] for r in data})
            assert len(families)==32 and len(data)==288
            arr=np.array([[np.mean([r[g][k] for r in data if r['family']==f]) for g,k in
                [('raw','distinct'),('raw','valid_fraction'),('selected','distinct'),('selected','valid_fraction'),('raw','recall')]] for f in families])
            rows[arm].append(arr);seed_results[arm].append(dict(seed=seed,model=name,metrics=arr.mean(0).tolist()))
        rows[arm]=np.array(rows[arm]);summary[arm]=rows[arm].mean((0,1)).tolist()
    differences=rows['dense']-rows['success'];rng=np.random.default_rng(910095)
    family_draw=rng.integers(32,size=(10000,32));seed_draw=rng.integers(5,size=(10000,5))
    family_only=differences.mean(0)[family_draw].mean(1)
    both=np.array([differences[s][:,f].mean((0,1)) for s,f in zip(seed_draw,family_draw)])
    streams={}
    for seed in range(5):
        ss=read(RUN/('success_context_fit%d'%seed)/'SUMMARY.json')['stream']
        ds=read(RUN/('dense_context_fit%d'%seed)/'SUMMARY.json')['stream'];assert ss==ds
        streams[str(seed)]=ss
    result=dict(metrics=['U8','V8','U4','V4','known_recall'],seed_results=seed_results,means=summary,
        dense_minus_success=differences.mean((0,1)).tolist(),family95=np.quantile(family_only,[.025,.975],axis=0).T.tolist(),
        seed_family95=np.quantile(both,[.025,.975],axis=0).T.tolist(),streams=streams,
        scope='5 allocation-head training seeds conditional on fixed C0 and shared TRAIN feedback; decoder/VLM unchanged; reusedDEV32, not independent final test')
    write(RUN/output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='FIVE_SEED.json');a=p.parse_args();main(a.output)
