"""Crossed seed/family retention/repair diagnosis on unchanged parent witnesses."""
import argparse,json
import numpy as np
from research_feasible_space_v1.prepare import RUN
from scripts.run_observed_probability import read,write


def main(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);d=read(RUN/'fixed_witness_refitted_v2/RESULTS.json')['models']
    families=sorted(d['parent']['families']);den=np.array([[d['parent']['families'][f]['denom'],d['parent']['families'][f]['adaptation_denom']] for f in families])
    assert den.sum(0).tolist()==[2169,270]
    models={};arrays={};kinds=['xyz','bounded','refitted_projection','refitted_center','refitted_boundcenter']
    for k in kinds:
        results=[];values=[]
        for seed in range(3):
            key=('refitted_control_%s_seed%d:adaptive'%(k[len('refitted_'):],seed) if k.startswith('refitted_') else 'refreshed_tapered_%s_seed%d:adaptive'%(k,seed))
            m=d[key];results.append(dict(seed=seed,retention=m['retention'],repair=m['adaptation_fraction'],retained=m['retained'],
                         lost=m['lost_of_1872'],recovered=m['recovered_of_297'],repaired=m['adaptation_counts'].get('valid_same_mode_repair',0)))
            values.append([[m['families'][f]['retained'],m['families'][f]['adapted']] for f in families])
        arrays[k]=np.array(values);models[k]=results
    rng=np.random.default_rng(641010);sd=rng.integers(3,size=(10000,3));fd=rng.integers(32,size=(10000,32));comparisons={}
    for k in kinds:
        if k=='bounded':continue
        difference=arrays['bounded']-arrays[k]
        boot=difference[sd[:,:,None],fd[:,None,:]].sum((1,2))/(3*den[fd].sum(1))
        comparisons['bounded minus '+k]=dict(metrics=['retention','same_mode_repair'],mean=(difference.sum((0,1))/(3*den.sum(0))).tolist(),crossed_CI95=np.quantile(boot,[.025,.975],axis=0).T.tolist())
    report=dict(seed_results=models,means={k:{metric:float(np.mean([r[metric] for r in rows])) for metric in ('retention','repair','retained','lost','recovered','repaired')} for k,rows in models.items()},
                comparisons=comparisons,denominators=dict(survival=2169,adaptation=270),scope='Fixed historical parent witnesses;3shared-C0continuations;32reusedDEVfamilies',locked_access=False)
    write(out/'RESULTS.json',report);print(json.dumps(dict(means=report['means'],comparisons=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);main(**vars(p.parse_args()))
