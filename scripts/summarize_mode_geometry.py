"""Seed-complete comparison with layout-family paired intervals; no best seed."""
import argparse
import json
import numpy as np
from scripts.mode_geometry_experiment import RUN,read,write,sha


def main(groups,analysis,output):
    result=read(RUN/analysis/'RESULTS.json');models=result['models'];out=RUN/output;out.mkdir(exist_ok=False)
    families=sorted(models['parent']['families']);assert len(families)==32
    draws=np.random.default_rng(6100911).integers(32,size=(10000,32))
    tables={};vectors={};streams={};initial={}
    keys=['valid8','distinct8','recall8','valid4','distinct4','retention','adaptation']
    for g,prefix in (v.split('=') for v in groups):
        rows=[];vv=[]
        for seed in (0,1,2):
            name=prefix+'_seed'+str(seed);r=models[name+':adaptive'];folder=RUN/name/'eval_adaptive'
            data=read(folder/'rows.json');train=read(RUN/name/'summary.json')
            streams.setdefault(str(seed),{})[g]=train['stream_sha256'];initial.setdefault(str(seed),{})[g]=train['initial_tensor_sha256']
            rows.append(dict(seed=seed,valid8=r['raw']['valid_fraction'],distinct8=r['raw']['distinct'],recall8=r['raw']['recall'],
                valid4=r['selected']['valid_fraction'],distinct4=r['selected']['distinct'],retention=r['retention'],
                adaptation=r['adaptation_fraction'],lost=r['lost_of_1872'],recovered=r['recovered_of_297'],adaptation_counts=r['adaptation_counts']))
            family_rows=[]
            for f in families:
                rr=[x for x in data if x['family']==f];z=r['families'][f]
                family_rows.append([np.mean([x['raw']['valid_fraction'] for x in rr]),np.mean([x['raw']['distinct'] for x in rr]),
                    np.mean([x['raw']['recall'] for x in rr]),np.mean([x['selected']['valid_fraction'] for x in rr]),
                    np.mean([x['selected']['distinct'] for x in rr]),z['retained'],z['adapted']])
            vv.append(family_rows)
        tables[g]=dict(seeds=rows,mean={k:float(np.mean([r[k] for r in rows])) for k in keys+['lost','recovered']})
        vectors[g]=np.asarray(vv).mean(0)
    # Same pre-assignment random target draws, exact across arms per seed.
    assert all(len(set(x.values()))==1 for x in streams.values()),streams
    assert all(len(set(x.values()))==1 for x in initial.values()),initial
    denom=np.array([[models['parent']['families'][f]['denom'],models['parent']['families'][f]['adaptation_denom']] for f in families])
    comparisons={}
    for ga in tables:
        for gb in tables:
            if ga==gb:continue
            delta=vectors[ga]-vectors[gb];boot=delta[draws].mean(1)
            boot[:,5:]=delta[draws][:,:,5:].sum(1)/denom[draws].sum(1)
            comparisons[ga+' minus '+gb]={k:dict(delta=tables[ga]['mean'][k]-tables[gb]['mean'][k],CI95=np.quantile(boot[:,i],[.025,.975]).tolist()) for i,k in enumerate(keys)}
    payload=dict(groups=tables,comparisons=comparisons,stream_hashes=streams,initial_tensor_hashes=initial,
        analysis_sha256=sha(RUN/analysis/'RESULTS.json'),scope='Three continuation seeds; intervals bootstrap32 reused layout families after averaging seeds; not independent TEST or pretraining uncertainty')
    write(out/'RESULTS.json',payload)
    table=['|Method|Seed|Validity@8|Distinct@8|Known recall@8|Validity@4|Distinct@4|Retain /2169|Repair /270|Lost /1872|Recovered /297|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for g,t in tables.items():
        for row in t['seeds']+[dict(t['mean'],seed='mean')]:
            table.append('|%s|%s|%.2f%%|%.3f|%.2f%%|%.2f%%|%.3f|%.2f|%.2f|%.2f|%.2f|'%(g,row['seed'],100*row['valid8'],row['distinct8'],100*row['recall8'],100*row['valid4'],row['distinct4'],2169*row['retention'],270*row['adaptation'],row['lost'],row['recovered']))
    (out/'TABLE.md').write_text('\n'.join(table)+'\n')
    print(json.dumps(dict(groups={g:t['mean'] for g,t in tables.items()},comparisons=comparisons)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--groups',nargs='+',required=True);p.add_argument('--analysis',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.groups,a.analysis,a.output)
