"""Three full paired generator continuations; conditional crossed/family intervals."""
import argparse,json
import numpy as np
from research_feasible_space_v1.prepare import RUN
from scripts.run_observed_probability import read,write

def main(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);types=['xyz','bounded','projection','center','boundcenter','refitted_projection','refitted_center','refitted_boundcenter'];data={};results={};paired=[]
    for kind in types:
        arrays=[];values=[]
        for seed in range(3):
            n=('refitted_control_%s_seed%d'%(kind[len('refitted_'):],seed) if kind.startswith('refitted_') else 'refreshed_tapered_%s_seed%d'%(kind,seed))
            folder=RUN/n/'eval_adaptive';rows=read(folder/'rows.json');m=read(folder/'metrics.json')
            families=sorted(set(r['family'] for r in rows));assert len(rows)==288 and len(families)==32
            arrays.append([[np.mean([r['raw']['distinct'] for r in rows if r['family']==f]),np.mean([r['raw']['valid_fraction'] for r in rows if r['family']==f]),
                np.mean([r['selected']['distinct'] for r in rows if r['family']==f]),np.mean([r['selected']['valid_fraction'] for r in rows if r['family']==f])] for f in families])
            values.append(dict(seed=seed,U8=m['raw']['distinct'],V8=m['raw']['valid_fraction'],U4=m['selected']['distinct'],V4=m['selected']['valid_fraction'],known_recall=m['raw']['recall'],cell_feasibility=m['corridor_feasibility']))
        data[kind]=np.array(arrays);results[kind]=values
    for seed in range(3):
        a=read(RUN/('tapered_xyz_seed%d'%seed)/'SUMMARY.json');b=read(RUN/('tapered_bounded_seed%d'%seed)/'SUMMARY.json')
        pair=dict(seed=seed,initial_tensors_equal=a['initial_tensor_sha256']==b['initial_tensor_sha256'],actual_streams_equal=a['stream']==b['stream'])
        assert pair['initial_tensors_equal'] and pair['actual_streams_equal'];paired.append(pair)
    rng=np.random.default_rng(20261009);sd=rng.integers(3,size=(10000,3));fd=rng.integers(32,size=(10000,32));comparisons={}
    for other in ('xyz','projection','center','boundcenter','refitted_projection','refitted_center','refitted_boundcenter'):
        delta=data['bounded']-data[other];mean=delta.mean((0,1));cross=delta[sd[:,:,None],fd[:,None,:]].mean((1,2));family=delta.mean(0)[fd].mean(1)
        ci=np.quantile(cross,[.025,.975],axis=0).T
        comparisons['bounded minus '+other]=dict(metrics=['U8','V8','U4','V4'],mean=mean.tolist(),crossed_CI95=ci.tolist(),family_CI95=np.quantile(family,[.025,.975],axis=0).T.tolist(),
            primary_gain=bool(mean[0]>=.15 and ci[0,0]>0),quality_guards=bool(mean[1]>=-.01 and mean[2]>=-.03 and mean[3]>=-.005))
    report=dict(seed_results=results,means={k:{m:float(np.mean([v[m] for v in vals])) for m in vals[0] if m!='seed'} for k,vals in results.items()},comparisons=comparisons,paired_streams=paired,
        accepted=all(v['primary_gain'] and v['quality_guards'] for v in comparisons.values()),scope='3generator continuation seeds on shared pretrainedC0;32reusedDEVfamilies; not independent final test',locked_access=False)
    write(out/'RESULTS.json',report);print(json.dumps(dict(means=report['means'],comparisons=comparisons,accepted=report['accepted'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);main(**vars(p.parse_args()))
