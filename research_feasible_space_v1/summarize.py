"""All completed outputs and costs, paired families, streams and actual failures."""
import argparse,json
import numpy as np
from research_feasible_space_v1.prepare import RUN
from scripts.run_observed_probability import read,write,sha

def main(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);models={};rows={};streams={};stability={}
    for f in sorted(RUN.glob('*/eval_adaptive/metrics.json')):
        n=f.parents[1].name;metric=read(f);models[n]={k:v for k,v in metric.items() if k!='input_hashes'};rows[n]=read(f.parent/'rows.json')
        with np.load(f.parent/'pool.npz') as z:
            radii=z['radii'];models[n]['width_quantiles_m']=np.quantile(radii,[0,.1,.5,.9,1]).tolist();models[n]['cells_below_2mm']=float((radii<.002).mean())
        st=read(f.parent/'stability.json')
        if st:stability[n]={k:float(np.mean([r[k] for r in st])) for k in st[0] if k!='id'}
    for f in sorted(RUN.glob('*/SUMMARY.json')):
        summary=read(f)
        if 'initial_tensor_sha256' in summary:streams[f.parent.name]={k:summary[k] for k in ('initial_tensor_sha256','stream')}
    comparisons={};rng=np.random.default_rng(261009);families=sorted(set(r['family'] for r in next(iter(rows.values()))));draw=rng.integers(len(families),size=(10000,len(families)))
    def family_metrics(n):
        return np.array([[np.mean([r['raw']['distinct'] for r in rows[n] if r['family']==f]),np.mean([r['raw']['valid_fraction'] for r in rows[n] if r['family']==f]),
            np.mean([r['selected']['distinct'] for r in rows[n] if r['family']==f]),np.mean([r['selected']['valid_fraction'] for r in rows[n] if r['family']==f])] for f in families])
    for prefix in ('screen_','envelope_','refreshed_envelope_'):
        a=prefix+'bounded_seed0';b=prefix+'xyz_seed0'
        if a not in rows or b not in rows:continue
        difference=family_metrics(a)-family_metrics(b);ci=np.quantile(difference[draw].mean(1),[.025,.975],axis=0)
        comparisons[a+' minus '+b]=dict(metrics=['U8','V8','U4','V4'],difference=difference.mean(0).tolist(),CI95=ci.T.tolist(),
            primary_gain=bool(difference.mean(0)[0]>=.15 and ci[0,0]>0),guards=bool(difference.mean(0)[1]>=-.01 and difference.mean(0)[2]>=-.03 and difference.mean(0)[3]>=-.005))
    receipts=[read(f) for f in sorted((RUN/'jobs').glob('*/receipt.json'))]
    report=dict(models=models,stability=stability,paired_family_comparisons=comparisons,training_streams=streams,
        ledger=dict(jobs=len(receipts),completed=sum(r['status']=='completed' for r in receipts),failed=sum(r['status']=='failed' for r in receipts),
            elapsed_command_seconds=sum(r.get('elapsed_seconds',0) for r in receipts),unfinished=[r['id'] for r in receipts if r['status'] not in ('completed','failed')]),
        scope='Repeated DEV_MODEL32families; seed0 exploratory unless formal seeds explicitly recorded; not independent test',locked_access=False)
    write(out/'RESULTS.json',report)
    table=['|Model|V8|U8|V4|U4|Predicted-cell feasibility|','|---|---:|---:|---:|---:|---:|']
    for n,m in models.items():table.append('|%s|%.2f%%|%.4f|%.2f%%|%.4f|%.2f%%|'%(n,100*m['raw']['valid_fraction'],m['raw']['distinct'],100*m['selected']['valid_fraction'],m['selected']['distinct'],100*m['corridor_feasibility']))
    (out/'TABLE.md').write_text('\n'.join(table)+'\n');print(json.dumps(dict(comparisons=comparisons,ledger=report['ledger'])),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);main(**vars(p.parse_args()))
