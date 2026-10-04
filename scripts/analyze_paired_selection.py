"""Evaluate the existing public q-first K selector on sealed M8 predictions."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write,sha


def select(paths,q,k):
    order=np.argsort(-q,kind='stable').tolist();chosen=[order[0]]
    for i in order[1:]:
        if len(chosen)==k:break
        if q[i]>=.5 and all(np.linalg.norm(paths[i]-paths[j],axis=-1).mean()>=.04 for j in chosen):chosen.append(i)
    for i in order:
        if len(chosen)==k:break
        if i not in chosen:chosen.append(i)
    return chosen


def main(root,arms,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    results={};hashes={};checks=[]
    for arm in arms:
        for seed in range(3):
            model='%s_seed%d'%(arm,seed);base=root/'reliability'/model
            deployment=base/('deployment_seed%d'%seed)
            with np.load(deployment/'example.npz') as a:np.testing.assert_array_equal(select(a['paths'],a['q'],4),a['selected_indices'])
            checks.append(model)
            for split in ('paired_dev','old_dev','dev32'):
                folder=root/'evaluation'/split/model;p=folder/'pool.npz';c=base/('calibration_seed%d'%seed)/(split+'.npz')
                hashes[str(p)]=sha(p);hashes[str(c)]=sha(c)
                with np.load(p) as a:ids=a['ids'];paths=a['paths'];labels=a['labels'];parents=a['parents']
                with np.load(c) as a:np.testing.assert_array_equal(ids,a['ids']);q=a['q']
                if split=='paired_dev':parents=np.array([p.rsplit('_',1)[0] for p in parents])
                rows={r['id']:r for r in read(folder/'rows.json')}
                for k in (1,2,4,8):
                    values=[]
                    for i,ident in enumerate(ids):
                        selected=select(paths[i],q[i],k);y=labels[i,selected]
                        modes={rows[str(ident)]['words'][j] for j in selected if labels[i,j]}
                        values.append(dict(valid_fraction=float(y.mean()),any_valid=float(y.any()),
                            all_valid=float(y.all()),valid_modes=len(modes),two_distinct=float(len(modes)>=2)))
                    families={p:{m:float(np.mean([v[m] for j,v in enumerate(values) if parents[j]==p])) for m in values[0]} for p in sorted(set(parents))}
                    results['%s/%s/K%d'%(split,model,k)]=dict(metrics={m:float(np.mean([v[m] for v in families.values()])) for m in values[0]},families=families)
    write(output/'RESULTS.json',dict(results=results,input_sha256=hashes,exact_public_example_indices=checks,
        scope='Fixed deployed selector: highest q first; prefer q>=.5 and mean path distance>=.04m, then fill by q. Always generates eight paths for every returned K. No oracle selection, repair or new generation. K>1 set validity is measured, not inferred from independent q.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--arms',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.root,a.arms,a.output)
