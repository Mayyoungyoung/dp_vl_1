"""Report every frozen arm, with scene-family and continuation uncertainty."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

METRICS = ['U8','V8','U4','V4','duplicates','distinct_queries','condition_hit']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def array(rows, families):
    return np.array([[np.mean([float(r[g][k]) if g else float(r[k]) for r in rows if r['family']==f])
        for g,k in [('raw','distinct'),('raw','valid_fraction'),('selected','distinct'),('selected','valid_fraction'),
                    (None,'duplicate_valid_routes'),(None,'distinct_queries'),(None,'condition_hit')]] for f in families])


def main(root, output):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    specs={arm:[[f'shift_{arm}_seed{s}'] for s in range(3)] for arm in ('B0','Bset','C')}
    specs['C_ordinary']=[[f'shift_C_seed{s}_ordinary_{r}' for r in (71239,71240,71241)] for s in range(3)]
    specs['C0_success']=[[f'shift_C0_success_seed{s}'] for s in range(5)]
    arrays={};summary={};hashes={};all_ids=None;families=None
    for arm,reps in specs.items():
        arr=[];trial_results=[];variant_arrays={}
        for names in reps:
            group=[]
            for name in names:
                path=root/name/'rows.json';rows=read(path);ids=[r['id'] for r in rows]
                if all_ids is None:all_ids=ids;families=sorted({r['family'] for r in rows})
                assert ids==all_ids and len(rows)==336 and len(families)==16
                group.append(array(rows,families));m=read(root/name/'RESULTS.json')
                trial_results.append(dict(name=name,metrics=group[-1].mean(0).tolist(),generator_sha256=m['generator_sha256'],head_sha256=m['head_sha256']))
                hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
                for variant in sorted({r['variant'] for r in rows}):
                    vr=[r for r in rows if r['variant']==variant]
                    variant_arrays.setdefault(variant,[]).append(array(vr,families).mean(0))
            arr.append(np.mean(group,axis=0))
        arrays[arm]=np.array(arr)
        summary[arm]=dict(mean=dict(zip(METRICS,arrays[arm].mean((0,1)).tolist())),trials=trial_results,
                         variants={v:dict(zip(METRICS,np.mean(vals,axis=0).tolist())) for v,vals in variant_arrays.items()})
    rng=np.random.default_rng(20261010);comparisons={}
    for other in ('B0','Bset','C_ordinary'):
        delta=arrays['C']-arrays[other]
        sd=rng.integers(3,size=(10000,3));fd=rng.integers(16,size=(10000,16))
        boot=delta[sd[:,:,None],fd[:,None,:]].mean((1,2))
        comparisons['C minus '+other]=dict(mean=dict(zip(METRICS,delta.mean((0,1)).tolist())),
            crossed_CI95=dict(zip(METRICS,np.quantile(boot,[.025,.975],axis=0).T.tolist())))
    delta=arrays['C0_success']-arrays['C'][0:1]
    sd=rng.integers(5,size=(10000,5));fd=rng.integers(16,size=(10000,16))
    boot=delta[sd[:,:,None],fd[:,None,:]].mean((1,2))
    comparisons['C0_success minus C0']=dict(mean=dict(zip(METRICS,delta.mean((0,1)).tolist())),
        crossed_CI95=dict(zip(METRICS,np.quantile(boot,[.025,.975],axis=0).T.tolist())),
        scope='Head-seed/family uncertainty conditional on fixed generator C0')
    report=dict(metrics=METRICS,models=summary,comparisons=comparisons,row_sha256=hashes,
        families=families,requests=336,locked_access=False,
        scope='Reused development families distinct from original TRAIN. Three shared-parent continuation seeds; sampling repetitions averaged within continuation. No final-test inference.',
        seed_selection=False,pooled_candidates=False,robot_execution=False)
    (output/'RESULTS.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    table=['|Method|Valid modes@8|Validity@8|Valid modes@4|Validity@4|Duplicate valid routes|',
           '|---|---:|---:|---:|---:|---:|']
    for arm,data in summary.items():
        v=data['mean'];table.append(f"|{arm}|{v['U8']:.4f}|{100*v['V8']:.2f}%|{v['U4']:.4f}|{100*v['V4']:.2f}%|{v['duplicates']:.4f}|")
    table += ['','C_ordinary averages three independent inference trials within each generator continuation; C0_success averages five heads on fixed C0.']
    (output/'TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print(json.dumps({k:v['mean'] for k,v in summary.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    main(**vars(p.parse_args()))
