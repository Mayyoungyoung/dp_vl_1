"""Frozen-model fresh-family diagnosis; no tuning or replacement of old gates."""
import argparse,json
import numpy as np
from research_feasible_space_v1.prepare import RUN
from scripts.run_observed_probability import read,write


def main(name,recall_root=None):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    kinds=['xyz','bounded','refitted_projection','refitted_center','refitted_boundcenter']
    models={};arrays={};hashes={}
    for k in kinds:
        results=[];family_arrays=[]
        for seed in range(3):
            folder=RUN/('generalization_%s_seed%d'%(k,seed))/'eval_adaptive'
            view=(RUN/recall_root/folder.parent.name) if recall_root else folder
            rows=read(view/'rows.json');m=read(view/'metrics.json')
            assert len(rows)==336 and len({r['family'] for r in rows})==16
            families=sorted({r['family'] for r in rows})
            family_arrays.append([[np.mean([r['raw']['distinct'] for r in rows if r['family']==f]),
                                   np.mean([r['raw']['valid_fraction'] for r in rows if r['family']==f]),
                                   np.mean([r['selected']['distinct'] for r in rows if r['family']==f]),
                                   np.mean([r['selected']['valid_fraction'] for r in rows if r['family']==f])] for f in families])
            variants={}
            for v in sorted({r['variant'] for r in rows}):
                group=[r for r in rows if r['variant']==v]
                variants[v]=dict(requests=len(group),U8=float(np.mean([r['raw']['distinct'] for r in group])),
                                V8=float(np.mean([r['raw']['valid_fraction'] for r in group])),
                                U4=float(np.mean([r['selected']['distinct'] for r in group])),V4=float(np.mean([r['selected']['valid_fraction'] for r in group])))
            results.append(dict(seed=seed,U8=m['raw']['distinct'],V8=m['raw']['valid_fraction'],U4=m['selected']['distinct'],
                                V4=m['selected']['valid_fraction'],known_recall=m['raw']['recall'],cell_feasibility=m['corridor_feasibility'],
                                containment=m['containment'],variants=variants))
            hashes[k+str(seed)]={t:m[t] for t in ('generator_sha256','head_sha256','scorer_sha256','pool_sha256')}
        models[k]=results;arrays[k]=np.array(family_arrays)
    rng=np.random.default_rng(641009);sd=rng.integers(3,size=(10000,3));fd=rng.integers(16,size=(10000,16));comparison={}
    for k in kinds:
        if k=='bounded':continue
        d=arrays['bounded']-arrays[k];boot=d[sd[:,:,None],fd[:,None,:]].mean((1,2))
        comparison['bounded minus '+k]=dict(metrics=['U8','V8','U4','V4'],mean=d.mean((0,1)).tolist(),crossed_CI95=np.quantile(boot,[.025,.975],axis=0).T.tolist())
    write(out/'RESULTS.json',dict(seed_results=models,means={k:{m:float(np.mean([r[m] for r in rows])) for m in ('U8','V8','U4','V4','known_recall','cell_feasibility','containment')} for k,rows in models.items()},
          comparisons=comparison,checkpoint_hashes=hashes,scope='16fresh rendered families,7variants,336requests; frozen3generator continuations; no selection use; sparse teacher recall',
          locked_access=False,method_selection_use=False,corrected_recall_root=recall_root))
    print(json.dumps(dict(comparisons=comparison,models={k:[r['U8'] for r in v] for k,v in models.items()})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--recall-root');main(**vars(p.parse_args()))
