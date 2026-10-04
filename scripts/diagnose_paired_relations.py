"""Post-seed0 band sensitivity on saved predictions; primary labels unchanged."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write,sha
from scripts.observed_layout_variation import crossing_signature,gap_certificates
from scripts.analyze_paired_modes import comparison


def main(root,arms,seeds,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    result={};family_values={};hashes={}
    for arm in arms:
        for seed in seeds:
            folder=root/'evaluation/paired_dev'/('%s_seed%d'%(arm,seed))
            if not (folder/'pool.npz').exists():continue
            key='%s/seed%d'%(arm,seed);refs=read(folder/'REFERENCE_GEOMETRY.json');rows=read(folder/'rows.json')
            with np.load(folder/'pool.npz') as a:paths=dict(zip(a['ids'],a['paths']))
            hashes[key]=dict(pool=sha(folder/'pool.npz'),rows=sha(folder/'rows.json'),references=sha(folder/'REFERENCE_GEOMETRY.json'))
            for name,offset in (('original',None),('expanded_to_clearance_top',.02)):
                groups={};known=0;valid=0
                for row in rows:
                    ident=row['id'];cfg=refs[ident]['config'];pred=set();positive=set()
                    for path,candidate in zip(paths[ident],row['candidates']):
                        signature=crossing_signature(path,cfg,offset)
                        if name=='original':assert signature is None if candidate['declared_passage_type'] is None else signature==tuple(candidate['declared_passage_type'])
                        if candidate['TipValid']:
                            valid+=1;known+=signature is not None
                            if signature is not None:pred.add(signature)
                    for path,ok in zip(refs[ident]['paths'],row['reference_valid']):
                        if ok:
                            signature=crossing_signature(path,cfg,offset)
                            if signature is not None:positive.add(signature)
                    prefix,target=ident.rsplit('_target',1);family,variant=prefix.rsplit('_',1)
                    groups.setdefault((family,target),{})[variant]=(pred,positive,cfg)
                families={}
                for (family,_),g in groups.items():
                    pa,wa,_=g['open'];pb,wb,cfg=g['closed'];ps,ws,_=g['shifted'];shared=wa&wb
                    changed=[c['row'] for c in gap_certificates(cfg) if c['closed_for_this_low_relation']][0]
                    opened={m for m in wa if m[changed]=='gap1'}
                    values=dict(shared_recall=len(shared&pa&pb)/len(shared),
                        shifted_shared_recall=len(wa&ws&pa&ps)/len(wa&ws),opened_recall=len(opened&pa)/len(opened))
                    families.setdefault(family,[]).append(values)
                aggregate={p:{k:float(np.mean([r[k] for r in items])) for k in items[0]} for p,items in families.items()}
                family_values[key+'/'+name]=aggregate
                result[key+'/'+name]=dict(metrics={k:float(np.mean([r[k] for r in aggregate.values()])) for k in next(iter(aggregate.values()))},
                    valid_candidates=valid,classified_valid_candidates=known)
    comparisons={}
    for seed in seeds:
        for name in ('original','expanded_to_clearance_top'):
            a,b=['%s/seed%d/%s'%(arm,seed,name) for arm in ('R1','R2')]
            if a in family_values and b in family_values:comparisons['R2-R1/seed%d/%s'%(seed,name)]=comparison(family_values[a],family_values[b])
    write(output/'RESULTS.json',dict(results=result,comparisons=comparisons,input_sha256=hashes,
        scope='Exploratory classification sensitivity after seed0. Remove the unclassified top +/-2cm band for laterally free gaps; retain the exact original candidate validity and over threshold. No training, relabeling of primary results or new predictions.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arms',nargs='+',default=['R0','R1','R2']);p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    a=p.parse_args();main(a.root,a.arms,a.seeds,a.output)
