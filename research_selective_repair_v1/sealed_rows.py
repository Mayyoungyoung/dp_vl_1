"""Canonical independent metadata for already sealed constraint predictions."""
import argparse
import numpy as np
from research_selective_repair_v1.core import *
from scripts.research_v3_audit import mode

def convert(name,pool,data):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);pool_hash=sha(pool)
    with np.load(pool) as z:d={k:z[k] for k in ('ids','paths','events','q')}
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']=='DEV_MODEL'};rows=[]
    for i,ident in enumerate(d['ids']):
        ident=str(ident);ref=references(labels[ident]);a=assess(d['paths'][i:i+1],d['events'][i:i+1],d['q'][i:i+1],ref)
        rows.append(dict(id=ident,valid=a['valid'][0].tolist(),words=[mode(p,ref['config']) if v else None for p,v in zip(d['paths'][i],a['valid'][0])],selected_indices=a['selected'][0].tolist()))
    assert sha(pool)==pool_hash;write(out/'rows.json',rows);write(out/'SUMMARY.json',dict(pool_sha256=pool_hash,requests=len(rows),truth_access='Independent post-seal DEV metadata only',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--data',type=Path,required=True);convert(**vars(p.parse_args()))
