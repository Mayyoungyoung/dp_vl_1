"""Optimistic structural ceiling for the actual fixed 18-node draft prefixes."""
import argparse
from pathlib import Path
import numpy as np
from research_selective_repair_v1.core import *
from scripts.research_v3_audit import mode

def maximum_matching(edges):
    assigned={}
    def visit(slot,seen):
        for word in edges[slot]:
            if word in seen:continue
            seen.add(word)
            if word not in assigned or visit(assigned[word],seen):assigned[word]=slot;return True
        return False
    return sum(visit(slot,set()) for slot in range(len(edges)))

def bound(name,data_name,data):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    with np.load(RUN/data_name/'samples.npz') as z:
        ix=z['splits']=='DEV_MODEL';d={k:z[k][ix] for k in ('ids','drafts','events','families')}
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']=='DEV_MODEL'};records=[]
    for i,ident in enumerate(d['ids']):
        ident=str(ident);p=d['drafts'][i];prefix=p[:,:18];padded=np.concatenate([prefix,np.repeat(prefix[:,-1:],6,axis=1)],axis=1)
        ref=references(labels[ident]);_,cc=check_candidates(padded,d['events'][i],ref['label'],ref['current'],ref['truth'],ref['config'])
        edges=[];bad=0;fixed=0
        for path,c in zip(prefix,cc):
            if not(c['post_segments_clear'] and c['workspace_floor_correct']):edges.append([]);bad+=1;continue
            w=mode(path,ref['config'])
            if w is None:edges.append(list(VOCAB)) # generous unknown/incomplete future
            else:edges.append([w]);fixed+=1
        records.append(dict(id=ident,family=str(d['families'][i]),optimistic_U8=maximum_matching(edges),irreversible_prefix_failures=bad,fixed_word_slots=fixed))
    report=dict(requests=len(records),optimistic_mean_U8=float(np.mean([r['optimistic_U8'] for r in records])),irreversible_prefix_failures=sum(r['irreversible_prefix_failures'] for r in records),fixed_word_slots=sum(r['fixed_word_slots'] for r in records),draft_samples_sha256=sha(RUN/data_name/'samples.npz'),scope='Current screen queries and immutable nodes0-17 only. Unrestricted future tail and any word for unresolved prefixes give an optimistic bound; not a global planner/generator bound or a solver infeasibility claim.',locked_access=False)
    write(out/'ROWS.json',records);write(out/'SUMMARY.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--data-name',required=True);p.add_argument('--data',type=Path,required=True);bound(**vars(p.parse_args()))
