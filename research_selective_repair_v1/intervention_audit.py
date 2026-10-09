"""Post-seal route transport audit; old scenes/routes never enter prediction."""
import argparse
from pathlib import Path
import numpy as np
from research_selective_repair_v1.core import *
from scripts.research_v3_audit import mode

def audit(name,pool,rows,data):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    with np.load(pool) as z:p={k:z[k] for k in ('ids','paths','events')}
    index={str(x):i for i,x in enumerate(p['ids'])};sealed={r['id']:r for r in read(rows)}
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    family={r['parent_id']:r for r in read(data/'registration.json')['parent_plan']};records=[]
    for ident in index:
        label=labels[ident];plan=family[label['parent_id']]
        if plan['variant']=='open':continue
        target=ident.rsplit('_target',1)[1];oldid=plan['family_id']+'_open_target'+target
        if oldid not in index:raise ValueError('Missing registered open counterpart')
        old=p['paths'][index[oldid]];events=p['events'][index[oldid]];oldvalid=np.asarray(sealed[oldid]['valid'],bool)
        ref=references(label);_,check=check_candidates(old,events,ref['label'],ref['current'],ref['truth'],ref['config'])
        transported=np.asarray([x['TipValid'] for x in check]);words=[mode(x,ref['config']) for x in old]
        still={w for w,v,ov in zip(words,transported,oldvalid) if v and ov}-{None}
        new=wordset(p['paths'][index[ident]],np.asarray(sealed[ident]['valid'],bool),ref['config'])
        selected=set(sealed[ident]['selected_indices']);returned=wordset(p['paths'][index[ident]],np.asarray(sealed[ident]['valid'],bool)&np.asarray([j in selected for j in range(8)]),ref['config'])
        oldwords=sealed[oldid]['words']
        records.append(dict(id=ident,family=plan['family_id'],variant=plan['variant'],old_valid_routes=int(oldvalid.sum()),transported_valid_old_routes=int((oldvalid&transported).sum()),invalidated_old_routes=int((oldvalid&~transported).sum()),surviving_current_words=sorted(still),retained_current_words=sorted(still&new),missed_current_words=sorted(still-new),returned_survivors=sorted(still&returned),new_words_outside_survivors=sorted(new-still),transported_relabelled_valid_routes=sum(bool(v and ov and w!=ow) for w,ow,v,ov in zip(words,oldwords,transported,oldvalid)),scope='Feasible transported current operational words; does not assert every historical word remains geometrically feasible'))
    write(out/'ROWS.json',records)
    denominator=sum(len(r['surviving_current_words']) for r in records)
    report=dict(pairs=len(records),families=len({r['family'] for r in records}),surviving_words=denominator,retained_words=sum(len(r['retained_current_words']) for r in records),returned_survivors=sum(len(r['returned_survivors']) for r in records),survival_recall=sum(len(r['retained_current_words']) for r in records)/denominator if denominator else None,invalidated_old_routes=sum(r['invalidated_old_routes'] for r in records),relabelled=sum(r['transported_relabelled_valid_routes'] for r in records),pool_sha256=sha(pool),rows_sha256=sha(rows),data=str(data),truth_access='After sealed generation, DEV only',locked_access=False)
    write(out/'SUMMARY.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--pool',type=Path,required=True);p.add_argument('--rows',type=Path,required=True);p.add_argument('--data',type=Path,required=True);audit(**vars(p.parse_args()))
