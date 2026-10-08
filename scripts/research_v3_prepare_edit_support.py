"""Frozen verified positives, original mode set and semantic grounding preserved."""
from collections import Counter,defaultdict
import hashlib
import numpy as np
from scripts.research_v3_frequency import RUN,SUPPORT,pad_targets
from scripts.run_observed_probability import read,write,sha,lines
from scripts.paired_modes_data import DATA
from scripts.evaluate_paired_modes import references,check_candidates


def main():
    out=RUN/'verified_edit_support_v1';out.mkdir(exist_ok=False)
    audit=RUN/'train_edit_retention_v1';r=read(audit/'RESULTS.json')
    assert sha(audit/'rows.json')==r['rows_sha256']
    pred=RUN/'posttrain_collision_v1/predictions.npz';assert sha(pred)==r['prediction_sha256']
    assert sha(SUPPORT/'support.npz')==r['support_sha256']
    with np.load(pred) as z:predictions={k:z[k] for k in z.files}
    pi={str(k):j for j,k in enumerate(predictions['ids'])}
    with np.load(SUPPORT/'support.npz') as z:base={k:z[k] for k in z.files}
    ids=np.flatnonzero(base['splits']=='TRAIN');assert len(ids)==1152
    labels={x['id']:x for x in lines(DATA/'export/supervision.jsonl') if x['split']=='TRAIN'}
    incoming=defaultdict(list)
    for row in read(audit/'rows.json'):incoming[row['destination_id']].append(row)
    xs=[];es=[];tags=[];records=[];counts=Counter()
    digest=lambda p,e:hashlib.sha256(p.tobytes()+e.tobytes()).hexdigest()
    for i in ids:
        ident=str(base['ids'][i]);mask=base['mask'][i];p=list(base['paths'][i,mask]);e=list(base['events'][i,mask]);m=list(base['modes'][i,mask]);known=set(m)
        original_count=len(p);seen={digest(a,b) for a,b in zip(p,e)};ref=references(labels[ident])
        for edge in incoming[ident]:
            si=pi[edge['source_id']];checks=check_candidates(predictions['paths'][si],predictions['events'][si],ref['label'],ref['current'],ref['truth'],ref['config'])[1]
            assert [c['TipValid'] for c in checks]==edge['cross_valid']
            for j,w in enumerate(edge['cross_words']):
                if not (edge['source_valid'][j] and edge['cross_valid'][j]):continue
                x=predictions['paths'][si,j];event=predictions['events'][si,j];h=digest(x,event)
                status='unreferenced_positive' if w not in known else 'duplicate' if h in seen else 'added'
                counts[status]+=1;records.append(dict(source_id=edge['source_id'],destination_id=ident,candidate=j,mode=w,path_event_sha256=h,status=status))
                if status=='added':p.append(x);e.append(event);m.append(w);seen.add(h)
        assert set(m)==known
        np.testing.assert_array_equal(np.array(p[:original_count]),base['paths'][i,mask])
        np.testing.assert_array_equal(np.array(e[:original_count]),base['events'][i,mask])
        xs.append(np.array(p));es.append(np.array(e));tags.append(m);counts['original_references']+=original_count
    paths,events,valid=pad_targets(xs,es);maximum=paths.shape[1]
    np.savez_compressed(out/'support.npz',paths=paths,events=events,mask=valid,modes=np.array([x+['']*(maximum-len(x)) for x in tags]),ids=base['ids'][ids],splits=base['splits'][ids])
    write(out/'records.json',records)
    write(out/'receipt.json',dict(support_sha256=sha(out/'support.npz'),base_support_sha256=r['support_sha256'],source_prediction_sha256=sha(pred),
        audit_rows_sha256=sha(audit/'rows.json'),records_sha256=sha(out/'records.json'),counts=dict(counts),requests=1152,
        mode_sets_unchanged=True,original_references_exact=True,scope='All registered TRAIN cross-valid same-known-mode positives; no distance/failure selection. Unreferenced positives excluded, never negatives. Original grounding targets retained separately.'))
    print(dict(counts=counts,maximum_targets=maximum),flush=True)


if __name__=='__main__':main()
