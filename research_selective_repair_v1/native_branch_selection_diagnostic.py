"""TRAIN measured all8 alternative replay, NEVER actual returned4 evidence."""
import argparse
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_allocation import COMBINATIONS,coverage

def run(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);dataset=RUN/'body_native_branch_data_v1/samples.npz'
    with np.load(dataset) as z:d={k:z[k] for k in ('ids','slots','options','paths','labels','families')}
    manifest=read(dataset.parent/'MANIFEST.json');assert manifest['native_joint_labels'] and sha(dataset)==manifest['samples_sha256']
    lookup={(str(i),int(s),int(o)):j for j,(i,s,o) in enumerate(zip(d['ids'],d['slots'],d['options']))}
    held=sorted(set(d['families']))[-2:]
    ids=sorted({str(i) for i,f,o in zip(d['ids'],d['families'],d['options']) if f in held and o==4})
    assert len(ids)==4
    recipes=[0,4,5];records=[];oracle=[]
    for label in ('state','aux','route','planned','binary','confusion','identity','null','loop'):
        folder=RUN/('body_native_%s_TRAIN_screen_seed0_v1'%label)
        with np.load(folder/'sealed_predictions.npz') as z:s={k:z[k] for k in ('ids','paths','choices','probabilities')}
        with np.load(folder/'pool.npz') as z:selected={str(i):v for i,v in zip(z['ids'],z['selected'])}
        index={str(v):j for j,v in enumerate(s['ids'])};rows=[]
        for ident in ids:
            ix=index[ident];actual=[];base=[];changed=[]
            for slot,option in enumerate(s['choices'][ix]):
                old=lookup[(ident,slot,0)];different=not np.array_equal(s['paths'][ix,slot],d['paths'][old])
                row=lookup[(ident,slot,recipes[option] if different else 0)]
                assert np.array_equal(s['paths'][ix,slot],d['paths'][row]),'Measured replay requires exact curve equality'
                actual.append(d['labels'][row]);base.append(d['labels'][old]);changed.append(different)
            actual=np.asarray(actual);base=np.asarray(base)
            p=s['probabilities'][ix,np.arange(8),s['choices'][ix]]
            rows.append(dict(id=ident,measured_alternative_replay_E8=len(set(actual)-{0}),identity_measured_E8=len(set(base)-{0}),
                successful_clear=int((actual>0).sum()),changed_slots=int(sum(changed)),
                old_positive_to_chosen_negative=int(((base>0)&(actual==0)).sum()),
                model_expected_E8=float(coverage(p).sum()),model_expected_E4_fixed_q=float(coverage(p[selected[ident]]).sum()),
                labels=actual.tolist(),identity_labels=base.tolist()))
        records.append(dict(method=label,mean_measured_alternative_replay_E8=float(np.mean([r['measured_alternative_replay_E8'] for r in rows])),
            successful_clear=sum(r['successful_clear'] for r in rows),changed_slots=sum(r['changed_slots'] for r in rows),rows=rows,prediction_sha256=sha(folder/'sealed_predictions.npz')))
    for ident in ids:
        labels=np.asarray([[d['labels'][lookup[(ident,slot,o)]] for o in recipes] for slot in range(8)])
        # Identity fallback labels remain the measured identity trial, rather
        # than manufacturing gains from different native repeats of one curve.
        for slot in range(8):
            original=d['paths'][lookup[(ident,slot,0)]]
            for j,o in enumerate(recipes):
                if np.array_equal(original,d['paths'][lookup[(ident,slot,o)]]):labels[slot,j]=labels[slot,0]
        target=labels[np.arange(8)[None],COMBINATIONS]
        utility=np.asarray([len(set(v)-{0}) for v in target]);original=len(set(labels[:,0])-{0})
        oracle.append(dict(id=ident,identity_measured_E8=original,best_among_measured_alternatives_E8=int(utility.max()),
            found_gain=int(utility.max()-original),candidate_opportunities=int(((labels[:,0]==0)&(labels[:,1:].max(-1)>0)).sum())))
    write(out/'SUMMARY.json',dict(records=records,TRAIN_oracle_measured_opportunity=oracle,scope='Four TRAIN held requests only. Mixing already observed slot-ranked teacher alternatives is a diagnostic. Not fresh, not native replay, not q-selected rank0..3 E4, not causal damage. Exact identity curves reuse their original observed trial; native randomness not controlled.',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
