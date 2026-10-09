"""TRAIN-only reference conflict diagnosis before choosing a geometry repair."""
import json
import numpy as np
from research_realized_coverage_v1.core import *
from scripts.mode_geometry_experiment import POOL
from scripts.evaluate_paired_modes import references

def main():
    with np.load(POOL) as z:s={k:z[k] for k in z.files}
    assert set(s['splits'])=={'TRAIN'}
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='TRAIN'}
    counts=dict(groups=0,mean_invalid=0,base_valid=0,base_valid_mean_target_invalid=0);variance=[];loss_gap=[]
    for i,ident in enumerate(s['ids']):
        ref=references(labels[str(ident)]);check=checker(ref)
        with np.load(RUN/'feedback_C_train'/(str(ident)+'.npz')) as z:
            p=z['paths'][0];v=z['valid'][0];m=z['modes'][0];raw=z['raw_words'][0]
        for w in sorted(set(s['modes'][i,s['mask'][i]])):
            indices=np.flatnonzero(s['mask'][i]&(s['modes'][i]==w));x=s['paths'][i,indices];e=s['events'][i,indices]
            mean=x.mean(0);valid,_=check(mean[None],e[:1]);counts['groups']+=1;counts['mean_invalid']+=int(not valid[0])
            variance.append(float(np.var(x,axis=0).mean()))
            for slot in np.flatnonzero(m==VOCAB.index(w)):
                distances=np.square(p[slot]-x).mean((1,2));loss_gap.append(float(distances.mean()-distances.min()))
                counts['base_valid']+=int(v[slot]);counts['base_valid_mean_target_invalid']+=int(v[slot] and not valid[0])
    result=dict(counts=counts,within_mode_coordinate_variance=float(np.mean(variance)),random_minus_nearest_target_mse=float(np.mean(loss_gap)),
        reference_sha256=sha(POOL),scope='TRAIN target inconsistency diagnostic, no new positives or deployment changes',checker_queries=counts['groups'])
    write(RUN/'REFERENCE_DIAGNOSTIC.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
