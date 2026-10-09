"""Complete the reference diagnostic with word identity, reuse all-valid proof."""
import json
import numpy as np
from research_realized_coverage_v1.core import *
from scripts.mode_geometry_experiment import POOL
from scripts.evaluate_paired_modes import references

def main():
    previous=read(RUN/'REFERENCE_DIAGNOSTIC.json');assert previous['counts']['mean_invalid']==0 and previous['reference_sha256']==sha(POOL)
    with np.load(POOL) as z:s={k:z[k] for k in z.files}
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='TRAIN'}
    changed=[];count=0
    for i,ident in enumerate(s['ids']):
        ref=references(labels[str(ident)])
        for w in sorted(set(s['modes'][i,s['mask'][i]])):
            indices=np.flatnonzero(s['mask'][i]&(s['modes'][i]==w));mean=s['paths'][i,indices].mean(0);actual=mode(mean,ref['config']);count+=1
            if actual!=w:changed.append(dict(id=str(ident),target=str(w),mean_word=actual))
    assert count==previous['counts']['groups']
    result=dict(groups=count,mean_changed_word=len(changed),cases=changed,previous_diagnostic_sha256=sha(RUN/'REFERENCE_DIAGNOSTIC.json'),
        additional_checker_queries=0,note='Same exact TRAIN means; reuse prior proof all are valid, now inspect their official word identity')
    write(RUN/'REFERENCE_WORD_DIAGNOSTIC.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
