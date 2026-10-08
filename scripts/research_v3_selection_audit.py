"""Read-only selection/capacity audit on sealed outputs, with no threshold fit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.analyze_paired_selection import select


def main(pool_file,rows_file,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rows=json.loads(Path(rows_file).read_text(encoding='utf-8'))
    with np.load(pool_file) as z:paths=z['paths'];ids=z['ids'];labels=z['labels']
    assert [r['id'] for r in rows]==list(ids)
    q=np.array([r['q'] for r in rows]);assert q.shape==labels.shape
    policies={};details={}
    for k in (1,2,4,8):
        for method in ('top_q','public_diverse'):
            values=[]
            for i,r in enumerate(rows):
                keep=np.argsort(-q[i],kind='stable')[:k] if method=='top_q' else select(paths[i],q[i],k)
                valid=labels[i,keep];modes={r['words'][j] for j in keep if labels[i,j]}
                possible={w for w,y in zip(r['words'],labels[i]) if y}
                values.append(dict(id=r['id'],family=r['family'],valid_fraction=float(valid.mean()),
                    any_valid=float(valid.any()),all_valid=float(valid.all()),distinct=len(modes),
                    candidate_oracle_distinct=min(k,len(possible)),candidate_oracle_valid=min(k,int(labels[i].sum()))/k,
                    candidate_oracle_all_valid=float(labels[i].sum()>=k)))
            name=f'{method}_K{k}';details[name]=values
            policies[name]={key:float(np.mean([r[key] for r in values])) for key in values[0] if key not in ('id','family')}
    thresholds={}
    for t in (.5,.8,.95):
        mask=q>=t;counts=mask.sum(1);thresholds[str(t)]=dict(returned_candidates=int(mask.sum()),
            mean_returned=float(counts.mean()),request_coverage=float((counts>0).mean()),
            validity=float(labels[mask].mean()) if mask.any() else None,
            distinct_per_requested=float(np.mean([len({r['words'][j] for j in range(8) if mask[i,j] and labels[i,j]}) for i,r in enumerate(rows)])))
    hashes={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (pool_file,rows_file)}
    result=dict(policies=policies,thresholds=thresholds,input_sha256=hashes,
        scope='Descriptive audit, all fixed outputs and thresholds. Oracle is candidate-pool upper bound, not deployable selector or exhaustive feasible mode oracle. No threshold selected and no risk guarantee.')
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (output/'rows.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    print(json.dumps(policies))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pool',required=True);p.add_argument('--rows',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    main(a.pool,a.rows,a.output)
