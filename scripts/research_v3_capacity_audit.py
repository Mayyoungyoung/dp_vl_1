"""Read-only decomposition of finite-budget witness recall on sealed snapshots."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(rows, support):
    index={str(v):i for i,v in enumerate(support['ids'])}
    output=[]
    for r in rows:
        j=index[r['id']]
        known=set(support['modes'][j,support['mask'][j]])
        valid=np.asarray(r['valid'],dtype=bool);m=len(valid)
        found=[w for w,v in zip(r['words'],valid) if v and w is not None]
        distinct=set(found);covered=len(distinct&known)
        invalid=int((~valid).sum());unknown=int(valid.sum())-len(found)
        duplicates=len(found)-len(distinct);outside=len(distinct-known)
        slack=max(0,m-len(known));regret=min(m,len(known))-covered
        assert regret==invalid+unknown+duplicates+outside-slack
        unavoidable=max(0,len(known)-m)
        assert len(known)-covered==unavoidable+regret
        failures=Counter()
        for c,v in zip(r['candidates'],valid):
            if v:continue
            bad_goal=not c['semantic_goal_correct'];bad_clear=not c['tip_segments_clear']
            failures['both' if bad_goal and bad_clear else 'goal_only' if bad_goal else 'clearance_only' if bad_clear else 'other']+=1
        output.append(dict(id=r['id'],family=r['family'],variant=r['variant'],known=len(known),covered=covered,
            missing=len(known)-covered,unavoidable_capacity=unavoidable,budget_regret=regret,
            capacity_normalized_recall=covered/min(m,len(known)),invalid=invalid,
            unclassified_valid=unknown,duplicates=duplicates,valid_outside_witnesses=outside,unused_reference_capacity=slack,
            all_endpoints_wrong=int(not any(c['semantic_goal_correct'] for c in r['candidates'])),
            failures=dict(failures)))
    keys=[k for k,v in output[0].items() if isinstance(v,(int,float))]
    summary={k:float(np.mean([r[k] for r in output])) for k in keys}
    summary['requests_at_capacity_optimum']=sum(r['budget_regret']==0 for r in output)
    summary['failure_counts']=dict(sum((Counter(r['failures']) for r in output),Counter()))
    summary['slots_in_all_endpoint_failures']=sum(len(rows[i]['valid']) for i,r in enumerate(output) if r['all_endpoints_wrong'])
    return summary,output


def main(snapshot,output):
    root=Path(snapshot);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    support_path=root/'frequency_support_v1/support.npz'
    with np.load(support_path) as z:support={k:z[k] for k in ('ids','modes','mask')}
    paths={'support':support_path};summaries={};allrows={}
    for name in ('frequency_balanced','frequency_set_matching','frequency_set_sampled','safety_mean','safety_worst'):
        path=root/name/'evaluation_fixed_q_v2/rows.json';paths[name]=path
        rows=json.loads(path.read_text());assert len(rows)==288
        summaries[name],allrows[name]=audit(rows,support)
    result=dict(summary=summaries,input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        scope='Descriptive development decomposition at M8. Witnesses are incomplete: valid outside-witness modes are not geometric errors. Capacity normalization does not replace registered recall, change candidates, or establish novelty.')
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'rows.json').write_text(json.dumps(allrows,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.snapshot,a.output)
