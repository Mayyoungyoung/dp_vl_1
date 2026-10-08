"""Descriptive q values on all shared endpoint failures; not a detector."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(snapshot,closure,output):
    snapshot,closure,output=map(Path,(snapshot,closure,output));output.mkdir(parents=True,exist_ok=False)
    original=snapshot/'safety_mean/evaluation_fixed_q_v2/rows.json';base=json.loads(original.read_text())
    files={'original_q':original}
    for seed in range(3):files['paired_seed'+str(seed)]=closure/f'matched_q_paired_v1/reliability/mean_seed0/calibration_seed{seed}/rows.json'
    summaries={};details={}
    for name,path in files.items():
        rows={r['id']:r for r in json.loads(path.read_text())};groups={False:[],True:[]}
        for b in base:
            r=rows[b['id']];assert r['valid']==b['valid']
            failed=all(not c['semantic_goal_correct'] for c in b['candidates']);q=np.asarray(r['q'])
            groups[failed].append(dict(id=b['id'],family=b['family'],mean_q=float(q.mean()),max_q=float(q.max()),
                top_q_valid=bool(r['valid'][q.argmax()]),q=q.tolist()))
        summaries[name]={}
        for failed,rr in groups.items():
            key='all_endpoint_wrong' if failed else 'any_endpoint_correct'
            summaries[name][key]=dict(requests=len(rr),families=len({r['family'] for r in rr}),
                mean_q=float(np.mean([r['mean_q'] for r in rr])),mean_max_q=float(np.mean([r['max_q'] for r in rr])),
                max_q_ge08_requests=sum(r['max_q']>=.8 for r in rr),top_q_valid_requests=sum(r['top_q_valid'] for r in rr))
        details[name]=groups[True]
    result=dict(summary=summaries,all_shared_endpoint_failures=details,
        input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files.values()},
        scope='All12 label-defined shared-endpoint failures, fixed generator and all three q seeds. Threshold0.8 reused from prior descriptive reporting, not fitted. Conditioning on observed error does not test calibration within an independently identifiable input group and cannot be used as an inference detector or safety guarantee.')
    (output/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summaries,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('snapshot','closure','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();main(a.snapshot,a.closure,a.output)
