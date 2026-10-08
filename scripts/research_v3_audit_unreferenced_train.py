"""Local TRAIN reference-vocabulary completeness/capacity audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np


def main(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    receipt=json.loads((root/'receipt.json').read_text());records=json.loads((root/'records.json').read_text())
    assert sha(root/'records.json')==receipt['records_sha256']
    assert sha(root/'support.npz')==receipt['support_sha256']
    with np.load(root/'support.npz') as z:
        ids=list(map(str,z['ids']));assert len(ids)==1152 and set(z['splits'])=={'TRAIN'}
        known={i:set(z['modes'][j,z['mask'][j]]) for j,i in enumerate(ids)}
    idset=set(ids)
    for r in records:assert r['source_id'] in idset and r['destination_id'] in idset
    rr=[r for r in records if r['status']=='unreferenced_positive']
    assert len(rr)==receipt['counts']['unreferenced_positive']==853
    for r in rr:assert r['mode'] not in known[r['destination_id']] and r['mode'] is not None
    pairs={(r['destination_id'],r['mode']) for r in rr}
    affected={i for i,m in pairs};expanded={i:set(v) for i,v in known.items()}
    for i,m in pairs:expanded[i].add(m)
    family=lambda i:i.rsplit('_',2)[0]
    before=np.array([len(known[i]) for i in ids]);after=np.array([len(expanded[i]) for i in ids])
    def capacity(a):return dict(quantiles=np.quantile(a,[0,.25,.5,.75,1]).tolist(),above8=int((a>8).sum()),unavoidable_missing_total=int(np.maximum(a-8,0).sum()),unavoidable_missing_per_request=float(np.maximum(a-8,0).mean()))
    result=dict(records=len(rr),unique_destination_path_pairs=len({(r['destination_id'],r['path_event_sha256']) for r in rr}),
                unique_destination_mode_pairs=len(pairs),affected_requests=len(affected),affected_families=len({family(i) for i in affected}),
                record_mode_counts=dict(sorted(Counter(r['mode'] for r in rr).items())),
                unique_class_mode_counts=dict(sorted(Counter(m for i,m in pairs).items())),
                unique_class_family_counts=dict(sorted(Counter(family(i) for i,m in pairs).items())),
                before=capacity(before),after=capacity(after),scope='TRAIN-only sealed verified-positive vocabulary audit; no model updates, no exhaustive-mode or independent-demonstration claim.')
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'records.json').write_text(json.dumps(rr,indent=2)+'\n')
    (out/'request_counts.json').write_text(json.dumps([dict(id=i,before=len(known[i]),after=len(expanded[i]),added_modes=sorted(expanded[i]-known[i])) for i in ids],indent=2)+'\n')
    (out/'PROVENANCE.json').write_text(json.dumps(dict(command=[sys.executable]+sys.argv,script_sha256=sha(Path(__file__)),inputs={str((root/n).resolve()):sha(root/n) for n in ('support.npz','records.json','receipt.json')},outputs={n:sha(out/n) for n in ('RESULTS.json','records.json','request_counts.json')}),indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='unique_class_family_counts'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);a=p.parse_args();main(a.root,a.output)
