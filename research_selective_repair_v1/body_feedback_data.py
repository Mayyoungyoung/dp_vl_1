"""Build actual TRAIN outcome labels; prediction features remain observation-only."""
import argparse
from pathlib import Path
import numpy as np
from research_selective_repair_v1.io import RUN,ROOT,read,write,sha
from research_selective_repair_v1.body_options import options

WORDS=['%s|%s'%(a,b) for a in ('gap0','gap1','gap2','over') for b in ('gap0','gap1','gap2','over')]
DEFAULT=[('body_execution_train_lift0_pilot_v1',0,'semantics_v2'),('body_execution_train_lift.08_pilot_v1',1,'semantics_v2'),('body_execution_train_preservedlift_pilot_v1',2,'semantics_v2')]+[('body_execution_train_%s_families2_7_v1'%n,j,'semantics_v1') for j,n in enumerate(('identity','lift','preserved'))]

def build(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    pool=RUN/'constraints_global_interventions_TRAIN_body_v1'
    assert read(pool/'SEAL.json')['role']=='TRAIN'
    with np.load(pool/'sealed_predictions.npz') as z:p={k:z[k] for k in ('ids','paths','completed')}
    byid={str(v):i for i,v in enumerate(p['ids'])}
    with np.load(RUN/'interventions_calibrated_targets_v1/samples.npz') as z:
        legal=z['splits']=='TRAIN';context={str(i):c for i,c in zip(z['ids'][legal],z['context'][legal])};family={str(i):str(f) for i,f in zip(z['ids'][legal],z['families'][legal])}
    rows=[];hashes={};seen=set()
    for source,option,version in DEFAULT:
        folder=RUN/source;manifest=read(folder/'MANIFEST.json');audit=RUN/(source+'_'+version)
        assert manifest['TRAIN_feedback_only'] and not manifest['locked_access']
        assert manifest['pool_sha256']==sha(pool/'pool.npz');assert (folder/'SUMMARY.json').exists()
        hashes[str(folder/'MANIFEST.json')]=sha(folder/'MANIFEST.json');hashes[str(audit/'ROWS.json')]=sha(audit/'ROWS.json')
        plans={read(f)['execution_id']:read(f) for f in folder.glob('plan*.json')}
        for label in read(audit/'ROWS.json'):
            ident=label['id'];slot=label['slot'];key=(ident,slot,option);assert key not in seen;seen.add(key)
            assert ident in context,'Only explicitly cached TRAIN observation inputs allowed'
            ix=byid[ident];path=np.asarray(plans[ident]['execution_paths'],np.float32)[slot]
            expected=options(p['paths'][ix],p['completed'][ix])[slot,option]
            np.testing.assert_allclose(path,expected,atol=1e-7,rtol=0)
            w=label['successful_executable_word'];target=0 if w is None else WORDS.index(w)+1
            rows.append(dict(id=ident,slot=slot,option=option,family=family[ident],path=path,completed=p['completed'][ix],context=context[ident],label=target,trace_sha256=label['trace_sha256']))
    assert len(rows)==384 and len({r['family'] for r in rows})==8
    np.savez_compressed(out/'samples.npz',ids=np.array([r['id'] for r in rows]),families=np.array([r['family'] for r in rows]),roles=np.full(len(rows),'TRAIN'),slots=np.array([r['slot'] for r in rows]),options=np.array([r['option'] for r in rows]),paths=np.array([r['path'] for r in rows]),completed=np.array([r['completed'] for r in rows]),context=np.array([r['context'] for r in rows]),labels=np.array([r['label'] for r in rows]))
    write(out/'MANIFEST.json',dict(source_hashes=hashes,pool_sha256=sha(pool/'pool.npz'),samples_sha256=sha(out/'samples.npz'),rows=len(rows),TRAIN_families=sorted({r['family'] for r in rows}),targets=[0],labels='Actual unchanged fixed-controller outcome and actual sampled-tip operational word',no_DEV_feedback=True,locked_access=False,trace_sha256=[r['trace_sha256'] for r in rows]))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);build(**vars(p.parse_args()))
