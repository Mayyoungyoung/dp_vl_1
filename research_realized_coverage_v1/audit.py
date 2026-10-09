"""Bind actual new artifacts/jobs without touching reserved data or old outputs."""
import argparse
import json
import os
import platform
from pathlib import Path
import numpy as np
import torch
from research_realized_coverage_v1.core import RUN,ROOT,BASE,Q,read,write,sha,torch_setup

def audit(output):
    torch_setup();jobs=[read(p) for p in sorted((RUN/'jobs').glob('*/receipt.json'))]
    completed=[r for r in jobs if r['status'] in ('completed','failed')]
    source={}
    for commit in sorted({r['source_commit'] for r in jobs}):
        p=ROOT/'research_v2/incoming'/('source_'+commit+'.tar');source[commit]=sha(p)
    ckpts={}
    for p in sorted(RUN.glob('*/last.pt')):
        c=torch.load(p,map_location='cpu',weights_only=False)
        assert 'rng' in c and 'optimizer' in c and c['step']==c['settings']['steps']
        ckpts[str(p.relative_to(RUN))]=dict(sha256=sha(p),step=c['step'],settings=c['settings'],stream=c['stream'])
    feedback={};cost=0
    for p in RUN.glob('*/SUMMARY.json'):
        d=read(p)
        if 'manifest' in d and 'files' in d:
            for row in d['files']:assert sha(p.parent/(row['id']+'.npz'))==row['file_sha256']
            feedback[p.parent.name]=dict(manifest=d['manifest'],counts=d['counts'],sha256=sha(p))
            cost+=d['counts']['checked_routes']
    # Original sources3bc/723 used an extra discarded base decode, only these two pools.
    discarded=8*(feedback.get('feedback_C_train',{}).get('counts',{}).get('requests',0)+feedback.get('diagnostic_C_dev',{}).get('counts',{}).get('requests',0))
    evaluated={}
    for p in RUN.glob('*/eval_adaptive/metrics.json'):
        d=read(p);assert sha(p.parent/'pool.npz')==d['pool_sha256']
        evaluated[p.parent.parent.name]={k:d[k] for k in ('raw','selected','condition_hit','cost')}
    files={str(p.relative_to(RUN)):dict(sha256=sha(p),bytes=p.stat().st_size) for p in RUN.rglob('*')
           if p.is_file() and p.name!=output and p.suffix not in ('.tar','.log','.lock') and 'jobs' not in p.parts}
    result=dict(checkpoints=ckpts,feedback=feedback,evaluated=evaluated,files=files,source_exports=source,
        jobs=[{k:v for k,v in r.items() if k!='source_sha256'} for r in completed],
        job_receipt_sha256={r['id']:sha(RUN/'jobs'/r['id']/'receipt.json') for r in completed},
        spent_excluding_this_audit=sum(r['elapsed_seconds'] for r in completed),time_limit=None,
        feedback_verified_routes=cost,additional_discarded_feedback_routes=discarded,
        reference_diagnostic_checker_queries=read(RUN/'REFERENCE_DIAGNOSTIC.json')['checker_queries'],
        deployment_routes=sum(d['cost']['generated_routes'] for d in evaluated.values()),
        deployment_replay_benchmark_routes=sum(8*(1+read(p)['additional_benchmark_decodes']) for p in RUN.glob('*_DEPLOYMENT.json')),
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),memory_fraction=.35),
        initial_sha256=sha(BASE),scorer_sha256=sha(Q),locked_access=False,default_changed=False)
    write(RUN/output,result);print(json.dumps({k:len(result[k]) for k in ('checkpoints','feedback','evaluated','files','jobs')}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='ARTIFACT_AUDIT.json');a=p.parse_args();audit(a.output)
