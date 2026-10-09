"""Close only this research family; index actual commands, states and files."""
import argparse
import json
import platform
import numpy as np
from scripts.mode_geometry_experiment import RUN,ROOT,PREP,INITIAL,Q,POOL,read,write,sha,torch_setup


def main(output):
    torch=torch_setup()
    from routeset.observed_training_audit import tensor_state_digest
    jobs=ROOT/'runs/verified_set_v1/jobs';all_receipts=[read(p) for p in jobs.glob('*/receipt.json')]
    ours=[r for r in all_receipts if r['id'].startswith('mg_') and r['id']!='mg_close_audit']
    checkpoints={};streams={};files={}
    for path in sorted(RUN.glob('*/last.pt')):
        name=path.parent.name;c=torch.load(path,map_location='cpu',weights_only=False)
        recovery=torch.load(path.parent/'recovery.pt',map_location='cpu',weights_only=False)
        assert c['step']==recovery['step']==c['settings']['steps']
        digest=tensor_state_digest(c['model']);assert digest==tensor_state_digest(recovery['model'])
        frozen=tensor_state_digest({k:v for k,v in c['model'].items() if k.startswith(('geometry.','head.feature_encoder.','head.state_encoder.'))})
        assert frozen==c['frozen_sha256']
        assert set(c['rng'])=={'numpy_generator','numpy','python','torch','cuda'}
        checkpoints[name]=dict(step=c['step'],checkpoint_sha256=sha(path),model_tensor_sha256=digest,
            frozen_sha256=frozen,stream_sha256=c['stream'],initial_tensor_sha256=c['initial_tensor_sha256'],
            source_commit=c['settings']['source_commit'],parameter_count=sum(v.numel() for v in c['model'].values()),
            peak_allocated_bytes=read(path.parent/'summary.json')['peak_allocated_bytes'])
        if name.startswith('canonical_') and '_seed' in name:
            seed=name.rsplit('_seed',1)[1];streams.setdefault(seed,{})[name]=c['stream']
    assert all(len(set(v.values()))==1 for v in streams.values()),streams
    for f in RUN.rglob('*'):
        if f.is_file() and f.name!=output:files[str(f.relative_to(RUN))]=dict(sha256=sha(f),bytes=f.stat().st_size)
    source_exports={}
    for commit in sorted({r['source_commit'] for r in ours}):
        archive=ROOT/'research_v2/incoming'/('source_'+commit+'.tar')
        if archive.exists():source_exports[commit]=sha(archive)
    result=dict(checkpoints=checkpoints,paired_streams=streams,files=files,source_exports=source_exports,
        jobs=[{k:v for k,v in r.items() if k!='source_sha256'} for r in ours],
        source_hash_receipts={r['id']:sha(jobs/r['id']/'receipt.json') for r in ours},
        environment=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,cuda=torch.version.cuda,
                         device=torch.cuda.get_device_name(),threads=torch.get_num_threads(),memory_fraction=.35),
        initial_sha256=sha(INITIAL),scorer_sha256=sha(Q),support_sha256=sha(POOL),prepared_sha256=sha(PREP/'train.npz'),
        budget_excluding_this_audit=dict(allowance=7200,spent=sum(r.get('elapsed_seconds',0) for r in all_receipts),
            research_family_spent=sum(r['elapsed_seconds'] for r in ours)),
        locked_access=False,claim='Artifact validation, not scientific acceptance; final audit receipt charged after completion')
    write(RUN/output,result)
    print(json.dumps(dict(checkpoints=len(checkpoints),artifacts=len(files),budget=result['budget_excluding_this_audit'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='ARTIFACT_AUDIT.json');a=p.parse_args();main(a.output)
