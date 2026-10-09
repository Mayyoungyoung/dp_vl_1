"""Verify completed model/replay/RNG artifacts without opening new data splits."""
import argparse
import json
import torch
from scripts.run_observed_probability import ROOT,read,write,sha


def main(job_id):
    torch.set_num_threads(4)
    base=ROOT/'runs/verified_set_v1';run=base/'all_population_v2'
    names=[a+'_seed'+str(s) for a in ('ordinary','gate','budget_match','set_point') for s in range(3)]
    names+=['project_seed0','set_project_seed0','replay_set_point_seed0']
    required={'model','optimizer','scheduler','rng','loss_rng','sampler','counts','settings','step'}
    artifacts={}
    for name in names:
        folder=run/name;summary=read(folder/'summary.json');config=read(folder/'config.json')
        digest=sha(folder/'last.pt');assert digest==summary['last_sha256']
        state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert required.issubset(state) and state['step']==1200
        assert state['settings']==config and config['population']=='all_train'
        assert state['sampler']==summary['sampler'] and state['loss_rng']
        assert state['optimizer']['state'] and state['scheduler'] and state['rng']
        for key,value in summary['replay_sha256'].items():assert sha(folder/key)==value
        recovery=torch.load(folder/'recovery.pt',map_location='cpu',weights_only=False)
        assert required.issubset(recovery) and recovery['step']==1200
        assert recovery['sampler']==state['sampler']
        policy=config['policy'];assert policy['memory_fraction']==.35 and policy['cpu_threads']==4
        artifacts[name]=dict(checkpoint_sha256=digest,recovery_sha256=sha(folder/'recovery.pt'),
            summary_sha256=sha(folder/'summary.json'),config_sha256=sha(folder/'config.json'),
            rng_keys=sorted(state['rng']),sampler=state['sampler'],replay_sha256=summary['replay_sha256'],
            source_sha256=config['source_sha256'],peak_allocated_bytes=summary['peak_allocated_bytes'])
        del state,recovery
    replay=read(run/'replay_set_point_seed0/config.json')
    assert replay['replay_source_sha256']==read(run/'set_point_seed0/summary.json')['replay_sha256']
    jobs={f.parent.name:read(f) for f in (base/'jobs').glob('*/receipt.json')}
    nonterminal={n:r['status'] for n,r in jobs.items() if n!=job_id and r['status'] not in ('completed','failed')}
    assert not nonterminal
    assert int((base/'active.lock').read_text())==jobs[job_id]['pid']
    result=dict(models=artifacts,jobs_before_self=len(jobs)-1,
        completed_before_self=sum(r['status']=='completed' for n,r in jobs.items() if n!=job_id),
        failures={n:r for n,r in jobs.items() if r['status']=='failed'},
        budget_before_self_seconds=sum(r.get('elapsed_seconds',0) for n,r in jobs.items() if n!=job_id),
        note='Current audit wrapper is still running; final receipt closes its cost. Verify lock absent after exit.',
        locked_access=False)
    path=run/'artifact_audit.json';assert not path.exists();write(path,result)
    print(json.dumps({k:v for k,v in result.items() if k!='models'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);a=p.parse_args();main(a.job_id)
