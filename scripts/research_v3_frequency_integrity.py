"""Read-only audit of actual checkpoints, target exposure and initial baseline."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.research_v3_analyze_frequency import ARMS, paired, row_metrics
from scripts.research_v3_frequency import probabilities
from scripts.run_observed_probability import read, write, sha


def replay_exposure(support, cfg, arm):
    """Reproduce label RNG, including matching permutations, independently."""
    train=np.flatnonzero(support['splits']=='TRAIN')
    rng=np.random.default_rng(cfg['seed']);lrng=np.random.default_rng(cfg['seed']+6100)
    per=defaultdict(Counter);counts=Counter();unique=set();trace=hashlib.sha256()
    for step in range(cfg['steps']):
        indices=train[rng.integers(len(train),size=cfg['batch_size'])]
        tags_batch=[]
        for s in indices:
            n=int(support['mask'][s].sum());tags=support['modes'][s,:n];tags_batch.append(tags)
            if arm=='set_matching':chosen=np.arange(n)
            else:
                p=probabilities(tags,arm);w=np.array([p[t]/np.count_nonzero(tags==t) for t in tags])
                chosen=lrng.choice(n,8,replace=True,p=w)
            trace.update(np.asarray([s,len(chosen)],dtype='<i8').tobytes())
            trace.update(np.asarray(chosen,dtype='<i8').tobytes())
            for j in chosen:
                per[str(support['ids'][s])][str(tags[j])]+=1
                counts[str(tags[j])]+=1;unique.add('%d:%d'%(s,j))
        if arm=='set_matching':
            for tags in tags_batch:lrng.permutation(len(set(tags)))
    missing=0;total=0;records=[]
    for s in train:
        ident=str(support['ids'][s]);known=set(support['modes'][s,support['mask'][s]])
        rare=known-{'gap0|gap0'};unseen=sorted(g for g in rare if per[ident][g]==0)
        missing+=len(unseen);total+=len(rare)
        records.append(dict(id=ident,counts=dict(per[ident]),unexposed_minorities=unseen))
    return dict(global_counts=dict(counts),unique_routes=len(unique),target_trace_sha256=trace.hexdigest(),
        unexposed_minority_request_modes=missing,total_minority_request_modes=total,rows=records)


def initial_rows(root):
    audit=root/'audit_v4';rows=read(audit/'candidate_rows.json')['0']
    refs={r['id']:set(r['relation_counts']) for r in read(audit/'data_requests.json')}
    for r in rows:
        rare=refs[r['id']]-{'gap0|gap0'}
        raw={w for w,v in zip(r['words'],r['valid']) if v}
        selected={r['words'][j] for j in r['selected_indices'] if r['valid'][j]}
        r['rare_recall8']=len(raw&rare)/len(rare);r['rare_recall4']=len(selected&rare)/len(rare)
    return rows


def main(root, output, evaluation='evaluation_fixed_q_v2'):
    import torch
    from routeset.observed_training_audit import tensor_state_digest
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    cfg=read(root/'frequency_empirical_uniform/config.json')['policy']
    with np.load(root/'frequency_support_v1/support.npz') as z:support={k:z[k] for k in z.files}
    baseline=initial_rows(root);results={};streams={};states={};hashes={}
    for arm in ARMS:
        f=root/('frequency_'+arm);s=read(f/'summary.json');c=read(f/'config.json')
        assert s['last_sha256']==sha(f/'last.pt')
        checkpoint=torch.load(f/'last.pt',map_location='cpu',weights_only=False)
        assert checkpoint['step']==cfg['steps'];assert checkpoint['settings']==c
        exposure=replay_exposure(support,cfg,arm)
        assert exposure['global_counts']==s['mode_exposure']==checkpoint['mode_exposure']
        assert exposure['unique_routes']==s['unique_routes']==len(checkpoint['unique_routes'])
        assert tensor_state_digest(checkpoint['model'])==s['final_sha256']
        write(out/(arm+'_exposure.json'),exposure)
        rows=read(f/evaluation/'rows.json');metrics=read(f/evaluation/'metrics.json')
        assert metrics['generator_sha256']==s['last_sha256']
        assert metrics['pool_sha256']==sha(f/evaluation/'pool.npz')
        assert metrics['scoring_contract']=='fixed complete deployment scorer and observation encoder'
        with np.load(f/'evaluation/pool.npz') as old,np.load(f/evaluation/'pool.npz') as fixed:
            for key in ('paths','events','ids','parents','labels'):
                np.testing.assert_array_equal(old[key],fixed[key])
            score_change=dict(mean_absolute=float(np.abs(old['q']-fixed['q']).mean()),
                max_absolute=float(np.abs(old['q']-fixed['q']).max()))
        states[arm]=s['final_sha256'];streams[arm]=s['sampler']
        hashes[arm]=dict(checkpoint=s['last_sha256'],pool=metrics['pool_sha256'],config=sha(f/'config.json'),
            rows=sha(f/evaluation/'rows.json'),scorer_bundle=metrics['scorer_bundle_sha256'])
        results[arm]=dict(versus_initial=paired(baseline,rows),fixed_encoder_score_change=score_change,
            exposure={k:v for k,v in exposure.items() if k!='rows'},
            by_variant={v:dict(n=sum(r['variant']==v for r in rows),
                collision_fraction=float(np.mean([not c['post_segments_clear'] for r in rows if r['variant']==v for c in r['candidates']])),
                semantic_failure_fraction=float(np.mean([not c['semantic_goal_correct'] for r in rows if r['variant']==v for c in r['candidates']])))
                for v in ('open','closed','shifted')})
        del checkpoint
    assert all(stream==streams[ARMS[0]] for stream in streams.values())
    assert len({v['scorer_bundle'] for v in hashes.values()})==1
    assert states['balanced']==states['empirical_uniform']
    with np.load(root/'frequency_balanced'/evaluation/'pool.npz') as b,np.load(root/'frequency_empirical_uniform'/evaluation/'pool.npz') as u:
        assert b.files==u.files
        for k in b.files:np.testing.assert_array_equal(b[k],u[k])
    initial={k:float(np.mean([row_metrics(r)[k] for r in baseline])) for k in row_metrics(baseline[0])}
    write(out/'RESULTS.json',dict(initial_baseline=initial,results=results,artifact_sha256=hashes,
        identical_initialization_and_observation_stream=True,uniform_balanced_exact=True,
        initial_source_sha256=sha(root/'audit_v4/candidate_rows.json'),support_sha256=sha(root/'frequency_support_v1/support.npz'),
        scope='Read-only exposure replay and fixed-seed family analysis; no optimizer or TEST_LOCKED access'))
    print(json.dumps(dict(initial=initial,uniform_balanced_exact=True,
        missing={a:results[a]['exposure']['unexposed_minority_request_modes'] for a in ARMS})),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.root,a.output)
