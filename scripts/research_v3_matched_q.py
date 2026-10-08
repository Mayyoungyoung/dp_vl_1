"""Ordinary matched single-q with independent frozen observation features."""
import argparse
import copy
import sys
import subprocess
import time
import numpy as np
from scripts.run_observed_probability import read,write,sha,lines,torch_setup
from scripts.research_v3_frequency import RUN,fixed_path_scores,coverage
from scripts.paired_modes_data import RUN as OLD_RUN
from scripts.evaluate_paired_modes import dataset,inputs_for,references,check_candidates
from scripts import paired_modes_reliability as legacy
from scripts.analyze_paired_selection import select

ROOT=RUN/'matched_q_v1'
BUNDLE=OLD_RUN/'reliability/R1_seed0/deployment_seed0/planner.pt'
GEN=RUN/'safety_mean/last.pt'


def pool(role):
    torch=torch_setup()
    from routeset.observed_probability import load_scored_planner,route_observation_features
    folder,split=dataset(role)
    rows=[r for r in lines(folder/'observations.jsonl') if r['split']==split]
    labels={r['id']:r for r in lines(folder/'supervision.jsonl') if r['split']==split}
    scorer=load_scored_planner(BUNDLE,'cuda');generator=copy.deepcopy(scorer.generator)
    state=torch.load(GEN,map_location='cpu',weights_only=False)
    generator.load_state_dict(state['model']);generator.eval();generator.requires_grad_(False)
    out=ROOT/'evaluation'/role/'mean_seed0';out.mkdir(parents=True,exist_ok=False)
    vals={k:[] for k in ('paths','events','q','nodes','context','ids','parents','labels')};hashes={}
    for row in rows:
        inp=inputs_for(row,labels[row['id']],folder/'qwen_cache',torch,hashes)
        with torch.inference_mode():
            paths,events,_=generator(**inp)
            geo=scorer.generator.geometry(**inp,return_point_features=True)
            ctx=scorer.generator.head.feature_encoder(inp['features'])+scorer.generator.head.state_encoder(inp['current'])+geo['context']
            nodes,context=route_observation_features(paths,events,inp['current'],inp['world_xyz'],inp['rgb'],inp['valid_mask'],geo['point_features'],ctx,geo['anchor_xyz'])
            q=fixed_path_scores(scorer,paths,events,inp)
        for k,v in dict(paths=paths,events=events,q=q,nodes=nodes,context=context).items():vals[k].append(v[0].cpu().numpy())
        vals['ids'].append(row['id']);vals['parents'].append(row['parent_id'])
    # Seal actual proposals before reading privileged checking geometry.
    np.savez_compressed(out/'proposals.npz',**{k:np.array(v) for k,v in vals.items() if k!='labels'})
    for i,row in enumerate(rows):
        ref=references(labels[row['id']])
        _,cand=check_candidates(vals['paths'][i],vals['events'][i],ref['label'],ref['current'],ref['truth'],ref['config'])
        vals['labels'].append([c['TipValid'] for c in cand])
    vals={k:np.array(v) for k,v in vals.items()}
    if role=='paired_dev':
        with np.load(RUN/'safety_mean/evaluation_fixed_q_v2/pool.npz') as old:
            for key in ('paths','events','q','labels','ids','parents'):np.testing.assert_array_equal(vals[key],old[key])
    np.savez_compressed(out/'pool.npz',**vals)
    write(out/'receipt.json',dict(pool_sha256=sha(out/'pool.npz'),generator_sha256=sha(GEN),complete_scorer_sha256=sha(BUNDLE),
        input_sha256=hashes,role=role,requests=len(rows),paired_pool_exact=role=='paired_dev'))
    print(dict(role=role,requests=len(rows),valid=float(vals['labels'].mean())),flush=True)


def check_roles():
    parents={}
    for role in ('SCORE_TRAIN','DEV_SCORE','CALIBRATION','paired_dev'):
        data,_=legacy.pool('mean',0,role);parents[role]=set(map(str,data['parents']))
    for a in parents:
        for b in parents:
            if a!=b:assert not parents[a]&parents[b]
    write(ROOT/'roles.json',{k:sorted(v) for k,v in parents.items()})


def calibrate(seed):
    torch=torch_setup()
    from scipy.optimize import minimize_scalar
    from routeset.observed_probability import RouteValidityHead,load_scored_planner
    folder=legacy.location('mean',0);saved=torch.load(folder/f'q_seed{seed}/best.pt',map_location='cpu',weights_only=False)
    norm=saved['normalization'];model=RouteValidityHead().cuda().eval();model.load_state_dict(saved['model'])
    def predict(d):
        x=[torch.tensor((d[k]-norm[k+'_mean'])/norm[k+'_std'],device='cuda') for k in ('nodes','context')]
        with torch.inference_mode():return model(*x).cpu().numpy()
    cal,ch=legacy.pool('mean',0,'CALIBRATION');clog=predict(cal).astype(np.float64)
    def nll(t):
        z=clog/np.exp(t);return float(np.mean(np.logaddexp(0,z)-cal['labels']*z))
    fitted=minimize_scalar(nll,bounds=(-3,3),method='bounded');assert fitted.success
    temperature=float(np.exp(fitted.x));data,dh=legacy.pool('mean',0,'paired_dev');logits=predict(data)
    out=folder/f'calibration_seed{seed}';out.mkdir(exist_ok=False)
    # Persist exactly the torch float32 deployment arithmetic.
    q=(torch.tensor(logits)/temperature).sigmoid().numpy()
    np.savez_compressed(out/'predictions.npz',logits=logits,q=q,labels=data['labels'],ids=data['ids'])
    write(out/'RESULTS.json',dict(temperature=temperature,calibration_pool_sha256=ch,dev_pool_sha256=dh,
        calibration_nll_before=nll(0),calibration_nll_after=nll(fitted.x),
        uncalibrated=legacy.reliability_metrics(logits,data),calibrated=legacy.reliability_metrics(logits,data,temperature)))
    bundle=torch.load(BUNDLE,map_location='cpu',weights_only=False);planner=load_scored_planner(BUNDLE,'cuda')
    planner.scorer.load_state_dict(saved['model'])
    for k in ('nodes_mean','nodes_std','context_mean','context_std'):
        getattr(planner,k).copy_(torch.tensor(norm[k],device='cuda'))
    planner.temperature=temperature
    bundle.update(planner_state=planner.state_dict(),temperature=temperature,
        proposal_checkpoint_sha256=sha(GEN),scorer_checkpoint_sha256=sha(folder/f'q_seed{seed}/best.pt'))
    torch.save(bundle,out/'scorer_bundle.pt')
    if seed==0:
        f,_=dataset('paired_dev');row=next(r for r in lines(f/'observations.jsonl') if r['split']=='DEV_MODEL')
        label=next(r for r in lines(f/'supervision.jsonl') if r['id']==row['id'])
        cmd=[sys.executable,'-m','scripts.predict_research_v3','--bundle',str(out/'scorer_bundle.pt'),
            '--checkpoint',str(GEN),'--manifest',str(f/'observations.jsonl'),'--id',row['id'],
            '--observation',label['observation'],'--qwen-cache',str(f/'qwen_cache'),'--output',str(out/'public_cli.json'),'--k','4']
        tic=time.monotonic();subprocess.run(cmd,check=True);elapsed=time.monotonic()-tic;p=read(out/'public_cli.json')
        j=list(data['ids']).index(row['id'])
        for key in ('paths','events'):np.testing.assert_array_equal(p[key],data[key][j])
        # Batched scoring matmul vs one-request matmul may differ in last bits.
        np.testing.assert_allclose(p['q'],q[j],rtol=1e-5,atol=1e-6)
        np.testing.assert_array_equal(p['selected_indices'],select(data['paths'][j],q[j],4))
        write(out/'CLI_RECEIPT.json',dict(command=cmd,elapsed_seconds=elapsed,paths_events_exact=True,
            q_max_abs_error=float(np.abs(np.array(p['q'])-q[j]).max()),selected_exact=True,
            bundle_sha256=sha(out/'scorer_bundle.pt'),proposal_sha256=sha(GEN)))


def analyze():
    from scripts.research_v3_analyze_frequency import paired,row_metrics
    root=legacy.location('mean',0);base=read(RUN/'safety_mean/evaluation_fixed_q_v2/rows.json')
    data,_=legacy.pool('mean',0,'paired_dev');summary={};contrasts={}
    with np.load(RUN/'frequency_support_v1/support.npz') as z:support={k:z[k] for k in z.files}
    index={str(k):i for i,k in enumerate(support['ids'])}
    for seed in (0,1,2):
        with np.load(root/f'calibration_seed{seed}/predictions.npz') as z:q=z['q']
        rows=copy.deepcopy(base)
        for i,r in enumerate(rows):
            assert r['id']==str(data['ids'][i]);r['q']=q[i].tolist();j=index[r['id']]
            known=set(support['modes'][j,support['mask'][j]]);rare=known-{'gap0|gap0'}
            indices=select(data['paths'][i],q[i],4)
            r['selected']=coverage(r['words'],np.array(r['valid']),known,indices)
            found={r['words'][k] for k in indices if r['valid'][k]}
            r['rare_recall4']=len(found&rare)/len(rare)
        stats=[row_metrics(r) for r in rows];summary[str(seed)]={k:float(np.mean([r[k] for r in stats])) for k in stats[0]}
        contrasts[str(seed)]=paired(base,rows);write(root/f'calibration_seed{seed}/rows.json',rows)
    delta={k:float(np.mean([contrasts[str(s)][k]['delta'] for s in (0,1,2)])) for k in contrasts['0']}
    gate=delta['brier']<=-.01 and all(contrasts[str(s)]['brier']['delta']<0 for s in (0,1,2)) and delta['modes4']>=-.05 and delta['all_valid4']>=-.01
    write(ROOT/'RESULTS.json',dict(summary=summary,contrasts=contrasts,mean_delta=delta,ordinary_matched_gate=gate,
        probability_metrics={str(s):read(root/f'calibration_seed{s}/RESULTS.json') for s in (0,1,2)},
        scope='Three ordinary scorer seeds on one fixed generator; common exact candidate pool, no generator replication or novelty claim'))
    print(dict(summary=summary,mean_delta=delta,gate=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pool','roles','fit','calibrate','analyze'])
    p.add_argument('--role',choices=['SCORE_TRAIN','DEV_SCORE','CALIBRATION','paired_dev']);p.add_argument('--seed',type=int,choices=[0,1,2],default=0);a=p.parse_args()
    legacy.RUN=ROOT
    if a.stage=='pool':pool(a.role)
    elif a.stage=='roles':check_roles()
    elif a.stage=='fit':legacy.fit('mean',0,a.seed)
    elif a.stage=='calibrate':calibrate(a.seed)
    else:analyze()
