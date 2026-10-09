"""DEV-only fair trigger/scale tuning, full frozen q, explicit diagnostic budget."""
import argparse,time
import numpy as np
import torch
from research_selective_repair_v1.core import *
from research_selective_repair_v1.model import load_repair
from routeset.observed_probability import load_scored_planner,route_observation_features
from scripts.research_v3_audit import mode

def grid(name,checkpoint,data_name='goal_prepared_v1',data=None):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False);data=data or DATA
    center,head,ck=load_repair(checkpoint);center.requires_grad_(False);head.requires_grad_(False)
    with np.load(RUN/data_name/'samples.npz') as z:
        ix=z['splits']=='DEV_MODEL';d={k:z[k][ix] for k in z.files}
    t={k:torch.as_tensor(d[k],device='cuda',dtype=torch.long if k=='modes' else torch.float32) for k in ('context','anchor','current','drafts','local','modes')}
    with torch.no_grad():
        drafts=t['drafts']
        if ck['settings']['arm']=='recenter':drafts=center.corridor(t['context'],t['anchor'],t['current'],t['modes'],torch.zeros_like(t['modes']))[0]
        if ck['settings']['arm']=='local_diffusion':settings=[dict(threshold=x,scale=s) for x in (0,.01,.025,.05) for s in read(POLICY)['scales']]
        elif ck['settings']['arm']=='selective':settings=[dict(threshold=x,scale=1.) for x in read(POLICY)['thresholds']]
        else:settings=[dict(threshold=.5,scale=x) for x in read(POLICY)['scales']]
        generated=[head(t['context'],t['modes'],drafts,t['local'],hard=True,**s)[0].cpu().numpy() for s in settings]
    # Above uses legal cached observed intermediates only. Truth starts below.
    observations={r['id']:r for r in lines(data/'export/observations.jsonl') if r['split']=='DEV_MODEL'}
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    scorer=load_scored_planner(Q,'cuda').requires_grad_(False);values=[[] for _ in settings];hashes={};tic=time.monotonic()
    for i,ident in enumerate(d['ids']):
        ident=str(ident);inp=inputs_for(observations[ident],labels[ident],data/'export/qwen_cache',torch,hashes);runner=SceneRunner(center,scorer,inp)
        paths=np.stack([g[i] for g in generated]);events=np.repeat(d['events'][i:i+1],len(settings),axis=0);qs=[]
        for j in range(0,len(settings),4):
            n=min(4,len(settings)-j);expand=lambda x:x.expand(n,*x.shape[1:])
            with torch.no_grad():
                p=torch.tensor(paths[j:j+n],device='cuda');ev=torch.tensor(events[j:j+n],device='cuda')
                nodes,ctx=route_observation_features(p,ev,expand(inp['current']),expand(inp['world_xyz']),expand(inp['rgb']),expand(inp['valid_mask']),expand(runner.qgeo['point_features']),expand(runner.qcontext),expand(runner.qgeo['anchor_xyz']))
                q=(scorer.scorer((nodes-scorer.nodes_mean)/scorer.nodes_std,(ctx-scorer.context_mean)/scorer.context_std)/scorer.temperature).sigmoid()
            qs.extend(q.cpu().numpy())
        ref=references(labels[ident]);a=assess(paths,events,np.asarray(qs),ref);base=assess(d['drafts'][i:i+1],d['events'][i:i+1],np.asarray(qs[:1]),ref);old=wordset(d['drafts'][i],base['valid'][0],ref['config'])
        for j in range(len(settings)):
            new=wordset(paths[j],a['valid'][j],ref['config'])
            values[j].append(dict(id=ident,family=str(d['families'][i]),utility=a['utility'][j].tolist(),added=len(new-old),lost=len(old-new),repaired=int((~base['valid'][0]&a['valid'][j]).sum()),damaged=int((base['valid'][0]&~a['valid'][j]).sum()),base_valid=int(base['valid'][0].sum())))
    rows=[]
    for setting,v in zip(settings,values):
        damage=sum(r['damaged'] for r in v)/sum(r['base_valid'] for r in v)
        rows.append(dict(setting=setting,mean_utility=np.mean([r['utility'] for r in v],0).tolist(),added=sum(r['added'] for r in v),lost=sum(r['lost'] for r in v),repaired=sum(r['repaired'] for r in v),damaged=sum(r['damaged'] for r in v),damage_fraction=damage,damage_budget_pass=damage<=read(POLICY)['damage_budget'],rows=v))
    eligible=[r for r in rows if r['damage_budget_pass']]
    winner=max(eligible,key=lambda r:(r['mean_utility'][0],r['mean_utility'][2],-r['damage_fraction'])) if eligible else min(rows,key=lambda r:r['damage_fraction'])
    report=dict(arm=ck['settings']['arm'],checkpoint_sha256=sha(checkpoint),grid=rows,selected_setting=winner['setting'],selected_mean=winner['mean_utility'],selected_damage=winner['damage_fraction'],satisfies_damage_budget=winner['damage_budget_pass'],selection_scope='Reused DEV only; no confirmation use',postprocessing_q_sets=len(settings)*len(d['ids']),observed_encoding_sets=len(d['ids']),cached_repair_forward_sets=len(settings)*len(d['ids']),seconds=time.monotonic()-tic,locked_access=False)
    write(out/'RESULTS.json',plain(report));print(plain({k:v for k,v in report.items() if k!='grid'}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--data-name',default='goal_prepared_v1');p.add_argument('--data',type=Path);grid(**vars(p.parse_args()))
