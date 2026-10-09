"""Fixed 8->8->4 evaluation, with actual added/lost word sets and damage."""
import argparse,time
import numpy as np
from research_selective_repair_v1.grounding import predict
import torch
from research_selective_repair_v1.core import *
from research_selective_repair_v1.model import load_repair
from research_selective_repair_v1.local import features,observed_points,optimize
from routeset.observed_probability import route_observation_features,load_scored_planner
from scripts.research_v3_audit import coverage,average,mode

def evaluate(name,checkpoint=None,kind='learned',threshold=.5,scale=1.,data=None,role='DEV_MODEL',prototype=None,allocation_head=None,baseline_pool=None):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False);data=data or DATA
    if role not in ('TRAIN','DEV_MODEL'):raise ValueError('Role boundary')
    if checkpoint:center,head,ck=load_repair(checkpoint)
    else:center,_=center_model();head=None
    center.requires_grad_(False)
    if head:head.requires_grad_(False)
    proposal=base_success();scorer=load_scored_planner(Q,'cuda').requires_grad_(False)
    if allocation_head:
        h=torch.load(allocation_head,map_location='cpu',weights_only=False)
        assert h['settings']['generator_sha256']==sha(checkpoint)
        proposal.load_state_dict(h['model']);proposal.eval()
        assert ck['view']==dict(parent_sha256=ck['view']['parent_sha256'],prototype_sha256=sha(prototype),threshold=threshold,scale=scale)
    baseline_arrays=None
    if baseline_pool:
        with np.load(baseline_pool) as z:baseline_arrays={k:z[k] for k in ('ids','paths','events')}
        baseline_index={str(ident):i for i,ident in enumerate(baseline_arrays['ids'])}
    if prototype:
        from scripts import observation_prototype_grounding as proto
        prototype_model=read(prototype)
    rows=[r for r in lines(data/'export/observations.jsonl') if r['split']==role]
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']==role}
    results=[];pools={k:[] for k in ('ids','paths','drafts','events','q','modes','support','prob')};hashes={};times=[]
    for j,row in enumerate(rows):
        inp=inputs_for(row,labels[row['id']],data/'export/qwen_cache',torch,hashes);runner=SceneRunner(center,scorer,inp);m,v=base_queries(runner.context,runner,proposal)
        torch.cuda.synchronize();tic=time.monotonic()
        with torch.no_grad():drafts,ev,_=center.decode(runner.context,runner.anchor,inp['current'],m,v)
        d=drafts[0].cpu().numpy();prob=np.zeros((8,24));support=prob.copy()
        if kind=='geometric' or head is not None:
            points=observed_points(inp)
            with np.load(labels[row['id']]['observation']) as z:observation={k:z[k] for k in ('depth','camera_intrinsics','camera_extrinsics')}
            local=features(d,points,observation)
        if prototype:
            (rgb,xyz,vm),_=proto.load_observation(data/'export',row,labels[row['id']]['observation'])
            goal,detail=predict(rgb,xyz,vm,row['instruction'],prototype_model);available=goal is not None
            if goal is None:goal=runner.anchor[0].cpu().numpy()
            gf=np.concatenate([np.broadcast_to((goal-d[:,-1])/.4,(24,8,3)).transpose(1,0,2),np.full((8,24,1),available)],-1).astype(np.float32)
            if kind=='geometric' or head is not None:local=np.concatenate([local,gf],-1)
        if kind=='geometric':
            initial=d.copy()
            if prototype:
                s=np.linspace(0,1,7)[1:];s=s*s*(3-2*s)
                initial[:,18:]+=s[None,:,None]*(goal-d[:,-1])[:,None]
            pp,support=optimize(initial,points=points);p=torch.tensor(pp[None],device='cuda')
        elif kind=='goal_rule':
            diff=goal-d[:,-1];trigger=np.linalg.norm(diff,axis=-1)>=threshold if available else np.zeros(8,bool)
            pp=d.copy();s=np.linspace(0,1,7)[1:];s=s*s*(3-2*s)
            support[:,18:]=trigger[:,None]*s
            pp+=scale*support[...,None]*diff[:,None];p=torch.tensor(pp[None],device='cuda',dtype=drafts.dtype)
        elif head is not None:
            with torch.no_grad():p,info=head(runner.context,m,drafts,torch.tensor(local[None],device='cuda'),hard=True,threshold=threshold,scale=scale)
            support=info['support'][0].cpu().numpy();prob=info['prob'][0].cpu().numpy()
        else:p=drafts
        with torch.no_grad():
            expand=lambda x:x
            nodes,ctx=route_observation_features(p,ev,inp['current'],inp['world_xyz'],inp['rgb'],inp['valid_mask'],runner.qgeo['point_features'],runner.qcontext,runner.qgeo['anchor_xyz'])
            q=(scorer.scorer((nodes-scorer.nodes_mean)/scorer.nodes_std,(ctx-scorer.context_mean)/scorer.context_std)/scorer.temperature).sigmoid()
        torch.cuda.synchronize();times.append(time.monotonic()-tic)
        p,ev,q=p[0].cpu().numpy(),ev[0].cpu().numpy(),q[0].cpu().numpy()
        ref=references(labels[row['id']]);a=assess(p[None],ev[None],q[None],ref)
        baseline_path=d;baseline_event=ev
        if baseline_arrays:
            bi=baseline_index[row['id']];baseline_path=baseline_arrays['paths'][bi];baseline_event=baseline_arrays['events'][bi]
        b=assess(baseline_path[None],baseline_event[None],q[None],ref)
        valid=a['valid'][0];baseline=b['valid'][0];word=[VOCAB[w] if w>=0 else None for w in a['words'][0]];known={mode(rp,ref['config']) for rp,rv in zip(ref['paths'],ref['reference_valid']) if rv}-{None}
        new=wordset(p,valid,ref['config']);old=wordset(baseline_path,baseline,ref['config'])
        results.append(dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],variant=row['parent_id'].rsplit('_',1)[1],raw=coverage(word,valid,known,range(8)),selected=coverage(word,valid,known,a['selected'][0]),words=word,valid=valid.tolist(),base_valid=baseline.tolist(),added=sorted(new-old),lost=sorted(old-new),delta_U8=len(new-old)-len(old-new),repaired=int((~baseline&valid).sum()),damaged=int((baseline&~valid).sum()),raw_mode_changed=int((a['raw_words'][0]!=b['raw_words'][0]).sum()),edited_nodes=int((np.linalg.norm(p-d,axis=-1)>1e-8).sum()),selected_indices=a['selected'][0].tolist()))
        for k,val in dict(ids=row['id'],paths=p,drafts=d,events=ev,q=q,modes=m[0].cpu().numpy(),support=support,prob=prob).items():pools[k].append(val)
        if (j+1)%64==0:print(dict(requests=j+1),flush=True)
    np.savez_compressed(out/'pool.npz',**pools);write(out/'rows.json',results)
    report=dict(raw=average([r['raw'] for r in results]),selected=average([r['selected'] for r in results]),requests=len(rows),repaired=sum(r['repaired'] for r in results),damaged=sum(r['damaged'] for r in results),base_valid=sum(sum(r['base_valid']) for r in results),added=sum(len(r['added']) for r in results),lost=sum(len(r['lost']) for r in results),delta_U8=float(np.mean([r['delta_U8'] for r in results])),raw_mode_changed=sum(r['raw_mode_changed'] for r in results),cached_feature_repair_q_ms=float(np.mean(times)*1000),decode_sets=len(rows),generated_drafts=len(rows)*8,final_candidates=len(rows)*8,kind=kind,threshold=threshold,scale=scale,checkpoint_sha256=sha(checkpoint) if checkpoint else None,base_sha256=sha(BASE),base_head_sha256=sha(BASE_HEAD),allocation_head_sha256=sha(allocation_head) if allocation_head else sha(BASE_HEAD),scorer_sha256=sha(Q),data=str(data),role=role,locked_access=False,head_scope='Own matching TRAIN head' if allocation_head else 'Frozen base allocation head in screening; matching output-head controls required for formal acceptance',input_hashes=hashes)
    write(out/'METRICS.json',report);print({k:v for k,v in report.items() if k!='input_hashes'},flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--kind',choices=['learned','geometric','zero','goal_rule'],default='learned');p.add_argument('--threshold',type=float,default=.5);p.add_argument('--scale',type=float,default=1.);p.add_argument('--data',type=Path);p.add_argument('--role',default='DEV_MODEL');p.add_argument('--prototype',type=Path);p.add_argument('--allocation-head',type=Path);p.add_argument('--baseline-pool',type=Path);evaluate(**vars(p.parse_args()))
