"""One observed8route decode + original frozen4return; truth only after prediction."""
import argparse,json,time
import numpy as np
import torch
from research_realized_coverage_v1.core import Q,DATA,old,write,read,lines,sha,torch_setup,VOCAB,SceneRunner,assess
from research_realized_coverage_v1.allocator import SuccessHead,allocate
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.model import load_model
from research_feasible_space_v1.geometry import segment_radii,node_radii_numpy
from scripts.evaluate_paired_modes import inputs_for,references,check_candidates
from scripts.research_v3_audit import coverage,average,plain
from routeset.observed_probability import load_scored_planner

def evaluate(name,checkpoint,head=None,kind=None,perturb=False,data_root=None,expected_requests=288):
    torch_setup();model,ck=load_model(checkpoint);model.requires_grad_(False)
    if kind:model.kind=kind
    scorer=load_scored_planner(Q,'cuda').requires_grad_(False);proposal=None
    if head:
        saved=torch.load(head,map_location='cpu',weights_only=False);assert saved['settings']['generator_sha256']==sha(checkpoint),'Stale success head'
        proposal=SuccessHead().cuda();proposal.load_state_dict(saved['model']);proposal.eval()
    out=RUN/name/'eval_adaptive';out.mkdir(parents=True,exist_ok=False)
    data=data_root or DATA
    if data_root is not None:
        assert data.resolve()==(DATA.parent/'feasible_space_generalization_v1').resolve()
        registration=read(data/'registration.json');manifest=read(data/'export/manifest.json')
        assert registration['protocol']==manifest['protocol']=='feasible_space_frozen_generalization_v1'
        assert registration['evaluation_only'] and manifest['evaluation_only'] and expected_requests==336
        assert manifest['failed_parents']==[] and manifest['observations']==336
    rows=[r for r in lines(data/'export/observations.jsonl') if r['split']=='DEV_MODEL'];assert len(rows)==expected_requests
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    if data_root is None:
        with np.load(old.SUPPORT/'support.npz') as z:known={str(k):set(z['modes'][i,z['mask'][i]]) for i,k in enumerate(z['ids']) if z['splits'][i]=='DEV_MODEL'}
    else:
        # Incomplete geometric teachers supply evaluation recall only, never inputs.
        from scripts.research_v3_audit import mode
        known={}
        for k,r in labels.items():
            ref=references(r)
            known[k]={mode(p,ref['config']) for p,valid in zip(ref['paths'],ref['reference_valid']) if valid}
            known[k].discard(None)
    results=[];pools={k:[] for k in ('paths','events','q','ids','mode_ids','variant_ids','centers','radii')};hashes={};times=[];stability=[]
    calls=[];capture=[]
    hook=model.relative_output.register_forward_hook(lambda module,args,result:calls.append(tuple(result.shape)))
    def corridor_hook(module,args,result):capture.append(result.detach())
    # Direct decode hook records predicted geometry; it is saved before checking truth.
    original=model.decode
    def decoded(*args,**kwargs):
        p,e,info=original(*args,**kwargs);capture.append((info['centers'].detach().cpu().numpy(),info['radii'].detach().cpu().numpy()));return p,e,info
    model.decode=decoded
    for row in rows:
        inp=inputs_for(row,labels[row['id']],data/'export/qwen_cache',torch,hashes);runner=SceneRunner(model,scorer,inp)
        with torch.inference_mode():
            m=torch.tensor(runner.base[None],device='cuda');v=torch.tensor((runner.base_variants%2)[None],device='cuda')
            if proposal:m,v,_=allocate('success',proposal,runner.context,m,v);v.zero_()
            torch.cuda.synchronize();tic=time.monotonic();p,e,q=runner.run(m.cpu().numpy(),v.cpu().numpy());torch.cuda.synchronize();times.append(time.monotonic()-tic)
        centers,radii=capture[-1];centers,radii=centers[0],radii[0]
        # Persisted numerical predictions precede independent evaluation inputs.
        ref=references(labels[row['id']]);a=assess(p,e,q,ref);p,e,q=p[0],e[0],q[0]
        _,cc=check_candidates(p,e,ref['label'],ref['current'],ref['truth'],ref['config']);assert a['valid'][0].tolist()==[x['TipValid'] for x in cc]
        valid=a['valid'][0];chosen=a['selected'][0];words=[VOCAB[w] if w>=0 else None for w in a['words'][0]]
        rr,slack=segment_radii(centers,ref['truth']['obstacle_centers'],ref['truth']['obstacle_halfsizes'],ref['config']['post_base_z']+.02,factor=1.,cap=1.)
        if ck['settings'].get('tapered_cells',False):
            from research_feasible_space_v1.tapered import tapered_clearance_numpy
            ts=tapered_clearance_numpy(centers,node_radii_numpy(radii),ref['truth']['obstacle_centers'],ref['truth']['obstacle_halfsizes']+.02,ref['config']['post_base_z']+.02)
            corridor_safe=(ts>0).all(1)
        else:corridor_safe=(slack>0).all(1)&(radii<rr).all(1)
        constrained=(np.abs(p-centers)<=node_radii_numpy(radii)[...,None]+1e-6).all((1,2))
        results.append(dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],variant=row['parent_id'].rsplit('_',1)[1],
            raw=coverage(words,valid,known[row['id']],range(8)),selected=coverage(words,valid,known[row['id']],chosen),
            words=words,valid=valid.tolist(),q=q.tolist(),assigned_modes=[VOCAB[j] for j in m[0].tolist()],raw_words=[VOCAB[j] if j>=0 else None for j in a['raw_words'][0]],
            selected_indices=chosen.tolist(),candidates=cc,condition_hit=float(np.mean(valid&(a['words'][0]==m[0].cpu().numpy()))),
            corridor_certified_under_eval_truth=corridor_safe.tolist(),inside_predicted_cells=constrained.tolist(),
            invalid_with_bad_corridor=int((~valid&~corridor_safe).sum()),invalid_with_certified_corridor=int((~valid&corridor_safe).sum())))
        for k,val in dict(paths=p,events=e,q=q,ids=row['id'],mode_ids=m[0].cpu().numpy(),variant_ids=v[0].cpu().numpy(),centers=centers,radii=radii).items():pools[k].append(val)
        if perturb:
            mm=m.clone();mm[:,0]=(mm[:,0]+1)%16
            with torch.inference_mode():pp,ee,qq=runner.run(mm.cpu().numpy(),v.cpu().numpy())
            aa=assess(pp,ee,qq,ref);newc,newr=capture[-1]
            stability.append(dict(id=row['id'],mean_companion_motion_m=float(np.linalg.norm(pp[0,1:]-p[1:],axis=-1).mean()),
                corridor_boundary_motion_m=float(np.max(np.abs(newc[0,1:]-centers[1:]))),radius_change_m=float(np.max(np.abs(newr[0,1:]-radii[1:]))),
                valid_companions_lost=int((valid[1:]&~aa['valid'][0,1:]).sum()),mode_companions_changed=int((a['raw_words'][0,1:]!=aa['raw_words'][0,1:]).sum()),net_u8=float(aa['utility'][0,0]-a['utility'][0,0])))
    np.savez_compressed(out/'pool.npz',**pools);hook.remove();assert len(calls)==len(rows)*(2 if perturb else 1) and all(s[:2]==(1,8) for s in calls)
    report=dict(raw=average([r['raw'] for r in results]),selected=average([r['selected'] for r in results]),condition_hit=float(np.mean([r['condition_hit'] for r in results])),
        generator_sha256=sha(checkpoint),head_sha256=sha(head) if head else None,scorer_sha256=sha(Q),pool_sha256=sha(out/'pool.npz'),cell_form='tapered_endpoint_boxes' if ck['settings'].get('tapered_cells',False) else 'uniform_swept_boxes',
        corridor_feasibility=float(np.mean([r['corridor_certified_under_eval_truth'] for r in results])),containment=float(np.mean([r['inside_predicted_cells'] for r in results])),
        invalid_with_bad_corridor=sum(r['invalid_with_bad_corridor'] for r in results),invalid_with_certified_corridor=sum(r['invalid_with_certified_corridor'] for r in results),
        cost=dict(decoded_sets=len(rows),diagnostic_extra_decodes=len(rows) if perturb else 0,generated_routes=len(rows)*8,decode_q_ms_mean=float(np.mean(times)*1000)),input_hashes=hashes,locked_access=False,
        data_root=str(data),expected_requests=expected_requests,recall_scope='Sparse fresh geometric teachers' if data_root else 'Existing all-mode support')
    write(out/'rows.json',plain(results));write(out/'metrics.json',plain(report));write(out/'stability.json',stability)
    print(json.dumps(plain({k:v for k,v in report.items() if k!='input_hashes'})),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',required=True,type=type(RUN));p.add_argument('--head',type=type(RUN))
    p.add_argument('--kind',choices=['bounded','relative','xyz','center','projection']);p.add_argument('--perturb',action='store_true')
    p.add_argument('--data-root',type=type(RUN));p.add_argument('--expected-requests',type=int,default=288);evaluate(**vars(p.parse_args()))
