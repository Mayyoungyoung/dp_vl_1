"""One concentrated real-center opportunity analysis and repair target construction."""
import argparse,time
from collections import Counter
import numpy as np
import torch
from research_selective_repair_v1.core import *
from research_selective_repair_v1.local import features,observed_points,optimize
from routeset.observed_probability import load_scored_planner
from scripts.research_v3_audit import mode,exact_clear

def prepare(name='prepared_v1',data=None,roles=('TRAIN','DEV_MODEL')):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    data=data or DATA
    model,_=center_model();head=base_success();scorer=load_scored_planner(Q,'cuda').requires_grad_(False)
    rows=[r for r in lines(data/'export/observations.jsonl') if r['split'] in roles]
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split'] in roles}
    from scripts.mode_geometry_experiment import POOL
    with np.load(POOL) as z:lib={k:z[k] for k in ('ids','splits','paths','modes','mask')}
    assert set(lib['splits'])=={'TRAIN'}
    libindex={str(ident):i for i,ident in enumerate(lib['ids'])}
    arrays={k:[] for k in ('ids','splits','families','drafts','events','modes','context','anchor','current','local','targets','support','usable','valid','geometric','obstacle_centers','obstacle_halves','floors')}
    records=[];counts={role:Counter() for role in roles};hashes={};tic=time.monotonic()
    for j,row in enumerate(rows):
        inp=inputs_for(row,labels[row['id']],data/'export/qwen_cache',torch,hashes)
        runner=SceneRunner(model,scorer,inp);m,v=base_queries(runner.context,runner,head)
        with torch.no_grad():p,ev,_=model.decode(runner.context,runner.anchor,inp['current'],m,v)
        p,ev=p[0].cpu().numpy(),ev[0].cpu().numpy()
        # Current-observation fields are computed before any truth is read.
        with np.load(labels[row['id']]['observation']) as z:observation={k:z[k] for k in ('depth','camera_intrinsics','camera_extrinsics')}
        points=observed_points(inp);local=features(p,points,observation)
        geometric,gmask=optimize(p,points=points,steps=read(POLICY)['geometric_steps'])
        ref=references(labels[row['id']]);cs=ref['truth']['obstacle_centers'];hs=ref['truth']['obstacle_halfsizes'];floor=ref['config']['post_base_z']+.02
        _,cc=check_candidates(p,ev,ref['label'],ref['current'],ref['truth'],ref['config']);valid=np.array([c['TipValid'] for c in cc]);words=[mode(x,ref['config']) for x in p]
        goal=np.asarray(ref['label']['semantic_targets']['centers'])[ref['label']['semantic_targets']['target_index']]
        target_ok=np.linalg.norm(p[:,-1]-goal,axis=-1)<=.03;mode_ok=np.array(words)==np.array(VOCAB)[m[0].cpu().numpy()]
        eligible=~valid&target_ok&mode_ok
        teacher,tmask=optimize(p,cs,hs,floor,steps=read(POLICY)['teacher_steps'])
        _,tc=check_candidates(teacher,ev,ref['label'],ref['current'],ref['truth'],ref['config'])
        targets=p.copy();usable=valid.copy();found=np.zeros(8,bool);library=np.zeros(8,bool)
        for k in np.flatnonzero(eligible):
            opts=[]
            if tc[k]['TipValid'] and mode(teacher[k],ref['config'])==words[k]:opts.append(teacher[k])
            li=libindex.get(row['id']) if row['split']=='TRAIN' else None
            lp=lib['paths'][li,lib['mask'][li]] if li is not None else ref['paths']
            lv=np.ones(len(lp),bool) if li is not None else ref['reference_valid']
            for rp,rv in zip(lp,lv):
                if rv and mode(rp,ref['config'])==words[k]:
                    library[k]=True;candidate=rp.copy();candidate[0]=p[k,0];candidate[-1]=p[k,-1]
                    changed=np.linalg.norm(candidate-p[k],axis=-1)>.001;ix=np.flatnonzero(changed)
                    if len(ix) and ix[-1]-ix[0]+1<=12 and np.abs(candidate-p[k]).max()<=.08:
                        _,lc=check_candidates(candidate[None],ev[k:k+1],ref['label'],ref['current'],ref['truth'],ref['config'])
                        if lc[0]['TipValid'] and mode(candidate,ref['config'])==words[k]:opts.append(candidate)
            if opts:targets[k]=min(opts,key=lambda z:np.square(z-p[k]).sum());usable[k]=found[k]=True
        support=(np.linalg.norm(targets-p,axis=-1)>.001).astype(np.float32)
        _,gc=check_candidates(geometric,ev,ref['label'],ref['current'],ref['truth'],ref['config']);gv=np.array([c['TipValid'] for c in gc])
        base=wordset(p,valid,ref['config']);repair=wordset(targets,usable,ref['config']);optimistic=base|{words[k] for k in np.flatnonzero(eligible&library)}
        c=counts[row['split']];c.update(requests=1,routes=8,valid=int(valid.sum()),local_geometry_eligible=int(eligible.sum()),found_local_repairs=int(found.sum()),unknown_unresolved=int((eligible&~found).sum()),optimistic_new_modes=len(optimistic-base),found_new_modes=len(repair-base),geometric_added=len(wordset(geometric,gv,ref['config'])-base),geometric_lost=len(base-wordset(geometric,gv,ref['config'])))
        primary=['valid' if valid[k] else 'goal' if not target_ok[k] else 'mode' if not mode_ok[k] else 'local_geometry' for k in range(8)]
        c.update({'primary_'+key:primary.count(key) for key in set(primary)})
        record=dict(id=row['id'],family=row['parent_id'].rsplit('_',1)[0],role=row['split'],primary=primary,valid=valid,mode_ok=mode_ok,target_ok=target_ok,geometry_ok=exact_clear(p,cs,hs),floor_ok=(p[:,:,2]>=floor).all(1),eligible=eligible,found=found,library=library,unknown_fraction=local[:,:,18].mean(1),candidates=cc,points=len(points),body_constraints='not yet checked')
        records.append(plain(record))
        vals=dict(ids=row['id'],splits=row['split'],families=record['family'],drafts=p,events=ev,modes=m[0].cpu().numpy(),context=runner.context[0].cpu().numpy(),anchor=runner.anchor[0].cpu().numpy(),current=inp['current'][0].cpu().numpy(),local=local,targets=targets,support=support,usable=usable,valid=valid,geometric=geometric,obstacle_centers=cs,obstacle_halves=hs,floors=floor)
        for key,val in vals.items():arrays[key].append(val)
        if (j+1)%64==0:print(plain(dict(requests=j+1,counts=counts)),flush=True)
    np.savez_compressed(out/'samples.npz',**arrays)
    report=dict(counts=plain(counts),elapsed_seconds=time.monotonic()-tic,base_sha256=sha(BASE),base_head_sha256=sha(BASE_HEAD),samples_sha256=sha(out/'samples.npz'),input_hashes=hashes,locked_access=False,teacher='Finite64step exact-geometry local optimization and nearest verified same-mode library; not global minimum',unknown_policy='No found solution is unknown, not infeasible; ray unknown is distinct from free')
    write(out/'RECORDS.json',records);write(out/'MANIFEST.json',report);print(plain({k:v for k,v in report.items() if k!='input_hashes'}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='prepared_v1');p.add_argument('--data',type=Path);a=p.parse_args();prepare(**vars(a))
