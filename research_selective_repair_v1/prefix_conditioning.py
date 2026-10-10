"""One fixed prefix-loop falsification, no IK/body checker in inference.

Both null and loop must satisfy the SAME current-NN geometry/type and8cm
per-node bound; otherwise both return identity. First three nodes alone change.
Closed task-space loops need not induce closed joint paths in redundant IK,
but this experiment does not assume this native controller exhibits that effect.
"""
import argparse,time
import numpy as np
from research_selective_repair_v1.io import RUN,SOURCE,read,write,sha
from research_selective_repair_v1.execution_semantics import word,tip_clear

def proposals(paths,completed,post_base):
    p=np.asarray(paths);assert p.shape==(8,24,3)
    height=np.clip(completed[:,2]-post_base,.025,.4)
    cfg=dict(row_x=[float(completed[j:j+2,0].mean()) for j in (0,2)],post_y=[completed[j:j+2,1].tolist() for j in (0,2)],
        post_heights=[float(height[j:j+2].max()) for j in (0,2)],post_base_z=post_base,tip_clearance_m=.02)
    null=p.copy();loop=p.copy();root=p[:,0]
    null[:,1:4]=root[:,None]
    loop[:,1]=root+[-.02,0,.04];loop[:,2]=root+[0,0,.04];loop[:,3]=root
    ok=[];reasons=[]
    for i in range(8):
        old=word(p[i],cfg)
        bounded=all(np.linalg.norm(candidate-p[i],axis=-1).max()<=.08+1e-7 for candidate in (null[i],loop[i]))
        proposed_words=[word(candidate,cfg) for candidate in (null[i],loop[i])]
        same=old is not None and all(w==old for w in proposed_words)
        clear=all([tip_clear(candidate,cfg) for candidate in (p[i],null[i],loop[i])])
        eligible=bool(bounded and same and clear);ok.append(eligible)
        reasons.append(dict(bounded=bool(bounded),same_predicted_word=bool(same),predicted_clear=bool(clear),predicted_word_queries=3,predicted_tip_queries=3))
        if not eligible:null[i]=loop[i]=p[i]
    assert np.array_equal(null[:,0],p[:,0]) and np.array_equal(loop[:,0],p[:,0])
    assert np.array_equal(null[:,4:],p[:,4:]) and np.array_equal(loop[:,4:],p[:,4:])
    return np.stack([p,null,loop],1),np.asarray(ok),reasons

def diagnostic(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
    source=RUN/'constraints_global_interventions_TRAIN_body_v1';file=source/'sealed_predictions.npz'
    seal=read(source/'SEAL.json');assert seal['role']=='TRAIN' and seal['current_observation_only'] and seal['prediction_sha256']==sha(file)
    base=read(RUN/'constraints_seed0_v1/config.json')['post_base'];rows=[]
    with np.load(file) as z:
        for ident,p,c in zip(z['ids'],z['paths'],z['completed']):
            _,ok,reasons=proposals(p,c,base)
            rows.append(dict(id=str(ident),eligible=int(ok.sum()),mask=ok.tolist(),reasons=reasons))
    write(out/'SUMMARY.json',dict(rows=rows,requests=len(rows),eligible_slots=sum(r['eligible'] for r in rows),
        candidate_slots=8*len(rows),source_sha256=sha(file),source_commit=SOURCE.name,seconds=time.monotonic()-tic,
        primitive='Single fixed -2cmX/+4cmZ triangle; null repeats start for three nodes; same eligibility intersection; no amplitude/grid search',
        guards='Unchanged first/endpoints,nodes4..23 unchanged,max8cm per-node,current-NN mode/type-clear proxies only',
        geometry_scope='Pure geometry functions consume predicted current boxes,not ground truth/checker labels; no body feasibility certificate',
        scope='TRAIN support diagnostic only. No actual native controller/body/joint effect measured',locked_access=False))
    print(dict(requests=len(rows),eligible_slots=sum(r['eligible'] for r in rows),candidate_slots=8*len(rows)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);diagnostic(**vars(p.parse_args()))
