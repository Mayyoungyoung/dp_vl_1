"""TRAIN oracle local support opportunity, same single fixed triangle primitive.

Oracle failure indices are TRAIN target acquisition only, never deployment.
The geometric proposal function itself takes a supplied predicted/target window
and current inferred posts/path; it never queries native execution or true boxes.
"""
import argparse
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.execution_semantics import word,tip_clear

def proposals(paths,completed,post_base,starts):
    p=np.asarray(paths);assert p.shape==(8,24,3);starts=np.asarray(starts);assert starts.shape==(8,)
    height=np.clip(completed[:,2]-post_base,.025,.4)
    cfg=dict(row_x=[float(completed[j:j+2,0].mean()) for j in (0,2)],post_y=[completed[j:j+2,1].tolist() for j in (0,2)],
        post_heights=[float(height[j:j+2].max()) for j in (0,2)],post_base_z=post_base,tip_clearance_m=.02)
    null=p.copy();loop=p.copy();eligible=np.zeros(8,bool);reasons=[]
    for i,j in enumerate(starts):
        j=int(j);window=0<=j<=19
        if window:
            root=p[i,j];null[i,j+1:j+4]=root
            loop[i,j+1]=root+[-.02,0,.04];loop[i,j+2]=root+[0,0,.04];loop[i,j+3]=root
        bounded=all(np.linalg.norm(c-p[i],axis=-1).max()<=.08+1e-7 for c in (null[i],loop[i]))
        w=[word(c,cfg) for c in (p[i],null[i],loop[i])];same=w[0] is not None and w[1:]==w[:1]*2
        clear=all([tip_clear(c,cfg) for c in (p[i],null[i],loop[i])])
        eligible[i]=bool(window and bounded and same and clear)
        reasons.append(dict(start=j,window=bool(window),bounded=bool(bounded),same_predicted_word=bool(same),predicted_clear=bool(clear),predicted_word_queries=3,predicted_tip_queries=3))
        if not eligible[i]:null[i]=loop[i]=p[i]
    assert np.array_equal(null[:,0],p[:,0]) and np.array_equal(loop[:,-1],p[:,-1])
    return np.stack([p,null,loop],1),eligible,reasons

def progress_proposals(paths,completed,post_base,starts):
    """Two interior offsets retain baseline progression and exact window end.

    Unlike the closed-cycle template, no waypoint is spent returning backward
    to the window start. Late failures use the last two INTERIOR approach nodes,
    so the mandatory endpoint remains untouched. One fixed anchoring rule.
    """
    p=np.asarray(paths);starts=np.asarray(starts);assert p.shape==(8,24,3) and starts.shape==(8,)
    h=np.clip(completed[:,2]-post_base,.025,.4)
    cfg=dict(row_x=[float(completed[j:j+2,0].mean()) for j in (0,2)],post_y=[completed[j:j+2,1].tolist() for j in (0,2)],post_heights=[float(h[j:j+2].max()) for j in (0,2)],post_base_z=post_base,tip_clearance_m=.02)
    vertical=p.copy();side=p.copy();ok=np.zeros(8,bool);reasons=[]
    for i,stop in enumerate(starts):
        stop=int(stop);valid=0<=stop<=22;j=min(stop,20)
        if valid:
            vertical[i,j+1:j+3]+=np.array([0,0,.04],dtype=p.dtype)
            side[i,j+1]+=np.array([-.02,0,.04],dtype=p.dtype);side[i,j+2]+=np.array([0,0,.04],dtype=p.dtype)
        words=[word(c,cfg) for c in (p[i],vertical[i],side[i])]
        same=words[0] is not None and words[1:]==words[:1]*2
        clear=all([tip_clear(c,cfg) for c in (p[i],vertical[i],side[i])])
        bounded=all(np.linalg.norm(c-p[i],axis=-1).max()<=.08+1e-7 for c in (vertical[i],side[i]))
        ok[i]=bool(valid and same and clear and bounded)
        reasons.append(dict(observed_stop=stop,start=j,window=bool(valid),bounded=bool(bounded),same_predicted_word=bool(same),predicted_clear=bool(clear),predicted_word_queries=3,predicted_tip_queries=3))
        if not ok[i]:vertical[i]=side[i]=p[i]
    assert np.array_equal(vertical[:,0],p[:,0]) and np.array_equal(side[:,-1],p[:,-1])
    return np.stack([p,vertical,side],1),ok,reasons

def diagnostic(name,progress=False):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);dataset=RUN/'body_native_branch_data_v1/samples.npz'
    manifest=read(dataset.parent/'MANIFEST.json');assert manifest['no_DEV_feedback'] and manifest['samples_sha256']==sha(dataset)
    with np.load(dataset) as z:d={k:z[k] for k in ('ids','slots','options','paths','completed','labels','families','prefix_hazard','roles')}
    assert set(d['roles'])=={'TRAIN'};families=sorted(set(d['families']));fit=set(families[:-2]);base=read(RUN/'constraints_seed0_v1/config.json')['post_base'];rows=[]
    for ident in sorted(set(d['ids'][d['options']==0])):
        ix=np.flatnonzero((d['ids']==ident)&(d['options']==0));ix=ix[np.argsort(d['slots'][ix])];assert len(ix)==8
        failed=(d['labels'][ix]==0)&d['prefix_hazard'][ix].any(-1)
        starts=np.where(failed,d['prefix_hazard'][ix].argmax(-1),-1)
        candidates,ok,reasons=(progress_proposals if progress else proposals)(d['paths'][ix],d['completed'][ix[0]],base,starts)
        rows.append(dict(id=str(ident),family=str(d['families'][ix[0]]),fit_family=bool(d['families'][ix[0]] in fit),
            known_first_stop=int(failed.sum()),eligible=int(ok.sum()),starts=starts.tolist(),eligible_mask=ok.tolist(),reasons=reasons))
    # Active TRAIN acquisition is enriched deliberately; held families excluded.
    # Main-distribution acceptance can never be replaced by this diagnostic.
    candidates=sorted([r for r in rows if r['fit_family']],key=lambda r:(-r['eligible'],r['id']))
    selected=[];family_count={}
    for r in candidates:
        if r['eligible']==0:break
        if family_count.get(r['family'],0)>=2:continue
        selected.append(r);family_count[r['family']]=family_count.get(r['family'],0)+1
        if len(selected)==4:break
    write(out/'SUMMARY.json',dict(rows=rows,requests=len(rows),eligible_failed_slots=sum(r['eligible'] for r in rows),known_first_stop_slots=sum(r['known_first_stop'] for r in rows),
        selected_TRAIN_requests=selected,dataset_sha256=sha(dataset),progress_preserving=progress,
        primitive='Fixed two-node +4cmZ versus -2cmX/+4cmZ residual, clipped last-interior approach window; endpoints/progression retained' if progress else 'Fixed -2cmX/+4cmZ three-node triangle around requested waypoint just BEFORE observed native first failure; no amplitude/window-grid sweep',
        scope='TRAIN oracle first-stop support opportunity only; no correction/model/execution advantage. Selected acquisition is enriched,not overall-distribution/fresh evidence. Held14/15 excluded from acquisition.',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--progress',action='store_true');diagnostic(**vars(p.parse_args()))
