"""TRAIN witness compaction diagnostic, never an oracle deployed route repair."""
import argparse,time
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,SOURCE,read,write,sha
from research_selective_repair_v1.execution_semantics import word,tip_clear

def compact(trace,cfg,start,goal,nodes=24):
    # Include both exact first-cross points. Earlier trace prefixes cannot cross
    # their respective planes before these mandatory anchors.
    points=[np.asarray(trace[0])];mandatory=[0];seen=set()
    for a,b in zip(trace[:-1],trace[1:]):
        events=[]
        for row,x in enumerate(cfg['row_x']):
            if row not in seen and a[0]<x<=b[0]:
                t=(x-a[0])/(b[0]-a[0]);events.append((t,row,a+t*(b-a)));seen.add(row)
        for _,_,p in sorted(events,key=lambda e:e[0]):points.append(p);mandatory.append(len(points)-1)
        points.append(b)
    points=np.asarray(points);points[0]=start;points[-1]=goal;selected=sorted(set(mandatory+[len(points)-1]))
    while len(selected)<min(nodes,len(points)):
        candidates=[]
        for a,b in zip(selected[:-1],selected[1:]):
            if b==a+1:continue
            segment=points[a:b+1];delta=points[b]-points[a]
            t=np.clip(((segment-points[a])*delta).sum(-1)/max(np.square(delta).sum(),1e-15),0,1)
            distance=np.linalg.norm(segment-points[a]-t[:,None]*delta,axis=-1);j=int(distance[1:-1].argmax())+1
            invalid=not tip_clear(points[[a,b]],cfg)
            candidates.append((float(distance[j])+(100. if invalid else 0.),a+j))
        if not candidates:break
        selected.append(max(candidates)[1]);selected.sort()
    path=points[selected]
    if len(path)<nodes:path=np.concatenate([path,np.repeat(path[-1:],nodes-len(path),axis=0)])
    assert path.shape==(nodes,3)
    return path,dict(actual_word=word(trace,cfg),compact_word=word(path,cfg),tip_clear=tip_clear(path,cfg),native_trace_points=len(trace),compaction_trace_points=len(points),mandatory_firstcross_anchors=len(mandatory)-1,
        compression_max_edge_distance_m=float(max(np.linalg.norm(points[a:b+1]-points[a],axis=-1).max() for a,b in zip(selected[:-1],selected[1:]))))

def run(name):
    tic=time.monotonic();out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    source=RUN/'constraints_global_interventions_TRAIN_body_v1'
    assert read(source/'SEAL.json')['role']=='TRAIN'
    with np.load(source/'pool.npz') as z:pool={k:z[k] for k in z.files}
    ids={str(v):i for i,v in enumerate(pool['ids'])};teacher=RUN/'body_execution_train_identity_families2_7_v1'
    labels={(r['id'],r['slot']):r for r in read(RUN/(teacher.name+'_semantics_v1')/'ROWS.json')}
    records=[];hashes={str(source/'pool.npz'):sha(source/'pool.npz')}
    for file in sorted(teacher.glob('plan*.json')):
        plan=read(file);ident=plan['execution_id']
        if not ident.startswith(('sr_family_642006_','sr_family_642007_')):continue
        assert plan['role']=='TRAIN';parent=teacher/('parent'+file.stem[4:]);result=read(parent/'EXECUTION.json')
        for record in result['records']:
            label=labels[ident,record['slot']]
            if label['successful_executable_word'] is None:continue
            trace=parent/plan['parent_id']/record['trace']['file'];assert sha(trace)==record['trace']['sha256'];hashes[str(trace)]=sha(trace)
            with np.load(trace) as z:tip=z['gripper_pose'][:,:3]
            ix=ids[ident];slot=record['slot'];old=pool['paths'][ix,slot].copy()
            path,detail=compact(tip,plan['config'],old[0],old[-1])
            accepted=detail['tip_clear'] and detail['compact_word']==label['successful_executable_word']
            if accepted:pool['paths'][ix,slot]=path.astype(pool['paths'].dtype)
            records.append(dict(id=ident,slot=slot,accepted=accepted,actual_teacher_word=label['successful_executable_word'],original_requested_word=word(old,plan['config']),maximum_edit_m=float(np.linalg.norm(path-old,axis=-1).max()),**detail))
    assert len(records)==19
    np.savez_compressed(out/'pool.npz',**pool)
    write(out/'SEAL.json',dict(role='TRAIN',prediction_sha256=sha(out/'pool.npz'),source_commit=SOURCE.name,current_observation_only=False,
        scope='TRAIN actual executed trace compaction upper-level representational replay diagnostic; NOT a deployable model, oracle training geometry only',locked_access=False))
    write(out/'COMPACTION.json',dict(records=records,successful_teacher_routes=19,geometry_mode_preserving_compactions=sum(r['accepted'] for r in records),input_sha256=hashes,seconds=time.monotonic()-tic,source_commit=SOURCE.name,scope='Two held TRAIN families, identity teachers only; not learned method/fresh metric',locked_access=False))
    print(read(out/'COMPACTION.json'),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
