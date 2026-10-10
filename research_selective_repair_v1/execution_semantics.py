"""Independent post-execution word audit of the actual tip traces, not requests."""
import argparse
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha
from routeset.research_v3_modes import passage_signature
from routeset.geometry import segment_aabb_intersection

def word(path,cfg):
    result=passage_signature(path,cfg);return '|'.join(result) if result else None

def tip_clear(path,cfg):
    centers=[];halves=[]
    for x,ys,height in zip(cfg['row_x'],cfg['post_y'],cfg['post_heights']):
        for y in ys:centers.append([x,y,cfg['post_base_z']+height/2]);halves.append([.0175,.0175,height/2])
    c=np.asarray(centers);h=np.asarray(halves);margin=cfg['tip_clearance_m']
    if len(path)<2:return False
    a,b,lower,upper=np.broadcast_arrays(path[:-1,None],path[1:,None],c[None]-h[None]-margin,c[None]+h[None]+margin)
    hit=segment_aabb_intersection(a,b,lower,upper)
    return bool(not hit.any() and (path[:,2]>=cfg['post_base_z']+.02).all())

def audit(name,source):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);source=RUN/source
    assert (source/'SUMMARY.json').exists() or (source/'RESULTS.json').exists(),'Require completed whole-source receipt'
    rows=[];requests=[]
    for file in sorted(source.glob('plan*.json')):
        index=int(file.stem[4:]);plan=read(file);child=source/('parent%d'%index)
        if not child.exists():child=source/('request%d'%index)
        result=read(child/'EXECUTION.json');cfg=plan['config'];current=[]
        for record in result['records']:
            trace=child/plan['parent_id']/record['trace']['file'];assert sha(trace)==record['trace']['sha256']
            with np.load(trace) as z:actual=z['gripper_pose'][:,:3]
            planned=np.asarray(plan['execution_paths'])[record['slot']]
            w=word(actual,cfg);pw=word(planned,cfg);clear=tip_clear(actual,cfg)
            row=dict(id=plan['execution_id'],slot=record['slot'],rank=record['rank'],success=record['success'],sampled_tip_polyline_clear=clear,planned_word=pw,actual_word=w,successful_executable_word=w if record['success'] and clear else None,mode_changed=bool(record['success'] and w!=pw),trace_sha256=sha(trace),scope='Actual sampled-step tip polyline/operational word; full-arm safety is monitored at simulator steps, not a continuous certificate')
            rows.append(row);current.append(row)
        words={r['successful_executable_word'] for r in current}-{None};requests.append(dict(id=plan['execution_id'],successful_routes=sum(r['success'] for r in current),successful_clear_routes=sum(r['success'] and r['sampled_tip_polyline_clear'] for r in current),executable_distinct_words=len(words),actual_words=sorted(words),mode_changed=sum(r['mode_changed'] for r in current)))
    report=dict(requests=len(requests),attempted_routes=len(rows),successful_routes=sum(r['success'] for r in rows),successful_clear_routes=sum(r['success'] and r['sampled_tip_polyline_clear'] for r in rows),mean_executable_distinct=float(np.mean([r['executable_distinct_words'] for r in requests])),success_mode_changes=sum(r['mode_changed'] for r in rows),requests_results=requests,source=str(source),truth_access='Post-execution geometry/trace audit only',locked_access=False)
    write(out/'ROWS.json',rows);write(out/'SUMMARY.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--source',required=True);audit(**vars(p.parse_args()))
