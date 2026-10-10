"""TRAIN full executor feedback on legal corrected drafts and one fixed lift.

The two teachers are separate8-route experiments, not16deployment candidates.
Actual get_path/controller/collision monitor stays unchanged.
"""
import argparse,copy,os
from pathlib import Path
import numpy as np
from research_selective_repair_v1.io import ROOT,SOURCE,RUN,read,write,sha
from research_selective_repair_v1.execute import callback
from research_selective_repair_v1.body_options import options

def collect(name,pool,family_start=0,families=2,lift=0.,preserve_row_crossings=False,targets='0'):
    from scripts import collect_observed_layout_variation as physical
    from scripts.collect_observed_layout_hash_recovery import validate_initial_geometry
    data=ROOT/'data/selective_repair_interventions_v1';registration=read(data/'registration.json')
    assert read(pool.parent/'SEAL.json')['role']=='TRAIN'
    chosen=sorted({p['family_id'] for p in registration['parent_plan'] if p['role']=='TRAIN'})[family_start:family_start+families]
    plans=[p for p in registration['parent_plan'] if p['family_id'] in chosen and p['variant'] in ('open','closed')];assert len(plans)==2*families
    target_ids=[int(v) for v in targets.split(',')];assert target_ids and set(target_ids).issubset({0,1,2})
    plans=[(p,t) for p in plans for t in target_ids]
    with np.load(pool) as z:ids=z['ids'];paths=z['paths']
    if lift:
        with np.load(pool.parent/'sealed_predictions.npz') as z:
            assert np.array_equal(ids,z['ids']);completed=z['completed']
    byid={str(v):i for i,v in enumerate(ids)};out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    source={str(f.relative_to(SOURCE)):sha(f) for d in ('scripts','configs','research_selective_repair_v1') for f in (SOURCE/d).rglob('*') if f.is_file() and f.suffix in ('.py','.json','.sh')}
    write(out/'MANIFEST.json',dict(source_commit=os.environ.get('CODE_COMMIT'),pool_sha256=sha(pool),TRAIN_families=chosen,variants=['open','closed'],target_ids=target_ids,teacher_routes_per_parent=8,lift_m=lift,shape='sin(pi*t),fixed endpoints; optional unchanged nodes around predicted row crossings',preserve_row_crossings=preserve_row_crossings,retry_budget=0,executor='Unchanged pilot.execute,23segments,max1000steps/segment,collisionaware300trialget_path',maximum_get_path_per_parent=184,TRAIN_feedback_only=True,locked_access=False))
    physical.collect_routes=callback;physical.validate_initial_geometry=validate_initial_geometry;physical.PROTOCOL='selective_body_execution_teacher_train_v1';records=[]
    for i,(original,target) in enumerate(plans):
        ident=original['parent_id']+'_target%d'%target;p=paths[byid[ident]].copy()
        if lift:
            p=options(p,completed[byid[ident]],lift)[:,2 if preserve_row_crossings else 1]
        plan=copy.deepcopy(original);plan.update(execution_id=ident,execution_target=target,execution_paths=p.tolist(),execution_selected=list(range(8)))
        file=out/('plan%d.json'%i);write(file,plan);dest=out/('parent%d'%i);r=physical.physical_worker(registration,dict(source_sha256=source),plan,file,dest)
        if r['status']=='error':raise RuntimeError('Technical TRAIN execution failed, retain: '+str(r['fatal_error']))
        result=read(dest/'EXECUTION.json');records.append(result);print(dict(id=ident,attempted=result['attempted'],success=result['success']),flush=True)
    write(out/'SUMMARY.json',dict(TRAIN_parents=len(plans),attempted=sum(r['attempted'] for r in records),success=sum(r['success'] for r in records),lift_m=lift,preserve_row_crossings=preserve_row_crossings,records=records,actual_full_arm=True,locked_access=False,scope='TRAIN feedback on all8 teacher routes, not a deployment returned4 metric; comparison with same stored seeds/ranks/initials'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--pool',required=True,type=Path);p.add_argument('--family-start',type=int,default=0);p.add_argument('--families',type=int,default=2);p.add_argument('--lift',type=float,default=0.);p.add_argument('--preserve-row-crossings',action='store_true');p.add_argument('--targets',default='0');collect(**vars(p.parse_args()))
