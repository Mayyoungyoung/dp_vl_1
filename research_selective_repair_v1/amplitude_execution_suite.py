"""Actual all8 interpolation diagnostic on two held TRAIN families,not DEV."""
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,read,write,sha
from research_selective_repair_v1.body_execution_teacher import collect
from research_selective_repair_v1.execution_semantics import audit

def summarize(rows,ids):
    selected=[r for r in rows if r['id'] in ids];assert len(selected)==32
    requests=[]
    for ident in ids:
        rr=[r for r in selected if r['id']==ident]
        words={r['successful_executable_word'] for r in rr}-{None}
        requests.append(dict(id=ident,E8=len(words),words=sorted(words),successful_clear=sum(r['success'] and r['sampled_tip_polyline_clear'] for r in rr)))
    return dict(E8=float(np.mean([r['E8'] for r in requests])),successful_clear=sum(r['successful_clear'] for r in requests),requests=requests)

def run(name,version='v2'):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);data=ROOT/'data/selective_repair_interventions_v1'
    reg=read(data/'registration.json');families=sorted({p['family_id'] for p in reg['parent_plan'] if p['role']=='TRAIN'})[6:8]
    ids=[p['parent_id']+'_target0' for p in reg['parent_plan'] if p['family_id'] in families and p['variant'] in ('open','closed')]
    assert len(ids)==4;records=[];original=None
    for label in ('identity','lift','preserved'):
        folder=RUN/('body_execution_train_%s_families2_7_v1_semantics_v1'%label)
        rows=read(folder/'ROWS.json');summary=summarize(rows,ids)
        records.append(dict(method=label,source=str(folder),reused_existing_single_episode=True,**summary))
        if label=='identity':original={(r['id'],r['slot']):r for r in rows if r['id'] in ids}
    for label in ('actual','planned','actual_nonrecurrent','binary_planned'):
        pool=RUN/('body_amplitude_%s_TRAIN_screen_seed0_%s'%(label,version))
        assert read(pool/'SEAL.json')['role']=='TRAIN'
        execution='body_amplitude_%s_TRAIN_executor_seed0_%s'%(label,version)
        collect(execution,pool/'pool.npz',family_start=6,families=2,targets='0')
        semantic=execution+'_semantics_v1';audit(semantic,execution);rows=read(RUN/semantic/'ROWS.json');summary=summarize(rows,ids)
        damaged=sum(original[(r['id'],r['slot'])]['success'] and original[(r['id'],r['slot'])]['sampled_tip_polyline_clear'] and not(r['success'] and r['sampled_tip_polyline_clear']) for r in rows)
        denominator=sum(r['success'] and r['sampled_tip_polyline_clear'] for r in original.values())
        old={r['id']:set(r['words']) for r in records[0]['requests']}
        lost=sum(len(old[r['id']]-set(r['words'])) for r in summary['requests'])
        records.append(dict(method=label,source=semantic,reused_existing_single_episode=False,prediction_sha256=sha(pool/'pool.npz'),
            observed_common_body_damage=damaged,common_original_successful_clear=denominator,observed_lost_actual_words=lost,**summary))
        write(out/'PROGRESS.json',dict(records=records,complete=False));print(records[-1],flush=True)
    write(out/'SUMMARY.json',dict(records=records,complete=True,scope='4held TRAIN requests,actual all8 teacher routes; interpolation diagnostic,not returned4/DEV/native-repeat proof',locked_access=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--version',default='v2');run(**vars(p.parse_args()))
