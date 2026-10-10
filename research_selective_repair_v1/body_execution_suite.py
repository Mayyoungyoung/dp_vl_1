"""Actual returned-four paired trials; reuse only exactly identical executor inputs."""
import argparse,hashlib,json
import numpy as np
from research_selective_repair_v1.io import ROOT,RUN,read,write,sha
from research_selective_repair_v1.execute import run
from research_selective_repair_v1.execution_semantics import audit

METHODS=['identity','lift','preserved','success','planned','actual','coordinate','actual_unprotected','actual_nonrecurrent']

def signature(plans):
    # Every controller-relevant field is included; path/selection rank order matters.
    content=[{k:p[k] for k in ('parent_id','seed','config','execution_id','execution_target','execution_paths','execution_selected')} for p in plans]
    return hashlib.sha256(json.dumps(content,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def expected(folder,data):
    registration=read(data/'registration.json');parents={p['parent_id']:p for p in registration['parent_plan']}
    families=sorted({p['family_id'] for p in parents.values() if p['role']=='DEV_MODEL'})[:2]
    ids=[p['parent_id']+'_target0' for p in parents.values() if p['family_id'] in families and p['variant'] in ('open','closed')]
    with np.load(folder/'pool.npz') as z:paths={str(i):p for i,p in zip(z['ids'],z['paths'])}
    rows={r['id']:r for r in read(folder/'rows.json')};plans=[]
    for ident in ids:
        p=dict(parents[ident.rsplit('_target',1)[0]]);p.update(execution_id=ident,execution_target=0,execution_paths=paths[ident].tolist(),execution_selected=rows[ident]['selected_indices']);plans.append(p)
    return plans

def suite(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);data=ROOT/'data/selective_repair_interventions_v1';known={};records=[]
    existing=RUN/'constraints_global_interventions_v1_executor_v2'
    if (existing/'RESULTS.json').exists():
        plans=[read(f) for f in sorted(existing.glob('plan*.json'))];known[signature(plans)]=existing.name
    for kind in METHODS:
        folder=RUN/('body_%s_DEV_screen_seed0_v1'%kind);plans=expected(folder,data);fingerprint=signature(plans)
        reused=fingerprint in known;execution=known.get(fingerprint,'body_%s_DEV_executor_seed0_v1'%kind)
        if not reused:
            run(execution,folder/'pool.npz',folder/'rows.json',data,4)
            actual=[read(f) for f in sorted((RUN/execution).glob('plan*.json'))]
            assert signature(actual)==fingerprint,'Actual executor must match sealed selected inputs'
            known[fingerprint]=execution
        semantic=execution+'_suite_semantics_v1'
        if not (RUN/semantic/'SUMMARY.json').exists():audit(semantic,execution)
        result=read(RUN/semantic/'SUMMARY.json')
        records.append(dict(method=kind,execution=execution,semantic_audit=semantic,executor_input_sha256=fingerprint,reused_exact_same_executor_inputs=reused,E4=result['mean_executable_distinct'],successful_clear_routes=result['successful_clear_routes'],attempted_routes=result['attempted_routes'],mode_changes=result['success_mode_changes'],requests=result['requests_results'],prediction_pool_sha256=sha(folder/'pool.npz')))
        write(out/'PROGRESS.json',dict(records=records,complete=False));print(records[-1],flush=True)
    write(out/'SUMMARY.json',dict(records=records,complete=True,scope='Reused two DEV families,open/closed,target0,actual frozen returned4. Initial paired screen,not fresh confirmation/three-seed evidence.',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);suite(**vars(p.parse_args()))
