"""Frozen full-feedback seed0 discrete support pilot, before formal acceptance.

Seven predeclared arms, exactly the same three measured TRAIN recipes. No
off-support XYZ or amplitude optimizer. Original complete q returns four.
"""
import argparse
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha

ARMS=[('identity','identity','crossing',None),('lift','lift','crossing',None),
      ('preserved','preserved','crossing',None),
      ('actual','actual','crossing','body_crossing_nonrecurrent_analytic_seed0_full_v1'),
      ('planned','planned','crossing','body_crossing_nonrecurrent_analytic_seed0_full_v1'),
      ('categorical','actual','crossing','body_crossing_nonrecurrent_categorical_aux_seed0_full_v1'),
      ('binary','planned','binary','body_binary_seed0_v1')]

def screens():
    from research_selective_repair_v1.body_screen import screen
    assert read(RUN/'jobs/body_crossing_full_TRAIN_study_v1/receipt.json')['status']=='completed'
    protocol=RUN/'full_body_pilot_protocol_v1';protocol.mkdir(parents=True,exist_ok=False)
    write(protocol/'PROTOCOL.json',dict(arms=ARMS,seed=0,
        DEV_subset='First two registered DEV_MODEL families,open/closed,target0; fixed before outcomes',
        scope='Full2304feedback frozen discrete3recipe diagnostic,initialC0head only,not formal matched-head acceptance',
        return_count=4,final_count=8,internal_options=24,actual_native_RNG_controlled=False,locked_access=False))
    for label,kind,forecast,folder in ARMS:
        screen('body_full_%s_DEV_screen_seed0_v1'%label,kind,RUN/folder/'last.pt' if folder else None,
            word_safe=True,forecast=forecast,all_goals=True,role='DEV_MODEL')

def execute(name):
    from research_selective_repair_v1.execute import run
    from research_selective_repair_v1.execution_semantics import audit
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);records=[]
    assert read(RUN/'jobs/body_full_DEV_screens_seed0_v1/receipt.json')['status']=='completed'
    for label,_,_,_ in ARMS:
        folder=RUN/('body_full_%s_DEV_screen_seed0_v1'%label)
        execname='body_full_%s_DEV_returned4_seed0_v1'%label
        if (RUN/execname).exists():
            previous=read(RUN/execname/'RESULTS.json')
            assert previous['pool_sha256']==sha(folder/'pool.npz') and previous['rows_sha256']==sha(folder/'ROWS.json')
            assert previous['requests']==4 and previous['requested_routes']==16
            assert all(r['attempted']==4 for r in previous['records'])
        else:
            run(execname,folder/'pool.npz',folder/'ROWS.json',data=RUN.parents[1]/'data/selective_repair_interventions_v1')
        audit(execname+'_semantics_v1',execname)
        summary=read(RUN/(execname+'_semantics_v1')/'SUMMARY.json');rows=summary['requests_results']
        assert summary['attempted_routes']==16 and summary['requests']==4
        records.append(dict(method=label,E4=summary['mean_executable_distinct'],
            successful_clear=summary['successful_clear_routes'],attempted=summary['attempted_routes'],
            source=execname+'_semantics_v1',prediction_sha256=sha(folder/'sealed_predictions.npz'),requests=rows))
        write(out/'PROGRESS.json',dict(records=records,locked_access=False))
    write(out/'SUMMARY.json',dict(records=records,scope='Actual ordered returned4,fullfeedback7armDEV pilot; initialC0head,not matched heads/fresh layouts/causal native evidence',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',default='body_full_DEV_returned4_suite_seed0_v1');p.add_argument('--screen',action='store_true');a=p.parse_args()
    screens() if a.screen else execute(a.name)
