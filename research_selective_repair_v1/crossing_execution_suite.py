"""Actual TRAIN-held returned4, never inferred by subsetting all8 traces."""
from research_selective_repair_v1.io import ROOT,RUN,read,write,sha
from research_selective_repair_v1.execute import run as execute
from research_selective_repair_v1.execution_semantics import audit

def run(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);records=[]
    arms=[('actual','body_crossing_actual_TRAIN_screen_seed0_v1'),
        ('actual_nonrecurrent','body_crossing_actual_nonrecurrent_TRAIN_screen_seed0_v1'),
        ('categorical_nonrecurrent','body_crossing_categorical_nonrecurrent_TRAIN_screen_seed0_v1'),
        ('planned','body_crossing_planned_TRAIN_screen_seed0_v1'),
        ('binary_planned','body_amplitude_binary_planned_TRAIN_screen_seed0_v2')]
    for label,folder in arms:
        pool=RUN/folder;assert read(pool/'SEAL.json')['role']=='TRAIN'
        execution='body_crossing_%s_TRAIN_returned4_executor_seed0_v1'%label
        execute(execution,pool/'pool.npz',pool/'ROWS.json',data=ROOT/'data/selective_repair_interventions_v1',role='TRAIN',family_start=6)
        semantics=execution+'_semantics_v1';audit(semantics,execution);summary=read(RUN/semantics/'SUMMARY.json')
        records.append(dict(method=label,E4=summary['mean_executable_distinct'],successful_clear=summary['successful_clear_routes'],attempted=summary['attempted_routes'],prediction_sha256=sha(pool/'pool.npz'),source=semantics,requests=summary['requests_results']))
        write(out/'PROGRESS.json',dict(records=records,complete=False));print(records[-1],flush=True)
    write(out/'SUMMARY.json',dict(records=records,complete=True,
        scope='Four held TRAIN requests,actual ordered returned4 per method; initial C0 allocation head,not matched heads/DEV/native paired causal evidence',locked_access=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
