"""Zero-edit TRAIN native-repeat diagnostic, not a method evaluation."""
import numpy as np
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_execution_teacher import collect
from research_selective_repair_v1.execution_semantics import audit
from research_selective_repair_v1.amplitude_execution_suite import summarize

def run(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    pool=RUN/'constraints_global_interventions_TRAIN_body_v1'/'pool.npz'
    original_source=RUN/'body_execution_train_identity_families2_7_v1'
    original_rows=read(RUN/(original_source.name+'_semantics_v1')/'ROWS.json')
    ids=sorted({r['id'] for r in original_rows if r['id'].startswith(('sr_family_642006_','sr_family_642007_'))})
    assert len(ids)==4
    old={(r['id'],r['slot']):r for r in original_rows if r['id'] in ids}
    old_plans={read(f)['execution_id']:read(f) for f in original_source.glob('plan*.json')}
    records=[]
    for repeat in range(2):
        execution=name+'_repeat%d'%repeat
        collect(execution,pool,family_start=6,families=2,targets='0')
        semantics=execution+'_semantics_v1';audit(semantics,execution)
        rows=read(RUN/semantics/'ROWS.json')
        for f in (RUN/execution).glob('plan*.json'):
            plan=read(f);baseline=old_plans[plan['execution_id']]
            assert plan['seed']==baseline['seed'] and plan['config']==baseline['config']
            np.testing.assert_array_equal(plan['execution_paths'],baseline['execution_paths'])
        report=summarize(rows,ids)
        report.update(repeat=repeat,source=semantics,pool_sha256=sha(pool),
            unchanged_routes=32,common_original_successful_clear=sum(r['success'] and r['sampled_tip_polyline_clear'] for r in old.values()),
            observed_zero_edit_damage=sum(old[r['id'],r['slot']]['success'] and old[r['id'],r['slot']]['sampled_tip_polyline_clear'] and not(r['success'] and r['sampled_tip_polyline_clear']) for r in rows),
            actual_word_disagreements=sum(old[r['id'],r['slot']]['actual_word']!=r['actual_word'] for r in rows))
        records.append(report);write(out/'PROGRESS.json',dict(records=records,complete=False));print(report,flush=True)
    write(out/'SUMMARY.json',dict(records=records,complete=True,baseline=summarize(original_rows,ids),
        scope='Two zero-edit TRAIN all8 repeats; same Python/numpy seeds, starts/config/requested curves; native planner/physics randomness not assumed controlled',locked_access=False))

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
