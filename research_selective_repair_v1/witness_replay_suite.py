"""Reexecute compacted TRAIN-only actual trace witnesses under unchanged executor."""
from research_selective_repair_v1.io import RUN,read,write
from research_selective_repair_v1.execution_trace_compaction import run as compact
from research_selective_repair_v1.body_execution_teacher import collect
from research_selective_repair_v1.execution_semantics import audit

def run(name):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    pool='body_TRAIN_trace_compaction_v1';compact(pool)
    execution='body_TRAIN_compacted_trace_replay_v1';collect(execution,RUN/pool/'pool.npz',family_start=6,families=2)
    semantics=execution+'_semantics_v1';audit(semantics,execution)
    old=read(RUN/'body_execution_train_identity_families2_7_v1_semantics_v1/ROWS.json');old={(r['id'],r['slot']):r for r in old}
    rows=read(RUN/semantics/'ROWS.json');compressed=read(RUN/pool/'COMPACTION.json')
    eligible={(r['id'],r['slot']) for r in compressed['records'] if r['accepted']}
    report=read(RUN/semantics/'SUMMARY.json')
    report.update(replayed_compactions=len(eligible),replayed_successful_clear=sum(r['success'] and r['sampled_tip_polyline_clear'] for r in rows if (r['id'],r['slot']) in eligible),
        successful_original_damage=sum(old[r['id'],r['slot']]['success'] and old[r['id'],r['slot']]['sampled_tip_polyline_clear'] and not(r['success'] and r['sampled_tip_polyline_clear']) for r in rows),
        replayed_actual_words_retained=sum(old[r['id'],r['slot']]['successful_executable_word']==r['successful_executable_word'] for r in rows if (r['id'],r['slot']) in eligible),
        current_observation_only=False,scope='Oracle TRAIN witness expressiveness/causal replay diagnostic only,not learned inference or returned4/fresh result')
    write(out/'SUMMARY.json',report);print(report,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);run(**vars(p.parse_args()))
