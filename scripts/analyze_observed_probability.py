"""Read-only synthesis of completed saved experiments, no new model selection."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.run_observed_probability import RUN,DATA,POLICY,sha,read,write


def analyze(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    families=[RUN]+sorted(RUN.glob('M8_*_scores'))
    results={};sources={}
    def checked(path):
        sources[str(path)]=sha(path)
        return read(path)
    for family in families:
        name='M4' if family==RUN else family.name
        scores={}
        for q in sorted(family.glob('q_seed*/summary.json')):
            seed=q.parent.name
            scores[seed]=checked(q)
            cal=family/('calibration_'+seed.removeprefix('q_') if hasattr(str,'removeprefix') else 'calibration_'+seed[2:])/'summary.json'
            if cal.exists():scores[seed]['calibration']=checked(cal)
        if not scores:continue
        results[name]=dict(scorers=scores)
        for pool in (family/'pools').glob('*/pool.npz'):
            sources[str(pool)]=sha(pool)
            with np.load(pool,allow_pickle=False) as a:
                compact={k:a[k] for k in ('paths','events','labels','parents','ids','pi') if k in a.files}
            dest=output/name/pool.parent.name;dest.mkdir(parents=True)
            np.savez_compressed(dest/'predictions.npz',**compact)
            for file in ('receipt.json','per_scene.json'):
                shutil.copyfile(pool.parent/file,dest/file);sources[str(pool.parent/file)]=sha(pool.parent/file)
        for cal in family.glob('calibration_seed*/predictions.npz'):
            dest=output/name/cal.parent.name;dest.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(cal,dest/cal.name);sources[str(cal)]=sha(cal)
    generators={}
    for summary in RUN.glob('M8_*/summary.json'):
        if summary.parent.name.endswith('_scores'):continue
        generators[summary.parent.name]=checked(summary)
    comparisons={}
    for suffix in ('v2','expanded'):
        ordinary=generators.get('M8_ordinary_seed0_'+suffix)
        proposed=generators.get('M8_balanced_probability_seed0_'+suffix)
        if ordinary is None or proposed is None:continue
        if ordinary['initial_sha256']!=proposed['initial_sha256'] or ordinary['draw_sha256']!=proposed['draw_sha256']:
            raise ValueError('Paired generator initialization or exposure streams differ')
        a,b=ordinary['fixed_last_metrics'],proposed['fixed_last_metrics']
        gate=b['UniqueClassifiedTipValidAtK']>a['UniqueClassifiedTipValidAtK'] and b['TipValidAtK']>=a['TipValidAtK'] and b['semantic_goal_accuracy']>=a['semantic_goal_accuracy']
        comparisons[suffix]=dict(initialization_exact=True,sampling_stream_exact=True,replication_gate_passed=gate,
            ordinary={k:a[k] for k in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','semantic_goal_accuracy')},
            balanced={k:b[k] for k in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','semantic_goal_accuracy')})
    jobs=[checked(p) for p in sorted((RUN/'jobs').glob('*/receipt.json')) if read(p)['status']!='running']
    spent=sum(j['elapsed_seconds'] for j in jobs)
    write(output/'RESULTS.json',dict(scorer_families=results,generators=generators,paired_comparisons=comparisons,
        jobs=[{k:v for k,v in j.items() if k!='source_sha256'} for j in jobs],cumulative_job_seconds=spent,
        budget_seconds=read(POLICY)['gpu_wall_budget_seconds'],cost_scope='Conservative sum of serial job outer times including CPU work and failures; nested timings not added',
        data_reservation=checked(DATA/'reservation.json'),all_evaluation_is_development=True,no_locked_reads=True))
    write(output/'SOURCE_INDEX.json',sources)
    write(output/'ARTIFACT_INDEX.json',{str(p.relative_to(output)):sha(p) for p in output.rglob('*') if p.is_file()})
    print(json.dumps(dict(output=str(output),families=list(results),comparisons=comparisons,job_seconds=spent)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();analyze(a.output)
