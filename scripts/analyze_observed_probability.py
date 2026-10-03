"""Read-only synthesis of completed saved experiments, no new model selection."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scripts.run_observed_probability import RUN,DATA,POLICY,sha,read,write


def selection_table(pool,logits,temperature,scene_rows):
    from routeset.observed_probability import select_route_indices
    import torch
    torch.set_num_threads(1)
    q=1/(1+np.exp(-np.clip(logits/temperature,-60,60)))
    paths=pool['paths'];labels=pool['labels'];n,m=labels.shape
    rowmap={r['id']:r for r in scene_rows};results=[]
    for k in (1,2,4):
        if k>m:continue
        qs=select_route_indices(torch.tensor(paths),torch.tensor(q),k).numpy()
        methods={'q_diverse':qs,'first':np.tile(np.arange(k),(n,1)),
                 'shortest':np.argsort(np.linalg.norm(np.diff(paths,axis=2),axis=-1).sum(2),axis=1,kind='stable')[:,:k]}
        if 'pi' in pool:methods['pi']=np.argsort(-pool['pi'],axis=1,kind='stable')[:,:k]
        for method,indices in methods.items():
            valid=labels[np.arange(n)[:,None],indices];unique=[];unknown=[]
            for i,identifier in enumerate(pool['ids']):
                chosen=[rowmap[str(identifier)]['candidates'][int(j)] for j in indices[i]]
                modes={tuple(c['declared_passage_type']) for c in chosen if c['TipValid'] and c['declared_passage_type'] is not None}
                unique.append(len(modes));unknown.append(sum(c['TipValid'] and c['declared_passage_type'] is None for c in chosen))
            results.append(dict(method=method,K=k,internal_M=m,valid=float(valid.mean()),any_valid=float(valid.max(1).mean()),
                                unique_classified_valid=float(np.mean(unique)),unknown_valid=float(np.mean(unknown))))
    return results


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
            cal=family/('calibration_'+seed[2:])/'summary.json'
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
        if 'q_seed0' in scores and 'calibration' in scores['q_seed0']:
            with np.load(family/'pools/DEV_MODEL/pool.npz',allow_pickle=False) as a:pool={k:a[k].copy() for k in ('paths','labels','ids','pi') if k in a.files}
            with np.load(family/'calibration_seed0/predictions.npz',allow_pickle=False) as a:logits=a['dev_logits'].copy()
            results[name]['K_selection']=selection_table(pool,logits,scores['q_seed0']['calibration']['temperature'],checked(family/'pools/DEV_MODEL/per_scene.json'))
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
        replicates=[]
        for seed in (0,1,2):
            oa=generators.get('M8_ordinary_seed%d_%s'%(seed,suffix))
            pb=generators.get('M8_balanced_probability_seed%d_%s'%(seed,suffix))
            if oa is None or pb is None:continue
            if oa['initial_sha256']!=pb['initial_sha256'] or oa['draw_sha256']!=pb['draw_sha256']:
                raise ValueError('Replication pair initialization or sampling differs')
            replicates.append(dict(seed=seed,ordinary={k:oa['fixed_last_metrics'][k] for k in comparisons[suffix]['ordinary']},
                                   balanced={k:pb['fixed_last_metrics'][k] for k in comparisons[suffix]['ordinary']}))
            for name in ('M8_ordinary_seed%d_%s'%(seed,suffix),'M8_balanced_probability_seed%d_%s'%(seed,suffix)):
                dest=output/'generator_predictions'/name;dest.mkdir(parents=True,exist_ok=True)
                for file in ('predictions.npz','per_scene.json'):
                    path=RUN/name/'dev/step03000'/file
                    shutil.copyfile(path,dest/file);sources[str(path)]=sha(path)
        comparisons[suffix]['replicates']=replicates
        comparisons[suffix]['mean']={arm:{k:float(np.mean([r[arm][k] for r in replicates]))
            for k in comparisons[suffix]['ordinary']} for arm in ('ordinary','balanced')}
        comparisons[suffix]['sample_std']={arm:{k:float(np.std([r[arm][k] for r in replicates],ddof=1)) if len(replicates)>1 else None
            for k in comparisons[suffix]['ordinary']} for arm in ('ordinary','balanced')}
        comparisons[suffix]['seed_scope']='Different extra-query initialization and training RNG, shared pretrained95-parent generator; not independent backbone pretraining'
        parent_delta={}
        for replicate in replicates:
            seed=replicate['seed']
            paired=[]
            for arm in ('ordinary','balanced_probability'):
                rows=read(RUN/('M8_%s_seed%d_%s'%(arm,seed,suffix))/'dev/step03000/per_scene.json')
                paired.append({row['scene_id']:row for row in rows})
            if set(paired[0])!=set(paired[1]):raise ValueError('Paired development scenes differ')
            for scene in paired[0]:
                a,b=paired[0][scene],paired[1][scene]
                if a['parent_id']!=b['parent_id']:raise ValueError('Parent identity differs')
                parent_delta.setdefault(a['parent_id'],[]).append({k:b['tip_evaluation'][k]-a['tip_evaluation'][k]
                    for k in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK')})
        rng=np.random.default_rng(78123)
        boot_indices=rng.integers(len(parent_delta),size=(2000,len(parent_delta)))
        comparisons[suffix]['paired_parent_bootstrap95']={}
        for metric in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK'):
            delta=np.array([np.mean([r[metric] for r in rows]) for rows in parent_delta.values()])
            comparisons[suffix]['paired_parent_bootstrap95'][metric]=dict(mean_difference=float(delta.mean()),
                interval=np.quantile(delta[boot_indices].mean(1),[.025,.975]).tolist(),parents=len(delta))
        comparisons[suffix]['bootstrap_scope']='Resample parents, average target conditions and observed seeds within each parent; conditional on these seeds and reused DEV, no selection correction or final-test claim'
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
