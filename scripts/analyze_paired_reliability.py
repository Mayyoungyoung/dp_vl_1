"""Parent-clustered scoring comparisons from completed calibrated pools."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write,sha
from scripts.analyze_paired_modes import comparison


def main(root,arms,seeds,output):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    results={};intervals={};missing=[];families={}
    for arm in arms:
        for seed in seeds:
            folder=root/'reliability'/('%s_seed%d'%(arm,seed))
            cal=folder/('calibration_seed%d'%seed)
            if not (cal/'summary.json').exists():missing.append('%s/seed%d'%(arm,seed));continue
            summary=read(cal/'summary.json');train=read(folder/('q_seed%d'%seed)/'summary.json')
            prevalence=train['train_prevalence']
            for split in ('paired_dev','old_dev','dev32'):
                key='%s/%s/seed%d'%(split,arm,seed);pool=root/'evaluation'/split/('%s_seed%d'%(arm,seed))/'pool.npz'
                assert sha(pool)==summary['results'][split]['pool_sha256']
                with np.load(pool) as a:labels=a['labels'].astype(float);frozen=a['q'].astype(float);parents=a['parents'];ids=a['ids']
                with np.load(cal/(split+'.npz')) as a:
                    assert np.array_equal(ids,a['ids']) and np.array_equal(labels,a['labels'])
                    q=a['q'].astype(float)
                if split=='paired_dev':parents=np.array([p.rsplit('_',1)[0] for p in parents])
                records={r['id']:r for r in read(pool.parent/'rows.json')}
                confident=q>=.8
                mode_counts=[len({w for j,w in enumerate(records[str(identifier)]['words']) if confident[i,j] and w is not None}) for i,identifier in enumerate(ids)]
                def parent_metrics(prob):
                    prob=np.clip(prob,1e-7,1-1e-7);pick=prob.argmax(1);idx=np.arange(len(labels))
                    values=dict(brier=((prob-labels)**2).mean(1),
                        nll=-(labels*np.log(prob)+(1-labels)*np.log1p(-prob)).mean(1),
                        selected_valid=labels[idx,pick],selected_brier=(prob[idx,pick]-labels[idx,pick])**2)
                    return {p:{k:float(v[parents==p].mean()) for k,v in values.items()} for p in sorted(set(parents))}
                families[key]=(parent_metrics(frozen),parent_metrics(q))
                intervals[key]=comparison(*families[key])
                calibrated=summary['results'][split]['calibrated']
                results[key]=dict(temperature=summary['temperature'],scorer_best_step=train['best_step'],
                    constant_train_prevalence_brier=float(((labels-prevalence)**2).mean()),
                    requests=len(labels),independent_families=len(set(parents)),
                    multi_route_08=dict(mean_reported_count=float(confident.sum(1).mean()),
                        mean_valid_distinct_modes=float(np.mean(mode_counts)),requests_with_two_valid_modes=float(np.mean(np.array(mode_counts)>=2)),
                        reported_candidate_validity=float(labels[confident].mean()) if confident.any() else None),
                    **{k:calibrated[k] for k in ('selected_valid','brier','nll','reliability_bins_ece','selected_reliability_bins_ece','selected_aurc')})
    aggregates={}
    for arm in arms:
        for split in ('paired_dev','old_dev','dev32'):
            keys=['%s/%s/seed%d'%(split,arm,seed) for seed in seeds]
            if not all(k in results for k in keys):continue
            metrics=('selected_valid','brier','nll','reliability_bins_ece','selected_reliability_bins_ece','selected_aurc')
            aggregates[split+'/'+arm]=dict(seeds=seeds,metrics={m:dict(mean=float(np.mean([results[k][m] for k in keys])),
                sd=float(np.std([results[k][m] for k in keys],ddof=1)) if len(keys)>1 else 0.) for m in metrics})
            means=[]
            for side in (0,1):
                first=families[keys[0]][side]
                means.append({p:{m:float(np.mean([families[k][side][p][m] for k in keys])) for m in first[p]} for p in first})
            intervals[split+'/'+arm+'/mean_seeds']=comparison(*means)
    write(output/'RESULTS.json',dict(results=results,seed_aggregates=aggregates,calibrated_minus_frozen_family_CI=intervals,missing=missing,
        scope='Fixed generator/scorer/calibration fits; cluster targets and variants by family. These intervals exclude training and temperature-fit uncertainty.'))
    lines=['# Calibrated actual-route scoring','','| split / arm / generator seed | selected valid % | Brier | constant Brier | NLL | ECE % | selected ECE % |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for key,r in results.items():lines.append('|%s|%.2f|%.4f|%.4f|%.4f|%.2f|%.2f|'%(key,100*r['selected_valid'],r['brier'],r['constant_train_prevalence_brier'],r['nll'],100*r['reliability_bins_ece'],100*r['selected_reliability_bins_ece']))
    (output/'TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--arms',nargs='+',default=['R0','R2']);p.add_argument('--seeds',nargs='+',type=int,default=[0])
    a=p.parse_args();main(a.root,a.arms,a.seeds,a.output)
