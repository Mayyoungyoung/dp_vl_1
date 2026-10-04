"""Compare sealed pools, with family-level uncertainty and explicit negatives."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write
from scripts.paired_modes_data import RUN


def family_metrics(rows):
    groups={}
    for r in rows:
        values=dict(valid=r['K']['8']['ValidCount']/8,modes=r['K']['8']['GeometricModeCount'],
            two_distinct=r['K']['8']['TwoDistinctValid'],coverage=r['K']['8']['ReferenceModeCoverage'],
            q_top1=r['K']['1']['ValidCount'],goal_failure=np.mean([not c['semantic_goal_correct'] for c in r['candidates']]),
            collision=np.mean([not c['tip_segments_clear'] for c in r['candidates']]))
        groups.setdefault(r['parent'],[]).append(values)
    return {p:{k:float(np.mean([v[k] for v in items if v[k] is not None])) for k in items[0]} for p,items in groups.items()}


def comparison(a,b):
    assert set(a)==set(b);parents=sorted(a);rng=np.random.default_rng(20261005)
    indices=rng.integers(len(parents),size=(10000,len(parents)))
    return {k:dict(delta=float(np.mean(d)),CI95=np.percentile(d[indices].mean(1),[2.5,97.5]).tolist())
        for k in next(iter(a.values())) for d in [np.array([b[p][k]-a[p][k] for p in parents])]}


def paired_family_metrics(details):
    keys=('shared_recall','opened_recall','shifted_shared_recall','closed_adaptation_gain','closed_low_center_attempt_rate')
    groups={}
    for row in details:groups.setdefault(row['family'],[]).append(row)
    return {p:{k:float(np.mean([r[k] for r in rows if r[k] is not None])) for k in keys} for p,rows in groups.items()}


def analyze(arms,seeds,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    results={};families={};allrows={};paired={};paired_families={}
    for split in ('paired_dev','old_dev','dev32'):
        for seed in seeds:
            for arm in arms:
                folder=RUN/'evaluation'/split/('%s_seed%d'%(arm,seed))
                if not (folder/'RESULTS.json').exists():continue
                key='%s/%s/seed%d'%(split,arm,seed);summary=read(folder/'RESULTS.json');rows=read(folder/'rows.json')
                families[key]=family_metrics(rows);allrows[key]=rows
                results[key]={k:float(np.mean([r[k] for r in families[key].values()])) for k in next(iter(families[key].values()))}
                if split=='paired_dev':
                    paired[key]=summary['paired']['metrics']
                    paired_families[key]=paired_family_metrics(summary['paired']['details'])
    comparisons={}
    for split in ('paired_dev','old_dev','dev32'):
        for a,b in zip(arms[:-1],arms[1:]):
            for seed in seeds:
                ka,kb=['%s/%s/seed%d'%(split,x,seed) for x in (a,b)]
                if ka in families and kb in families:comparisons[kb+' minus '+ka]=comparison(families[ka],families[kb])
                if ka in paired_families and kb in paired_families:
                    comparisons[kb+' minus '+ka+'/paired_response']=comparison(paired_families[ka],paired_families[kb])
            # Average generator seeds within family, then family bootstrap.
            if all('%s/%s/seed%d'%(split,x,s) in families for x in (a,b) for s in seeds):
                means=[]
                for arm in (a,b):
                    first=families['%s/%s/seed%d'%(split,arm,seeds[0])]
                    means.append({p:{k:float(np.mean([families['%s/%s/seed%d'%(split,arm,s)][p][k] for s in seeds])) for k in first[p]} for p in first})
                comparisons[split+'/'+b+' minus '+a+'/mean_seeds']=comparison(*means)
                if split=='paired_dev':
                    means=[]
                    for arm in (a,b):
                        first=paired_families['%s/%s/seed%d'%(split,arm,seeds[0])]
                        means.append({p:{k:float(np.mean([paired_families['%s/%s/seed%d'%(split,arm,s)][p][k] for s in seeds])) for k in first[p]} for p in first})
                    comparisons[split+'/'+b+' minus '+a+'/mean_seeds/paired_response']=comparison(*means)
    gate=None
    if all('paired_dev/'+a+'/seed0' in results for a in ('R0','R1','R2')):
        a,b,c=[results['paired_dev/'+a+'/seed0'] for a in ('R0','R1','R2')]
        p1,p2=[paired['paired_dev/'+a+'/seed0'] for a in ('R1','R2')]
        conditions=dict(modes_above_R0=c['modes']>a['modes'],modes_above_R1=c['modes']>b['modes'],
            shared_recall_above_R1=p2['shared_recall']>p1['shared_recall'],validity_loss_at_most_02=c['valid']>=b['valid']-.02)
        gate=dict(passed=all(conditions.values()),conditions=conditions)
    negatives={}
    for key,rows in allrows.items():
        negatives[key]=dict(zero_valid=[r['id'] for r in rows if r['K']['8']['ValidCount']==0],
            all_goal_failed=[r['id'] for r in rows if all(not c['semantic_goal_correct'] for c in r['candidates'])],
            at_most_one_mode=[r['id'] for r in rows if r['K']['8']['GeometricModeCount']<=1])
    # Exposures must match across arms for each seed.
    stream={}
    for seed in seeds:
        states={arm:read(RUN/('%s_seed%d'%(arm,seed))/'summary.json') for arm in arms if (RUN/('%s_seed%d'%(arm,seed))/'summary.json').exists()}
        if not states:continue
        streams={r['sampler']['index_chain_sha256'] for r in states.values()};initial={r['initial_sha256'] for r in states.values()}
        assert len(streams)==len(initial)==1,'Unequal training exposure/initialization'
        stream[str(seed)]=dict(arms=list(states),sampler_sha256=next(iter(streams)),initial_sha256=next(iter(initial)))
    write(output/'RESULTS.json',dict(results=results,paired=paired,comparisons=comparisons,seed0_gate=gate,negatives=negatives,stream_identity=stream,
        uncertainty_scope='Family bootstrap conditional on the trained seeds; not a confidence interval over all random initializations.'))
    lines=['# Paired route-set comparison','','All numbers use sealed, unrepaired M8 outputs. DEV is development evidence.','',
           '| split / arm / seed | valid % | modes | reference coverage % | goal fail % | collision % | frozen q top1 % |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for key,r in results.items():lines.append('|%s|%.2f|%.3f|%.2f|%.2f|%.2f|%.2f|'%(key,100*r['valid'],r['modes'],100*r['coverage'],100*r['goal_failure'],100*r['collision'],100*r['q_top1']))
    lines.extend(['','Seed0 continuation gate: `'+str(gate)+'`.','','Family bootstrap results, all failed requests and input-stream identities are saved in RESULTS.json.'])
    (output/'TABLE.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(__import__('json').dumps(dict(results=results,paired=paired,seed0_gate=gate)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['R0','R1','R2']);p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--output',required=True)
    a=p.parse_args();analyze(a.arms,a.seeds,a.output)
