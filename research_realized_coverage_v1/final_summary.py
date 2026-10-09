"""Combine fixed-witness outcomes with all five registered head seeds."""
import json
import numpy as np
from research_realized_coverage_v1.core import RUN,read,write

def main():
    formal=read(RUN/'FIVE_SEED_FINAL.json')
    witness=read(RUN/'fixed_witness_analysis/RESULTS.json')
    models=witness['models'];rows={};counts={};families=sorted(models['parent']['families'])
    assert len(families)==32
    for arm in ('success','dense'):
        rows[arm]=[];counts[arm]=[]
        for row in formal['seed_results'][arm]:
            d=models[row['model']+':adaptive']
            vals=[d['retained'],d['lost_of_1872'],d['recovered_of_297'],d['adaptation_counts'].get('valid_same_mode_repair',0),d['strict_adaptation']['repairs']]
            counts[arm].append(vals)
            row['fixed_witness_counts']=dict(zip(['retained_of2169','lost_of1872','recovered_of297','adapted_of270','strict_adapted_of255'],vals))
            rows[arm].append([[d['families'][f]['retained'],d['families'][f]['adapted']] for f in families])
        rows[arm]=np.array(rows[arm]);counts[arm]=np.array(counts[arm])
    den=np.array([[models['parent']['families'][f]['denom'],models['parent']['families'][f]['adaptation_denom']] for f in families])
    diff=rows['dense']-rows['success'];rng=np.random.default_rng(910097)
    samples=[]
    for _ in range(10000):
        s=rng.integers(5,size=5);f=rng.integers(32,size=32)
        samples.append(diff[s][:,f].sum(1).mean(0)/den[f].sum(0))
    formal['fixed_witness']=dict(denominators=[2169,1872,297,270,255],
        mean_counts={k:v.mean(0).tolist() for k,v in counts.items()},
        dense_minus_success_fraction=diff.sum(1).mean(0).tolist(),
        interval_metrics=['retention_fraction','adaptation_fraction'],seed_family95=np.quantile(samples,[.025,.975],axis=0).T.tolist())
    formal['fixed_witness']['dense_minus_success_fraction']=(diff.sum(1).mean(0)/den.sum(0)).tolist()
    all_evaluations={}
    for p in sorted(RUN.glob('*/eval_adaptive/metrics.json')):
        d=read(p);all_evaluations[p.parent.parent.name]={k:d[k] for k in ('raw','selected','condition_hit','cost')}
    result=dict(formal=formal,fixed_models=models,all_evaluations=all_evaluations,
        reference_diagnostic=read(RUN/'REFERENCE_DIAGNOSTIC.json'),reference_words=read(RUN/'REFERENCE_WORD_DIAGNOSTIC.json'),
        scientific_acceptance=False,independent_test=False,default_changed=False)
    write(RUN/'RESULTS.json',result)
    lines=['# Complete measured tables','',
        'All rows use 288 reused DEV requests. Percentages below describe end-effector routes.',
        'Five seeds train allocation heads on fixed C0, not five new geometry models.','',
        '|Model|U8|V8 %|Recall %|U4|V4 %|Retained /2169|Lost /1872|Recovered /297|Repair /270|Strict /255|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,d in models.items():
        lines.append('|%s|%.4f|%.3f|%.3f|%.4f|%.3f|%d|%d|%d|%d|%d|'%(name,d['raw']['distinct'],100*d['raw']['valid_fraction'],100*d['raw']['recall'],d['selected']['distinct'],100*d['selected']['valid_fraction'],d['retained'],d['lost_of_1872'],d['recovered_of_297'],d['adaptation_counts'].get('valid_same_mode_repair',0),d['strict_adaptation']['repairs']))
    lines+=['','## Every measured training curve / proposal evaluation','',
        '|Evaluation|U8|V8 %|Recall %|U4|V4 %|','|---|---:|---:|---:|---:|---:|']
    for name,d in all_evaluations.items():
        lines.append('|%s|%.4f|%.3f|%.3f|%.4f|%.3f|'%(name,d['raw']['distinct'],100*d['raw']['valid_fraction'],100*d['raw']['recall'],d['selected']['distinct'],100*d['selected']['valid_fraction']))
    (RUN/'RESULTS_TABLES.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(formal['fixed_witness']),flush=True)

if __name__=='__main__':main()
