"""All three caller predicates, paired across full families and continuations."""
import argparse,json
from pathlib import Path
import numpy as np
from research_route_portfolio_v1.statistics import read


def main(root,output):
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    rows=[read(root/f'preference_C_seed{s}'/'rows.json') for s in range(3)]
    names=['ground_only','over_any','first_gap0'];families=sorted({r['family'] for r in rows[0]})
    assert len(families)==16 and all(len(rr)==1008 for rr in rows)
    assert all([(r['id'],r['preference']) for r in rr]==[(r['id'],r['preference']) for r in rows[0]] for rr in rows)
    rng=np.random.default_rng(20261011);sd=rng.integers(3,size=(10000,3));fd=rng.integers(16,size=(10000,16));report={}
    metrics=['distinct','compliant_valid_fraction','any_compliant_valid']
    for pref in names:
        arrays={}
        for arm in ['proposed8','proposed4','baseline8','baseline4']:
            arrays[arm]=np.array([[[np.mean([r[arm][m] for r in rr if r['preference']==pref and r['family']==f]) for m in metrics] for f in families] for rr in rows])
        result=dict(means={arm:dict(zip(metrics,a.mean((0,1)).tolist())) for arm,a in arrays.items()},comparisons={})
        for budget in (8,4):
            delta=arrays[f'proposed{budget}']-arrays[f'baseline{budget}']
            boot=delta[sd[:,:,None],fd[:,None,:]].mean((1,2))
            result['comparisons'][str(budget)]=dict(mean=dict(zip(metrics,delta.mean((0,1)).tolist())),
                crossed_CI95=dict(zip(metrics,np.quantile(boot,[.025,.975],axis=0).T.tolist())))
        report[pref]=result
    payload=dict(preferences=report,metrics=metrics,requests_per_preference=336,generator_continuations=3,
                 inference_budget=8,returned_budget=4,locked_access=False,robot_execution=False,
                 scope='Three predicates fixed before evaluation; reusedDEV16; conditional crossed family/continuation intervals; raw per-predicate95 intervals without multiple-testing correction')
    (out/'RESULTS.json').write_text(json.dumps(payload,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    table=['|Preference|Baseline valid modes@8|Requested valid modes@8|Baseline compliant validity@4|Requested compliant validity@4|',
           '|---|---:|---:|---:|---:|']
    for pref,data in report.items():
        m=data['means'];table.append(f"|{pref}|{m['baseline8']['distinct']:.3f}|{m['proposed8']['distinct']:.3f}|{100*m['baseline4']['compliant_valid_fraction']:.2f}%|{100*m['proposed4']['compliant_valid_fraction']:.2f}%|")
    (out/'TABLE.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    main(**vars(p.parse_args()))
