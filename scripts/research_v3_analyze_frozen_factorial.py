"""Complete the registered ordinary freezing x augmentation attribution."""
import numpy as np
from scripts.research_v3_analyze_edit_augmentation import main as augmentation
from scripts.research_v3_frequency import RUN
from scripts.research_v3_analyze_frequency import paired
from scripts.run_observed_probability import read,write,sha


def main():
    frozen=['frozen_encoder_plain','frozen_encoder_augmented']
    augmentation(frozen,'frozen_augmentation_analysis_v1')
    out=RUN/'frozen_factorial_analysis_v1';out.mkdir(exist_ok=False)
    names=['safety_mean','margin_mean','verified_edit_augmented']+frozen
    rows={};metrics={};hashes={};training={}
    for n in names:
        root=RUN/n;ev=root/'evaluation_matched_q_v1'
        metrics[n]=read(ev/'metrics.json');rows[n]=read(ev/'rows.json')
        assert metrics[n]['pool_sha256']==sha(ev/'pool.npz')
        for p in (ev/'rows.json',ev/'metrics.json',ev/'pool.npz'):hashes[str(p)]=sha(p)
        if n!='safety_mean':training[n]=read(root/'summary.json')
    assert len({m['scorer_bundle_sha256'] for m in metrics.values()})==1
    for n in names[2:]:
        for k in ('initial_sha256','sampler'):assert training[n][k]==training['margin_mean'][k]
    audits={n:read(RUN/n/'frozen_input_audit.json') for n in frozen}
    assert audits[frozen[0]]==audits[frozen[1]]
    for a in audits.values():assert a['initial_sha256']==a['final_sha256']
    fixed=RUN/'strong_counterfactual_v1/rows.json';hashes[str(fixed)]=sha(fixed)
    assert sha(fixed)=='515b353acd0b2d6d0d410b58fd028681fbcb4fa249a597728a53f5818b60a084'
    witnesses=read(fixed)['safety_mean'];families=sorted({r['family'] for r in witnesses});assert len(families)==32
    denom=np.zeros(32);numer={n:np.zeros(32) for n in names};maps={n:{r['id']:r for r in rows[n]} for n in names}
    records=[]
    for r in witnesses:
        i=families.index(r['family']);w=set(r['surviving_modes']);denom[i]+=len(w)
        rec={k:r[k] for k in ('family','target','origin','destination','source_id','destination_id')};rec['opportunities']=len(w)
        for n in names:
            d=maps[n][r['destination_id']];actual={v for v,ok in zip(d['words'],d['valid']) if ok and v is not None}
            count=len(w&actual);numer[n][i]+=count;rec[n]=count
        records.append(rec)
    draws=np.random.default_rng(610093).integers(32,size=(10000,32))
    def estimate(values):
        boot=values[draws].sum(1)/denom[draws].sum(1)
        return dict(delta=float(values.sum()/denom.sum()),CI95=np.quantile(boot,[.025,.975]).tolist())
    retention={n:estimate(numer[n]-numer['safety_mean']) for n in names[1:]}
    interaction=estimate((numer[frozen[1]]-numer[frozen[0]])-(numer['verified_edit_augmented']-numer['margin_mean']))
    versus_parent={n:paired(rows['safety_mean'],rows[n]) for n in names[1:]}
    d=versus_parent[frozen[0]]
    gate=d['valid']['delta']>=.02 and d['valid']['CI95'][0]>0 and d['modes8']['delta']>=.15 and d['rare8']['delta']>=-.02 and d['oracle']['delta']>=-.01 and d['brier']['delta']<=.01
    primary=read(RUN/'frozen_augmentation_analysis_v1/RESULTS.json')
    for k,v in estimate(numer[frozen[1]]-numer[frozen[0]]).items():np.testing.assert_array_equal(v,primary['fixed_source_retention'][k])
    write(out/'retention_rows.json',records)
    result=dict(metrics=metrics,training=training,frozen_audits=audits,opportunities=int(denom.sum()),
                retained={n:int(v.sum()) for n,v in numer.items()},retention_versus_parent=retention,
                augmentation_retention_interaction=interaction,route_versus_parent=versus_parent,
                freezing_route_contrast={a:paired(rows[b],rows[a]) for a,b in zip(frozen,['margin_mean','verified_edit_augmented'])},
                ordinary_plain_freezing_gate=bool(gate),ordinary_frozen_augmentation_gate=primary['ordinary_augmentation_gate'],
                input_sha256=hashes,scope='One-seed ordinary 2x2 factorial; no new mechanism or independent test claim.')
    write(out/'RESULTS.json',result)
    print(dict(retained=result['retained'],interaction=interaction,freezing_gate=gate,augmentation_gate=result['ordinary_frozen_augmentation_gate']),flush=True)


if __name__=='__main__':main()
