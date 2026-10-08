"""Registered all-mode versus known-mode data control; unchanged parent context."""
import hashlib
import json
import numpy as np
from scripts.research_v3_frequency import RUN,SUPPORT
from scripts.research_v3_analyze_frequency import paired
from scripts.run_observed_probability import read,write,sha,torch_setup


def main():
    torch=torch_setup();out=RUN/'all_mode_completion_analysis_v1';out.mkdir(exist_ok=False)
    parent,old,new='safety_mean','verified_edit_augmented','verified_edit_all_modes'
    names=[parent,old,new];metrics={};rows={};training={};configs={};rngs={};hashes={}
    for n in names:
        f=RUN/n;ev=f/'evaluation_matched_q_v1'
        metrics[n]=read(ev/'metrics.json');rows[n]=read(ev/'rows.json')
        assert metrics[n]['pool_sha256']==sha(ev/'pool.npz')
        assert metrics[n]['generator_sha256']==sha(f/'last.pt')
        for p in (ev/'metrics.json',ev/'rows.json',ev/'pool.npz'):hashes[str(p)]=sha(p)
        if n!=parent:
            training[n]=read(f/'summary.json');configs[n]=read(f/'config.json')
            assert training[n]['steps']==1200 and training[n]['last_sha256']==metrics[n]['generator_sha256']
            saved=torch.load(f/'last.pt',map_location='cpu',weights_only=False);rngs[n]=saved['loss_rng'];del saved
    for k in ('sampler','initial_sha256'):assert training[old][k]==training[new][k]
    for k in ('initial_checkpoint_sha256','grounding_support_sha256'):assert configs[old][k]==configs[new][k]
    assert configs[new]['initial_checkpoint_sha256']==sha(RUN/parent/'last.pt')
    assert {m['scorer_bundle_sha256'] for m in metrics.values()}=={'2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e'}
    support_receipt=read(RUN/'verified_edit_all_modes_support_v1/receipt.json')
    assert support_receipt['known_pool_sha256']==configs[old]['support_sha256']
    assert support_receipt['support_sha256']==configs[new]['support_sha256']
    assert configs[new]['grounding_support_sha256']==sha(SUPPORT/'support.npz')
    with np.load(SUPPORT/'support.npz') as z:known={str(i):set(z['modes'][j,z['mask'][j]]) for j,i in enumerate(z['ids']) if z['splits'][j]=='DEV_MODEL'}
    assert len(known)==288
    fixed=RUN/'strong_counterfactual_v1/rows.json';assert sha(fixed)=='515b353acd0b2d6d0d410b58fd028681fbcb4fa249a597728a53f5818b60a084'
    hashes[str(fixed)]=sha(fixed);witnesses=read(fixed)[parent]
    families=sorted({r['family'] for r in witnesses});assert len(families)==32
    kinds=['all','teacher_known','unreferenced'];denom={k:np.zeros(32) for k in kinds}
    numer={k:{n:np.zeros(32) for n in names} for k in kinds};records=[]
    maps={n:{r['id']:r for r in rows[n]} for n in names}
    for r in witnesses:
        w=set(r['surviving_modes']);j=families.index(r['family']);kk=known[r['destination_id']]
        rec={k:r[k] for k in ('family','target','origin','destination','source_id','destination_id')}
        for kind,ww in [('all',w),('teacher_known',w&kk),('unreferenced',w-kk)]:
            denom[kind][j]+=len(ww);rec[kind]=dict(opportunities=len(ww),retained={})
            for n in names:
                d=maps[n][r['destination_id']];actual={v for v,ok in zip(d['words'],d['valid']) if ok and v is not None}
                count=len(ww&actual);numer[kind][n][j]+=count;rec[kind]['retained'][n]=count
        records.append(rec)
    draws=np.random.default_rng(610093).integers(32,size=(10000,32))
    def contrast(kind,a,b):
        den=denom[kind];v=numer[kind][b]-numer[kind][a];bootden=den[draws].sum(1);assert np.all(bootden>0)
        return dict(delta=float(v.sum()/den.sum()),CI95=np.quantile(v[draws].sum(1)/bootden,[.025,.975]).tolist())
    retention={k:dict(opportunities=int(denom[k].sum()),retained={n:int(v.sum()) for n,v in numer[k].items()},
        all_minus_known=contrast(k,old,new),all_minus_parent=contrast(k,parent,new),known_minus_parent=contrast(k,parent,old)) for k in kinds}
    assert retention['all']['opportunities']==2169
    delta=paired(rows[old],rows[new]);r=retention['all']['all_minus_known']
    gate=r['delta']>=.05 and r['CI95'][0]>0 and delta['valid']['delta']>=-.01 and delta['modes8']['delta']>=-.05 and delta['rare8']['delta']>=-.01 and delta['oracle']['delta']>=-.01 and delta['brier']['delta']<=.01
    directions={}
    for a,b in [('open','closed'),('open','shifted'),('closed','open'),('shifted','open')]:
        rr=[r for r in records if r['origin']==a and r['destination']==b]
        directions[a+'_to_'+b]={k:dict(opportunities=sum(r[k]['opportunities'] for r in rr),retained={n:sum(r[k]['retained'][n] for r in rr) for n in names}) for k in kinds}
    rng_hashes={n:hashlib.sha256(json.dumps(rngs[n],sort_keys=True).encode()).hexdigest() for n in rngs}
    result=dict(metrics=metrics,training=training,configs=configs,support_receipt=support_receipt,retention=retention,
        by_direction=directions,route_contrast=delta,versus_parent={n:paired(rows[parent],rows[n]) for n in (old,new)},
        actual_initial_and_input_equal=True,group_rng_equal=rngs[old]==rngs[new],final_group_rng_sha256=rng_hashes,
        ordinary_mode_completion_gate=bool(gate),input_sha256=hashes,
        scope='One-seed ordinary verified positive mode-vocabulary completion; same M8 and original grounding, expanded mode sets change group RNG. No new algorithm or independent-test claim.')
    write(out/'retention_rows.json',records);write(out/'RESULTS.json',result)
    print(dict(retention=retention,route_contrast=delta,gate=gate),flush=True)


if __name__=='__main__':main()
