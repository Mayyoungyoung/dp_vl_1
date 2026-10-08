"""Fixed-source retention versus a sealed same-update ordinary control."""
import numpy as np
from scripts.research_v3_frequency import RUN
from scripts.research_v3_analyze_frequency import paired
from scripts.run_observed_probability import read,write,sha,torch_setup


def main(names=None,output='verified_edit_analysis_v1'):
    torch=torch_setup();out=RUN/output;out.mkdir(exist_ok=False)
    names=names or ['margin_mean','verified_edit_augmented'];training={};metrics={};rows={};configs={};rngs={}
    for name in names:
        root=RUN/name;ev=root/'evaluation_matched_q_v1'
        training[name]=read(root/'summary.json');configs[name]=read(root/'config.json')
        metrics[name]=read(ev/'metrics.json');rows[name]=read(ev/'rows.json')
        assert metrics[name]['generator_sha256']==training[name]['last_sha256']==sha(root/'last.pt')
        assert metrics[name]['pool_sha256']==sha(ev/'pool.npz') and training[name]['steps']==1200
        saved=torch.load(root/'last.pt',map_location='cpu',weights_only=False);rngs[name]=saved['loss_rng'];del saved
    for key in ('sampler','initial_sha256'):assert training[names[0]][key]==training[names[1]][key]
    assert rngs[names[0]]==rngs[names[1]]
    assert configs[names[0]]['support_sha256']==configs[names[1]]['grounding_support_sha256']
    assert configs[names[0]]['initial_checkpoint_sha256']==configs[names[1]]['initial_checkpoint_sha256']==sha(RUN/'safety_mean/last.pt')
    assert metrics[names[0]]['scorer_bundle_sha256']==metrics[names[1]]['scorer_bundle_sha256']=='2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e'
    fixed=RUN/'strong_counterfactual_v1/rows.json'
    assert sha(fixed)=='515b353acd0b2d6d0d410b58fd028681fbcb4fa249a597728a53f5818b60a084'
    witnesses=read(fixed)['safety_mean'];families=sorted({r['family'] for r in witnesses});assert len(families)==32
    denom=np.zeros(32);numer={n:np.zeros(32) for n in names};records=[]
    mapped={n:{r['id']:r for r in rows[n]} for n in names}
    for r in witnesses:
        w=set(r['surviving_modes']);j=families.index(r['family']);denom[j]+=len(w)
        record={k:r[k] for k in ('family','target','origin','destination','source_id','destination_id')};record['opportunities']=len(w)
        for n in names:
            d=mapped[n][r['destination_id']];actual={v for v,valid in zip(d['words'],d['valid']) if valid and v is not None}
            count=len(w&actual);numer[n][j]+=count;record[n]=count
        records.append(record)
    rng=np.random.default_rng(610093);draw=rng.integers(32,size=(10000,32))
    boot=(numer[names[1]][draw].sum(1)-numer[names[0]][draw].sum(1))/denom[draw].sum(1)
    retention=dict(opportunities=int(denom.sum()),retained={n:int(numer[n].sum()) for n in names},
        fraction={n:float(numer[n].sum()/denom.sum()) for n in names},
        delta=float((numer[names[1]].sum()-numer[names[0]].sum())/denom.sum()),CI95=np.quantile(boot,[.025,.975]).tolist())
    directions={}
    for a,b in [('open','closed'),('open','shifted'),('closed','open'),('shifted','open')]:
        rr=[r for r in records if r['origin']==a and r['destination']==b]
        directions[a+'_to_'+b]=dict(opportunities=sum(r['opportunities'] for r in rr),retained={n:sum(r[n] for r in rr) for n in names})
    delta=paired(rows[names[0]],rows[names[1]])
    gate=retention['delta']>=.05 and retention['CI95'][0]>0 and delta['valid']['delta']>=-.01 and delta['modes8']['delta']>=-.05 and delta['rare8']['delta']>=-.01 and delta['oracle']['delta']>=-.01 and delta['brier']['delta']<=.01
    write(out/'retention_rows.json',records)
    write(out/'RESULTS.json',dict(metrics=metrics,training=training,configs=configs,route_contrast=delta,fixed_source_retention=retention,
        by_direction=directions,ordinary_augmentation_gate=bool(gate),fixed_source_sha256=sha(fixed),actual_initial_input_group_rng_equal=True,
        scope='One seed, fixed safety_mean source witnesses for both destinations; ordinary verified same-mode target augmentation, no new mechanism or q fitting.'))
    print(dict(retention=retention,route_contrast=delta,gate=gate),flush=True)


if __name__=='__main__':main()
