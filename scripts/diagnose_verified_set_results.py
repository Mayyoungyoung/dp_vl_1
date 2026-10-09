"""Secondary, read-only attribution on sealed DEV pools; no model selection."""
import argparse
from collections import Counter
import json
import numpy as np
from scripts.run_observed_probability import ROOT,read,write,sha
from scripts.research_v3_frequency import SUPPORT

RUN=ROOT/'runs/verified_set_v1/all_population_v2'
OLD=ROOT/'runs/research_v3_v1'


def main(names,output):
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    with np.load(SUPPORT/'support.npz') as s:
        known={str(s['ids'][i]):set(s['modes'][i,s['mask'][i]]) for i in range(len(s['ids']))
               if s['splits'][i]=='DEV_MODEL'}
    models={};hashes={};summaries={}
    for name in ['parent']+names:
        folder=(OLD/'safety_mean' if name=='parent' else RUN/name)/'evaluation_matched_q_v1'
        rows=read(folder/'rows.json');assert len(rows)==288
        models[name]={r['id']:r for r in rows};hashes[str(folder/'rows.json')]=sha(folder/'rows.json')
        counts=Counter();per_variant={}
        for r in rows:
            words={w for w,v in zip(r['words'],r['valid']) if v and w is not None}
            counts['valid_slots']+=sum(r['valid']);counts['distinct_valid_modes']+=len(words)
            counts['known_modes_hit']+=len(words&known[r['id']]);counts['unreferenced_modes_hit']+=len(words-known[r['id']])
            counts['duplicates']+=r['duplicates']
            for c in r['candidates']:
                for key in ('semantic_goal_correct','starts_at_current_state','tip_segments_clear','event_state_sequence_correct'):
                    counts['failed_'+key]+=int(not c[key])
            counts['all_wrong_endpoint_requests']+=int(not any(c['semantic_goal_correct'] for c in r['candidates']))
            per_variant.setdefault(r['variant'],[]).append(r['raw'])
        summaries[name]=dict(counts=dict(counts),by_variant={v:{k:float(np.mean([r[k] for r in rr]))
            for k in ('distinct','valid_fraction','recall')} for v,rr in per_variant.items()})
    witness_path=OLD/'strong_counterfactual_v1/rows.json'
    if sha(witness_path)!='515b353acd0b2d6d0d410b58fd028681fbcb4fa249a597728a53f5818b60a084':
        raise ValueError('Frozen parent edit witness file differs')
    witness=read(witness_path)['safety_mean'];hashes[str(witness_path)]=sha(witness_path)
    families=sorted({r['family'] for r in witness});index={f:i for i,f in enumerate(families)}
    denom=np.zeros(len(families));numer={n:np.zeros(len(families)) for n in models}
    for r in witness:
        i=index[r['family']];surviving=set(r['surviving_modes']);denom[i]+=len(surviving)
        for n,m in models.items():
            d=m[r['destination_id']];present={w for w,v in zip(d['words'],d['valid']) if v and w is not None}
            numer[n][i]+=len(surviving&present)
    assert int(denom.sum())==2169
    rng=np.random.default_rng(610093);draw=rng.integers(len(families),size=(5000,len(families)))
    ret={}
    for n in names:
        delta=numer[n]-numer['parent'];boot=delta[draw].sum(1)/denom[draw].sum(1)
        ret[n]=dict(retained=int(numer[n].sum()),fraction=float(numer[n].sum()/denom.sum()),
            difference_from_parent=float(delta.sum()/denom.sum()),CI95=np.quantile(boot,[.025,.975]).tolist())
    result=dict(summaries=summaries,fixed_parent_surviving_opportunities=int(denom.sum()),
        parent_retained=int(numer['parent'].sum()),retention=ret,source_hashes=hashes,
        interpretation='Secondary diagnostic; fixed source/destination witnesses, not complete-mode existence or a revised acceptance gate',
        locked_access=False)
    write(out/'RESULTS.json',result);print(json.dumps(dict(summaries=summaries,retention=ret)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.names,a.output)
