"""Read-only parent context for the registered fixed-source comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.research_v3_analyze_frequency import paired


def main(workspace,augmented_root,output):
    base=Path(workspace);new=Path(augmented_root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    paths={n:base/'runs/research_v3_linear_closure_20261009/runs/research_v3_v1'/folder/'evaluation_matched_q_v1' for n,folder in [('parent','safety_mean'),('fresh','margin_mean')]}
    paths['augmented']=new/'verified_edit_augmented/evaluation_matched_q_v1';hashes={}
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def read(p):
        hashes[str(p)]=sha(p);return json.loads(p.read_text())
    rows={n:read(p/'rows.json') for n,p in paths.items()};metrics={n:read(p/'metrics.json') for n,p in paths.items()}
    assert len({m['scorer_bundle_sha256'] for m in metrics.values()})==1
    for n,p in paths.items():assert sha(p/'pool.npz')==metrics[n]['pool_sha256']
    fixed=base/'runs/research_v3_strong_counterfactual_closure_20261009/runs/research_v3_v1/strong_counterfactual_v1/rows.json'
    assert sha(fixed)=='515b353acd0b2d6d0d410b58fd028681fbcb4fa249a597728a53f5818b60a084'
    witnesses=read(fixed)['safety_mean'];families=sorted({r['family'] for r in witnesses});denom=np.zeros(len(families))
    numer={n:np.zeros(len(families)) for n in rows};maps={n:{r['id']:r for r in rr} for n,rr in rows.items()}
    for r in witnesses:
        i=families.index(r['family']);w=set(r['surviving_modes']);denom[i]+=len(w)
        for n in rows:
            d=maps[n][r['destination_id']];actual={v for v,valid in zip(d['words'],d['valid']) if valid and v is not None}
            numer[n][i]+=len(w&actual)
    rng=np.random.default_rng(610093);draws=rng.integers(len(families),size=(10000,len(families)))
    contrasts={}
    for n,a,b in [('augmented_minus_parent','parent','augmented'),('fresh_minus_parent','parent','fresh'),('augmented_minus_fresh','fresh','augmented')]:
        values=numer[b]-numer[a];boot=values[draws].sum(1)/denom[draws].sum(1)
        contrasts[n]=dict(delta=float(values.sum()/denom.sum()),CI95=np.quantile(boot,[.025,.975]).tolist())
    primary=read(new/'verified_edit_analysis_v1/RESULTS.json')
    for k in ('delta','CI95'):np.testing.assert_array_equal(contrasts['augmented_minus_fresh'][k],primary['fixed_source_retention'][k])
    result=dict(retention=dict(opportunities=int(denom.sum()),retained={n:int(v.sum()) for n,v in numer.items()},fraction={n:float(v.sum()/denom.sum()) for n,v in numer.items()},contrasts=contrasts),
        versus_parent=paired(rows['parent'],rows['augmented']),metrics=metrics,input_sha256=hashes,primary_replay_exact=True,
        scope='Contextual unchanged-parent comparison registered before DEV outputs; no changed primary gate, training, rescoring, or checkpoint selection.')
    (out/'RESULTS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(retention=result['retention'],versus_parent=result['versus_parent']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--workspace',required=True);p.add_argument('--augmented-root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.workspace,a.augmented_root,a.output)
