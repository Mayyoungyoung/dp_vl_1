"""Fixed-seed mean contrast, resampling DEV layout families, not candidates."""
import argparse
import json
import numpy as np


def paired_summary(left, right, key, draws=5000):
    if len(left) != len(right) or not left:
        raise ValueError('Aligned nonempty seeds required')
    matrix=[]; families=None; ids=None
    for aa,bb in zip(left,right):
        a={r['id']:r for r in aa};b={r['id']:r for r in bb}
        if len(a)!=len(aa) or len(b)!=len(bb) or a.keys()!=b.keys():
            raise ValueError('Duplicate or missing paired request')
        if ids is None: ids=set(a)
        if set(a)!=ids: raise ValueError('Different seed populations')
        groups={}
        for ident in sorted(a):
            if a[ident]['family']!=b[ident]['family']:raise ValueError('Family mismatch')
            groups.setdefault(a[ident]['family'],[]).append(a[ident]['raw'][key]-b[ident]['raw'][key])
        fs=sorted(groups)
        if families is None:families=fs
        if families!=fs:raise ValueError('Different family populations')
        matrix.append([np.mean(groups[f]) for f in fs])
    values=np.asarray(matrix);average=values.mean(0)
    rng=np.random.default_rng(904100)
    boot=average[rng.integers(len(families),size=(draws,len(families)))].mean(1)
    return dict(mean_difference=float(values.mean()),seed_differences=values.mean(1).tolist(),
        fixed_seed_family_CI95=np.quantile(boot,[.025,.975]).tolist(),families=len(families),
        requests=len(ids),seeds=len(left),scope='Family bootstrap conditional on these continuation seeds; not independent pretraining uncertainty')


def main(output):
    from scripts.run_observed_probability import ROOT,read,write,sha
    run=ROOT/'runs/verified_set_v1/all_population_v2'
    parent=ROOT/'runs/research_v3_v1/safety_mean/evaluation_matched_q_v1'
    out=run/output;out.mkdir(parents=True,exist_ok=False)
    arms=['ordinary','gate','budget_match','set_point'];rows={};summaries={};hashes={}
    parents=read(parent/'rows.json');hashes[str(parent/'rows.json')]=sha(parent/'rows.json')
    for arm in arms:
        rows[arm]=[];summaries[arm]={}
        metrics=[]
        for seed in range(3):
            folder=run/(arm+'_seed'+str(seed))/'evaluation_matched_q_v1'
            rr=read(folder/'rows.json');assert len(rr)==288
            rows[arm].append(rr);metrics.append(read(folder/'metrics.json'))
            hashes[str(folder/'rows.json')]=sha(folder/'rows.json')
        for key in ('valid_fraction','distinct','recall'):
            vv=[m['raw'][key] for m in metrics]
            summaries[arm][key]=dict(mean=float(np.mean(vv)),seed_values=vv,min=float(min(vv)),max=float(max(vv)))
        for key in ('valid_fraction','distinct'):
            vv=[m['selected'][key] for m in metrics]
            summaries[arm]['selected_'+key]=dict(mean=float(np.mean(vv)),seed_values=vv)
        summaries[arm]['brier']=[m['reliability']['brier'] for m in metrics]
    contrast={}
    for arm in arms:
        contrast[arm]={control:{k:paired_summary(rows[arm],([parents]*3 if control=='parent' else rows[control]),k)
            for k in ('valid_fraction','distinct','recall')} for control in ['parent']+arms if control!=arm}
    result=dict(summary=summaries,contrasts=contrast,source_hashes=hashes,locked_access=False,
        interpretation='Post-diagnostic replication, shared parent; primary gate unchanged. No population-level seed CI from three seeds.')
    write(out/'RESULTS.json',result);print(json.dumps(summaries),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    main(p.parse_args().output)
