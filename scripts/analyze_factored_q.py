"""Complete crossed-seed analysis; intervals cluster original scene families."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import ROOT,read,write,sha


def analyze(root,output,version='evaluation'):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    arms=['single','joint','marginal','conditional'];data={};inputs={}
    for arm in arms:
        for gs in range(3):
            for ss in range(3):
                name='%s_g%d_s%d'%(arm,gs,ss);p=root/name/(version+'.json')
                data[name]=read(p);inputs[name]=sha(p)
    means={};per_generator={};comparisons={};rng=np.random.default_rng(731)
    for role in ('paired_dev','old_dev','dev32'):
        for mode in ('raw','calibrated'):
            for arm in arms:
                rows=[data['%s_g%d_s%d'%(arm,g,s)]['results'][role][mode] for g in range(3) for s in range(3)]
                values=[dict(r['set_metrics'],ece=r['reliability_bins_ece'],selected_ece=r['selected_reliability_bins_ece'],aurc=r['selected_aurc']) for r in rows]
                means['%s/%s/%s'%(role,mode,arm)]={k:dict(mean=float(np.mean([v[k] for v in values])),sd=float(np.std([v[k] for v in values],ddof=1))) for k in values[0]}
                per_generator['%s/%s/%s'%(role,mode,arm)]=[{k:float(np.mean([v[k] for v in values[g*3:(g+1)*3]])) for k in values[0]} for g in range(3)]
            for base in ('single','joint','marginal'):
                for metric in ('brier','nll','top1','k4_valid','k4_all','k4_modes','accepted08_modes'):
                    a=[data['conditional_g%d_s%d'%(g,s)]['results'][role][mode]['families'] for g in range(3) for s in range(3)]
                    b=[data['%s_g%d_s%d'%(base,g,s)]['results'][role][mode]['families'] for g in range(3) for s in range(3)]
                    families=sorted(a[0]);assert all(sorted(r)==families for r in a+b)
                    diff=np.array([np.mean([aa[f][metric]-bb[f][metric] for aa,bb in zip(a,b)]) for f in families])
                    boot=diff[rng.integers(len(diff),size=(5000,len(diff)))].mean(1)
                    comparisons['%s/%s/conditional-%s/%s'%(role,mode,base,metric)]=dict(mean=float(diff.mean()),ci95=np.quantile(boot,[.025,.975]).tolist(),families=len(diff))
    # Verify every capacity-matched arm used identical initialization and sample order.
    checks=[]
    for g in range(3):
        for s in range(3):
            configs=[read(root/('%s_g%d_s%d'%(a,g,s))/'config.json') for a in arms]
            fits=[read(root/('%s_g%d_s%d'%(a,g,s))/'fit_summary.json') for a in arms]
            assert len({c['initial_sha256'] for c in configs[1:]})==1
            # Sampler stores initialization too; compare only actual order fingerprint.
            sampler_keys=['batches','observation_draws','index_chain_sha256','initial_sampler_state_sha256']
            for k in sampler_keys:assert all(f['sampler'][k]==fits[0]['sampler'][k] for f in fits),k
            checks.append(dict(generator_seed=g,scorer_seed=s,matched_initialization=True,matched_sampler_keys=sampler_keys))
    write(output/'RESULTS.json',dict(means=means,per_generator=per_generator,comparisons=comparisons,checks=checks,input_sha256=inputs,
         scope='3 frozen generator seeds x3 scorer seeds; repeated seeds are not extra scene families. Family-bootstrap intervals condition on trained models. Reused DEV only. No generation improvements possible from frozen pools.'))
    lines=['# Factored route reliability: complete comparison','', '| Split / arm | Top1 % | Brier | NLL | ECE % | K4 all-valid % | K4 valid modes |','|---|---:|---:|---:|---:|---:|---:|']
    for role in ('paired_dev','old_dev','dev32'):
        for arm in arms:
            v={k:x['mean'] for k,x in means['%s/calibrated/%s'%(role,arm)].items()}
            lines.append('|%s/%s|%.2f|%.4f|%.4f|%.2f|%.2f|%.3f|'%(role,arm,100*v['top1'],v['brier'],v['nll'],100*v['ece'],100*v['k4_all'],v['k4_modes']))
    (output/'TABLE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(ROOT/'runs/factored_q_v1'));p.add_argument('--output',default=str(ROOT/'runs/factored_q_v1/analysis'));p.add_argument('--evaluation-version',default='evaluation');a=p.parse_args();analyze(a.root,a.output,a.evaluation_version)
