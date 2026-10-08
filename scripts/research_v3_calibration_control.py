"""Conventional monotone calibrators, calibrated on a separate reserved role."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize,minimize_scalar
from scripts.run_observed_probability import sha,read,write
from scripts.research_v3_frequency import RUN,coverage
from scripts.paired_modes_reliability import reliability_metrics
from scripts.analyze_paired_selection import select
from scripts.research_v3_analyze_frequency import paired,row_metrics


def fit_calibrators(q,y):
    from scipy.special import expit
    q=np.clip(np.asarray(q,dtype=np.float64),1e-7,1-1e-7);y=np.asarray(y,dtype=np.float64)
    x=np.log(q/(1-q))
    def objective(theta):
        a,b=theta;scaled=np.exp(a)*x;z=scaled+b
        loss=float(np.mean(np.logaddexp(0,z)-y*z));residual=expit(z)-y
        grad=np.array([np.mean(residual*scaled),np.mean(residual)])
        return loss,grad
    affine=minimize(objective,np.zeros(2),jac=True,method='L-BFGS-B',bounds=[(-3,3),(-10,10)])
    temp=minimize_scalar(lambda t:objective((-t,0))[0],bounds=(-3,3),method='bounded')
    if not affine.success or not temp.success:raise RuntimeError('Calibration optimization failed')
    return dict(temperature=dict(a=-float(temp.x),b=0.,calibration_nll=float(temp.fun),status=str(temp.message)),
                affine=dict(a=float(affine.x[0]),b=float(affine.x[1]),calibration_nll=float(affine.fun),status=str(affine.message)))


def main(output):
    from scipy.special import expit
    out=RUN/output;out.mkdir(exist_ok=False)
    root=RUN/'matched_q_paired_v1/evaluation'
    files={r:root/r/'mean_seed0/pool.npz' for r in ('CALIBRATION','paired_dev')}
    data={}
    for role,path in files.items():
        receipt=read(path.parent/'receipt.json');assert receipt['pool_sha256']==sha(path)
        with np.load(path) as z:data[role]={k:z[k] for k in z.files}
    assert not set(data['CALIBRATION']['parents'])&set(data['paired_dev']['parents'])
    fits=fit_calibrators(data['CALIBRATION']['q'],data['CALIBRATION']['labels'])
    dev=data['paired_dev'];base=read(RUN/'safety_mean/evaluation_fixed_q_v2/rows.json')
    with np.load(RUN/'frequency_support_v1/support.npz') as z:support={k:z[k] for k in z.files}
    si={str(k):i for i,k in enumerate(support['ids'])}
    x=np.log(np.clip(dev['q'].astype(np.float64),1e-7,1-1e-7)/(1-np.clip(dev['q'].astype(np.float64),1e-7,1-1e-7)))
    results={}
    for name,cfg in fits.items():
        logits=np.exp(cfg['a'])*x+cfg['b'];q=expit(logits);rows=copy.deepcopy(base)
        for i,r in enumerate(rows):
            assert r['id']==str(dev['ids'][i]);j=si[r['id']];known=set(support['modes'][j,support['mask'][j]])
            rare=known-{'gap0|gap0'};keep=select(dev['paths'][i],q[i],4);r['q']=q[i].tolist()
            r['selected']=coverage(r['words'],np.array(r['valid']),known,keep)
            found={r['words'][k] for k in keep if r['valid'][k]};r['rare_recall4']=len(found&rare)/len(rare)
        metrics=[row_metrics(r) for r in rows]
        results[name]=dict(fit=cfg,reliability=reliability_metrics(logits,dev),contrast=paired(base,rows),
            metrics={k:float(np.mean([r[k] for r in metrics])) for k in metrics[0]})
        write(out/(name+'_rows.json'),rows)
        np.savez_compressed(out/(name+'_predictions.npz'),q=q,logits=logits,labels=dev['labels'],ids=dev['ids'])
    write(out/'RESULTS.json',dict(results=results,input_sha256={r:sha(p) for r,p in files.items()},
        scope='Calibration only on newCALIBRATION. Ordinary controls, no DEV fit and no novelty claim.'))
    print({k:v['metrics'] for k,v in results.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
