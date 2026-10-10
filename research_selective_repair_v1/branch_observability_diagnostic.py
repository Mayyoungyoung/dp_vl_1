"""TRAIN oracle ablation: does achieved prefix state explain finite stops?

Never a deployment predictor. All three linear probes have identical fitting
families, observed-prefix rows and fixed regularization. No inference labels.
"""
import argparse,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from research_selective_repair_v1.io import RUN,SOURCE,read,write,sha

def legal_features(paths,completed):
    p=np.asarray(paths);c=np.asarray(completed)
    rel=(p[:,1:,None]-c[:,None])/.1
    tangent=np.diff(p,axis=1)/.05
    clock=np.broadcast_to(np.linspace(0,1,23)[None,:,None],p[:,1:,:1].shape)
    return np.concatenate([(p[:,1:]-p[:,:1])/.3,tangent,rel.reshape(len(p),23,12),np.linalg.norm(rel,axis=-1),clock],-1)

def logistic(x,y,train):
    mean=x[train].mean(0);std=x[train].std(0).clip(.05)
    a=np.column_stack([np.ones(len(x)),(x-mean)/std]);t=a[train];target=y[train]
    def obj(w):
        v=t@w;value=np.mean(np.logaddexp(0,v)-target*v)+.01*np.square(w[1:]).sum()/2
        grad=t.T@(expit(v)-target)/len(target);grad[1:]+=.01*w[1:]
        return value,grad
    result=minimize(obj,np.zeros(a.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-10})
    assert result.success,result.message
    return expit(a@result.x),dict(converged=bool(result.success),iterations=int(result.nit),parameters=len(result.x))

def metrics(p,y):
    a=int(y.sum());b=len(y)-a;r=rankdata(p)
    return dict(rows=len(y),failures=a,Brier=float(np.square(p-y).mean()),
        AUROC=float((r[y==1].sum()-a*(a+1)/2)/(a*b)),
        mean_predicted_failure=float(p.mean()),actual_failure_fraction=float(y.mean()))

def run(name,dataset):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
    manifest=read(dataset.parent/'MANIFEST.json');assert manifest['no_DEV_feedback'] and manifest['rows']==2304
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'}
    f=sorted(set(d['families']));held=np.isin(d['families'],f[-2:])
    observed=d['prefix_observed'];y=d['prefix_hazard'][observed].astype(int)
    held=np.broadcast_to(held[:,None],observed.shape)[observed]
    requested=legal_features(d['paths'],d['completed'])
    prev_q=np.concatenate([d['initial_q'][:,None],d['prefix_q'][:,:-1,-1]],1)
    prev_tip=np.concatenate([d['paths'][:,:1],d['prefix_tip'][:,:-1,-1]],1)
    # Every observed segment after the first has a completed predecessor. No
    # failed endpoint, future joint or unexecuted suffix enters these probes.
    assert np.all(d['prefix_valid'][:,:-1][observed[:,1:]])
    tip_error=(prev_tip-d['paths'][:,:-1])/.05
    variants={'requested':requested,'oracle_achieved_tip':np.concatenate([requested,tip_error],-1),
        'oracle_achieved_tip_joint_branch':np.concatenate([requested,tip_error,np.sin(prev_q),np.cos(prev_q)],-1)}
    records=[]
    for label,x in variants.items():
        p,fit=logistic(x[observed],y,~held)
        records.append(dict(signal=label,optimizer=fit,fit=metrics(p[~held],y[~held]),held_TRAIN=metrics(p[held],y[held])))
    write(out/'SUMMARY.json',dict(records=records,seconds=time.monotonic()-tic,source_commit=SOURCE.name,dataset_sha256=sha(dataset),
        diagnostic_TRAIN_families=list(map(str,f[-2:])),regularization=.01,
        scope='TRAIN-only conditional-stop linear probes. Actual achieved prefix tip/q are diagnostic oracles, never deployed inputs. Raw q metadata explicitly consumed only for this diagnostic, not an event-only learner.',locked_access=False))
    print(records,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',type=Path,required=True);run(**vars(p.parse_args()))
