"""TRAIN control for learned crossing position vs a fitted constant residual."""
import argparse,time
from pathlib import Path
import numpy as np
from scipy.special import ndtr
from research_selective_repair_v1.io import RUN,SOURCE,read,write,sha
from research_selective_repair_v1.execution_semantics import word
from research_selective_repair_v1.body_feedback_data import WORDS

def anchor(paths,completed):
    rows=completed.reshape(-1,2,2,3)[...,0].mean(2)
    a,b=paths[:,:-1],paths[:,1:]
    hits=(a[:,:,None,0]<rows[:,None])&(b[:,:,None,0]>=rows[:,None]);ix=hits.argmax(1)
    first=a[np.arange(len(a))[:,None],ix];second=b[np.arange(len(a))[:,None],ix]
    t=np.clip((rows-first[...,0])/np.maximum(second[...,0]-first[...,0],1e-6),0,1)
    return (first+t[...,None]*(second-first))[...,1:],hits.any(1)

def metrics(prob,y,mask):
    p=prob[mask];y=y[mask];pos=y>0
    return dict(rows=len(y),positive_rows=int(pos.sum()),
        full_composed_successful_clear_brier=float(np.square(1-p[:,0]-(y>0)).mean()),
        conditional_word_accuracy=float((p[:,1:].argmax(-1)[pos]+1==y[pos]).mean()),
        conditional_word_nll=float(-np.log(np.maximum(p[pos,y[pos]]/np.maximum(p[pos,1:].sum(-1),1e-8),1e-8)).mean()))

def run(name,dataset):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False);tic=time.monotonic()
    m=read(dataset.parent/'MANIFEST.json');assert m['rows']==2304 and m['no_DEV_feedback']
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'};fam=sorted(set(d['families']));held=np.isin(d['families'],fam[-2:])
    a,has=anchor(d['paths'],d['completed']);known=d['event_present'].any(1)
    point=(d['event_tip']*d['event_present'][...,None]).sum(1)[...,1:]
    bias=[];scale=[];counts=[]
    for row in range(2):
        use=known[:,row]&has[:,row]&~held;residual=point[use,row]-a[use,row]
        bias.append(residual.mean(0));scale.append(residual.std(0).clip(.005,.3));counts.append(int(use.sum()))
    bias=np.asarray(bias);scale=np.asarray(scale);mu=a+bias
    post_base=read(RUN/'constraints_seed0_v1/config.json')['post_base'];mass=np.zeros((len(a),16));planned=np.zeros_like(mass)
    for ident in sorted(set(d['ids'])):
        ix=np.flatnonzero(d['ids']==ident);c=d['completed'][ix[0]];heights=np.clip(c[:,2]-post_base,.025,.4)
        cfg=dict(row_x=[float(c[j:j+2,0].mean()) for j in (0,2)],post_y=[c[j:j+2,1].tolist() for j in (0,2)],post_heights=[float(heights[j:j+2].max()) for j in (0,2)],post_base_z=post_base,tip_clearance_m=.02)
        rows=[]
        for row in range(2):
            below=ndtr((post_base+cfg['post_heights'][row]+.02-mu[ix,row,1])/scale[row,1]);ys=cfg['post_y'][row];margin=.0375
            edges=[(-np.inf,ys[0]-margin),(ys[0]+margin,ys[1]-margin),(ys[1]+margin,np.inf)]
            gap=[below*(ndtr((hi-mu[ix,row,0])/scale[row,0])-ndtr((lo-mu[ix,row,0])/scale[row,0])) for lo,hi in edges]
            rows.append(np.stack(gap+[1-below],-1)*has[ix,row,None])
        mass[ix]=(rows[0][:,:,None]*rows[1][:,None,:]).reshape(len(ix),16)
        for i in ix:
            label=word(d['paths'][i],cfg)
            if label in WORDS:planned[i,WORDS.index(label)]=1
    reports=[];hashes={str(dataset):sha(dataset)}
    for seed in range(3):
        file=RUN/('body_binary_seed%d_v1'%seed)/'TRAIN_diagnostic_predictions.npz';hashes[str(file)]=sha(file)
        with np.load(file) as z:
            assert np.array_equal(z['ids'],d['ids']) and np.array_equal(z['slots'],d['slots']) and np.array_equal(z['options'],d['options'])
            success=z['success']
        for label,spatial in [('constant_position_residual',mass),('planned_word_binary',planned)]:
            positive=success[:,None]*spatial;prob=np.column_stack([1-positive.sum(-1),positive])
            reports.append(dict(seed=seed,control=label,fit=metrics(prob,d['labels'],~held),held_TRAIN=metrics(prob,d['labels'],held)))
    write(out/'SUMMARY.json',dict(records=reports,bias_yz_m=bias.tolist(),scale_yz_m=scale.tolist(),fit_known_crossings=counts,seconds=time.monotonic()-tic,
        source_commit=SOURCE.name,input_sha256=hashes,constant_fit='Independent per-row diagonal Gaussian moments on known TRAIN actual crossing residuals,including incomplete routes; unknown crossings excluded',
        scope='TRAIN held families14/15; common three trained binary heads. Tests analytic readout/global residual without any learned crossing map; no deployment result',locked_access=False))
    print(reports,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',type=Path,required=True);run(**vars(p.parse_args()))
