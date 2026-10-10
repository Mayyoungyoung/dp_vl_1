"""Censored first-crossing measures; actual joint states never enter forward.

This avoids identifying a periodic seven-joint trajectory merely to determine
two passage events. Gaussian mixtures, survival likelihoods and GRUs are
established tools. Any contribution needs matched counterfactual experiments.
"""
import argparse,hashlib,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from scipy.special import ndtr
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_prefix_forecast import prefix_features
from scripts.run_observed_probability import torch_setup
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest

class EventHead(nn.Module):
    def __init__(self,kind='recurrent',hypotheses=4):
        super().__init__();assert kind in ('recurrent','nonrecurrent')
        self.kind=kind;self.hypotheses=hypotheses
        self.context=nn.Sequential(nn.Linear(128,32),nn.SiLU())
        self.input=nn.Sequential(nn.Linear(26,64),nn.SiLU())
        if kind=='recurrent':self.sequence=nn.GRU(64,64,batch_first=True)
        else:self.sequence=nn.Sequential(nn.Linear(64,144),nn.SiLU(),nn.Linear(144,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.output=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,hypotheses*11))
        self.weights=nn.Linear(96,hypotheses);self.clear=nn.Linear(96,hypotheses)
    def forward(self,x,context,paths):
        z=self.input(x);cx=self.context(context)
        initial=torch.cat([z[:,0],cx],-1)
        if self.kind=='recurrent':z,_=self.sequence(z)
        else:z=self.sequence(z)
        hidden=torch.cat([z,cx[:,None].expand(-1,24,-1)],-1)
        p=self.output(hidden[:,1:]).reshape(len(x),23,self.hypotheses,11).permute(0,2,1,3)
        # Each segment conditions on its requested endpoint. No recursively
        # accumulated q or unknown future executed state is supplied.
        mean=paths[:,None,1:,None,1:]+.1*p[...,3:7].reshape(len(x),self.hypotheses,23,2,2)
        scale=.005+.295*p[...,7:11].reshape_as(mean).sigmoid()
        return dict(hazard=p[...,0],events=p[...,1:3],mean=mean,scale=scale,
            log_weights=self.weights(initial).log_softmax(-1),clear=self.clear(hidden[:,-1]))

def likelihood(p,d):
    risk=nn.functional.binary_cross_entropy_with_logits(p['hazard'],d['prefix_hazard'][:,None].expand_as(p['hazard']),reduction='none')
    loss=(risk*d['prefix_observed'][:,None]).sum(-1)
    event=nn.functional.binary_cross_entropy_with_logits(p['events'],d['event_present'][:,None].expand_as(p['events']).float(),reduction='none')
    loss+=(event*d['event_observed'][:,None]).sum((-1,-2))
    error=(p['mean']-d['event_tip'][:,None,...,1:])/p['scale']
    nll=(.5*error.square()+p['scale'].log()).sum(-1)
    loss+=(nll*d['event_present'][:,None]).sum((-1,-2))
    complete=d['prefix_valid'].all(-1)
    clear=nn.functional.binary_cross_entropy_with_logits(p['clear'],(d['labels']>0)[:,None].expand_as(p['clear']).float(),reduction='none')
    loss+=clear*complete[:,None]
    return -torch.logsumexp(p['log_weights']-loss,-1).mean()

def success(p):
    continuation=1/(1+np.exp(np.clip(p['hazard'],-40,40)))
    clear=1/(1+np.exp(np.clip(-p['clear'],-40,40)))
    return (np.exp(p['log_weights'])*continuation.prod(-1)*clear).sum(-1)

def compose(p,paths,cfg):
    """Integrate uncertain crossing positions against current inferred posts.

    Mixture component ties both rows. Conditional within-component spatial and
    temporal factorization remains an approximation, not a safety certificate.
    """
    n,k,s,r,_=p['mean'].shape;assert (s,r)==(23,2)
    occurrence=1/(1+np.exp(np.clip(-p['events'],-40,40)))
    prior=np.concatenate([np.ones((n,k,1,2)),np.cumprod(1-occurrence,axis=2)[:,:,:-1]],2)
    first=occurrence*prior;row_mass=np.zeros((n,k,2,4))
    for row in range(2):
        mean=p['mean'][...,row,:];scale=p['scale'][...,row,:]
        assert np.all(scale>0)
        top=cfg['post_base_z']+cfg['post_heights'][row]+cfg['tip_clearance_m']
        below=ndtr((top-mean[...,1])/scale[...,1])
        ys=cfg['post_y'][row];assert len(ys)==2
        margin=.0175+cfg['tip_clearance_m']
        limits=[(-np.inf,ys[0]-margin),(ys[0]+margin,ys[1]-margin),(ys[1]+margin,np.inf)]
        for j,(lo,hi) in enumerate(limits):
            mass=np.maximum(0,ndtr((hi-mean[...,0])/scale[...,0])-ndtr((lo-mean[...,0])/scale[...,0]))
            row_mass[...,row,j]=(first[...,row]*mass*below).sum(-1)
        row_mass[...,row,3]=(first[...,row]*(1-below)).sum(-1)
    continuation=1/(1+np.exp(np.clip(p['hazard'],-40,40)))
    clear=1/(1+np.exp(np.clip(-p['clear'],-40,40)))
    mass=np.exp(p['log_weights'])*continuation.prod(-1)*clear
    joint=(row_mass[:,:,0,:,None]*row_mass[:,:,1,None,:]).reshape(n,k,16)
    result=np.zeros((n,17));result[:,1:]=(mass[:,:,None]*joint).sum(1)
    result[:,0]=1-result[:,1:].sum(-1)
    assert np.isfinite(result).all() and result.min()>=-1e-7
    return result.astype(np.float32),dict(internal_route_forwards=n,event_hypotheses_per_route=k,
        Gaussian_CDF_evaluations=n*k*23*2*7,actual_controller_queries=0,public_FK_or_IK_queries=0,
        scope='Censored event measure with current inferred boxes; conditional factorization,not physical validation')

def fit(name,dataset,kind='recurrent',seed=0,steps=2400,resume=False,stop_after=None,threads=4):
    torch_setup();assert threads in (1,4);torch.set_num_threads(threads)
    out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed immutable fit')
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'} and read(dataset.parent/'MANIFEST.json')['no_DEV_feedback']
    families=sorted(set(d['families']));held=families[-2:];train=~np.isin(d['families'],held);test=~train
    x=prefix_features(d['paths'],d['completed']);mean=x[train].mean((0,1));std=x[train].std((0,1)).clip(.05)
    settings=dict(kind=kind,seed=seed,steps=steps,batch=32,lr=.0003,threads=threads,hypotheses=4,
        dataset_sha256=sha(dataset),mean=mean.tolist(),std=std.tolist(),source_commit=os.environ.get('CODE_COMMIT'),
        fit_families=[str(v) for v in families[:-2]],diagnostic_TRAIN_families=[str(v) for v in held],
        event_sigma='Learned independent y/z diagonal scales .005+.295*sigmoid; fixed parameterization,not tuned on DEV',
        likelihood='Sum censored hazard/event factors plus known event Gaussian density; successful-clear conditioned on completed route',
        scope='Exact actual firstcross positions/train stop labels only; forward current path/current observation')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    model=EventHead(kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    t=lambda v:torch.as_tensor(v,device='cuda')
    tx=t(((x-mean)/std).astype(np.float32));tc=t(d['context'].astype(np.float32));paths=t(d['paths'])
    target={key:t(d[key]) for key in ('prefix_hazard','prefix_observed','event_tip','event_present','event_observed','prefix_valid','labels')}
    initial=tensor_state_digest(model.state_dict());history=[];stream='';start=0;tic=time.monotonic();indices=np.flatnonzero(train)
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    step=start
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.choice(indices,size=32,replace=True);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        model.train();pred=model(tx[ix],tc[ix],paths[ix]);loss=likelihood(pred,{k:v[ix] for k,v in target.items()})
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss)));atomic_checkpoint(out/'recovery.pt',state(step));print(history[-1],flush=True)
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));model.eval()
        with torch.no_grad():p={k:v.cpu().numpy() for k,v in model(tx,tc,paths).items()}
        np.savez_compressed(out/'TRAIN_diagnostic_predictions.npz',**p,ids=d['ids'],slots=d['slots'],options=d['options'],families=d['families'],heldout=test)
        report={};weights=np.exp(p['log_weights']);prob=success(p)
        for label,mask in (('fit',train),('heldout',test)):
            ix=np.flatnonzero(mask);chosen=weights[mask].argmax(-1);mu=p['mean'][ix,chosen];valid=d['event_present'][mask]
            occurrence=(weights[mask,:,None,None]/(1+np.exp(np.clip(-p['events'][mask],-40,40)))).sum(1)
            report[label]=dict(firstcross_yz_rmse_m=float(np.sqrt(np.square(mu-d['event_tip'][mask,...,1:])[valid].mean())),
                firstcross_occurrence_brier=float(np.square(occurrence-d['event_present'][mask])[d['event_observed'][mask]].mean()),
                successful_clear_brier=float(np.square(prob[mask]-(d['labels'][mask]>0)).mean()),rows=int(mask.sum()))
        write(out/'SUMMARY.json',dict(settings=settings,seconds=time.monotonic()-tic,checkpoint_sha256=sha(out/'last.pt'),TRAIN_diagnostics=report,stream=stream,locked_access=False))

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);s=ck['settings'];model=EventHead(s['kind'],s['hypotheses']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);return model,ck

def predict(model,ck,paths,completed,context):
    x=prefix_features(paths,completed);s=ck['settings'];t=lambda v:torch.tensor(v,device='cuda',dtype=torch.float32)
    with torch.no_grad():p=model(t((x-np.asarray(s['mean']))/np.asarray(s['std'])),t(context),t(paths))
    return {k:v.cpu().numpy() for k,v in p.items()}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--name',required=True);parser.add_argument('--dataset',type=Path,required=True);parser.add_argument('--kind',choices=['recurrent','nonrecurrent'],default='recurrent');parser.add_argument('--seed',type=int,default=0);parser.add_argument('--steps',type=int,default=2400);parser.add_argument('--resume',action='store_true');parser.add_argument('--stop-after',type=int);parser.add_argument('--threads',type=int,choices=[1,4],default=4);fit(**vars(parser.parse_args()))
