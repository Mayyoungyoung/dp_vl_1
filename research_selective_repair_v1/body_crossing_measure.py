"""Censored crossing-position measure without unsupervised temporal emissions.

Current requested whole route is known input. Actual events and failed prefixes
are TRAIN labels only. Semantic composition is approximate, not body safety.
"""
import argparse,hashlib,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_prefix_forecast import prefix_features
from research_selective_repair_v1.body_predictive_repair import compose_tensor as timed_compose
from scripts.run_observed_probability import torch_setup
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest

def anchors(paths,completed):
    rows=completed.reshape(-1,2,2,3)[...,0].mean(2)
    a,b=paths[:,:-1],paths[:,1:]
    hits=(a[:,:,None,0]<rows[:,None])&(b[:,:,None,0]>=rows[:,None])
    ix=hits.long().argmax(1)
    first=a.gather(1,ix[...,None].expand(-1,-1,3))
    second=b.gather(1,ix[...,None].expand(-1,-1,3))
    fraction=((rows-first[...,0])/(second[...,0]-first[...,0]).clamp_min(1e-6)).clamp(0,1)
    return (first+fraction[...,None]*(second-first))[...,1:],hits.any(1)

class CrossingHead(nn.Module):
    def __init__(self,kind='recurrent',readout='analytic',hypotheses=4):
        super().__init__();assert kind in ('recurrent','nonrecurrent') and readout in ('analytic','categorical_aux')
        self.kind=kind;self.readout=readout;self.hypotheses=hypotheses
        self.context=nn.Sequential(nn.Linear(128,32),nn.SiLU())
        self.input=nn.Sequential(nn.Linear(26,64),nn.SiLU())
        if kind=='recurrent':self.sequence=nn.GRU(64,64,batch_first=True)
        else:
            self.sequence=nn.Sequential(nn.Linear(64,144),nn.SiLU(),nn.Linear(144,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
            self.aggregate=nn.Sequential(nn.Linear(24*64,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.hazard=nn.Linear(96,hypotheses)
        self.positions=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,hypotheses*8))
        self.weights=nn.Linear(96,hypotheses);self.clear=nn.Linear(96,hypotheses)
        if readout=='categorical_aux':self.category=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,17))
    def forward(self,x,context,paths,completed):
        z=self.input(x);cx=self.context(context)
        if self.kind=='recurrent':z,_=self.sequence(z);summary=z[:,-1]
        else:z=self.sequence(z);summary=self.aggregate(z.flatten(1))
        hidden=torch.cat([z,cx[:,None].expand(-1,24,-1)],-1)
        summary=torch.cat([summary,cx],-1)
        position=self.positions(summary).reshape(len(x),self.hypotheses,8)
        anchor,has=anchors(paths,completed)
        p=dict(hazard=self.hazard(hidden[:,1:]).permute(0,2,1),
            mean=anchor[:,None]+.1*position[...,:4].reshape(len(x),self.hypotheses,2,2),
            scale=.005+.295*position[...,4:].reshape(len(x),self.hypotheses,2,2).sigmoid(),
            log_weights=self.weights(summary).log_softmax(-1),clear=self.clear(summary),has_requested_crossings=has)
        if self.readout=='categorical_aux':p['category']=self.category(summary)
        return p

def likelihood(p,d):
    hazard=nn.functional.binary_cross_entropy_with_logits(p['hazard'],d['prefix_hazard'][:,None].expand_as(p['hazard']),reduction='none')
    loss=(hazard*d['prefix_observed'][:,None]).sum(-1)
    known=d['event_present'].any(1)
    point=(d['event_tip']*d['event_present'][...,None]).sum(1)[...,1:]
    error=(p['mean']-point[:,None])/p['scale']
    nll=(.5*error.square()+p['scale'].log()).sum(-1)
    loss+=(nll*known[:,None]).sum(-1)
    complete=d['prefix_valid'].all(-1)
    clear=nn.functional.binary_cross_entropy_with_logits(p['clear'],(d['labels']>0)[:,None].expand_as(p['clear']).float(),reduction='none')
    loss+=clear*complete[:,None]
    value=-torch.logsumexp(p['log_weights']-loss,-1).mean()
    if 'category' in p:value+=nn.functional.cross_entropy(p['category'],d['labels'].long())
    return value

def compose_tensor(p,cfg):
    if 'category' in p:return p['category'].softmax(-1)
    # Use a single certain emission as an algebraic implementation of the
    # position-only mixture. There are no learned timing variables. Each CDF is
    # evaluated once per row/component; no 23-slot expansion occurs.
    q=dict(p,mean=p['mean'][:,:,None],scale=p['scale'][:,:,None],
        events=torch.full(p['mean'].shape[:2]+(1,2),40.,device=p['mean'].device,dtype=p['mean'].dtype))
    prob=timed_compose(q,cfg)
    valid=p['has_requested_crossings'].all(-1).to(prob.dtype)
    positive=prob[:,1:]*valid[:,None]
    return torch.cat([1-positive.sum(-1,keepdim=True),positive],-1)

def compose(p,paths,cfg):
    t={k:torch.as_tensor(v) for k,v in p.items()}
    with torch.no_grad():prob=compose_tensor(t,cfg).numpy().astype(np.float32)
    analytic='category' not in p
    return prob,dict(internal_route_forwards=len(paths),crossing_hypotheses_per_route=p['mean'].shape[1],
        Gaussian_CDF_evaluations=len(paths)*p['mean'].shape[1]*2*7 if analytic else 0,
        actual_controller_queries=0,public_FK_or_IK_queries=0,
        scope='Direct crossing measure' if analytic else 'Same-prefix/event-feedback categorical auxiliary control')

def fit(name,dataset,kind='recurrent',readout='analytic',seed=0,steps=2400,resume=False,stop_after=None,threads=1):
    torch_setup();assert threads in (1,4);torch.set_num_threads(threads)
    out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed immutable fit')
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'} and read(dataset.parent/'MANIFEST.json')['no_DEV_feedback']
    assert (d['event_present'].sum(1)<=1).all()
    assert d['event_present'][d['labels']>0].any(1).all(),'Successful words need both measured crossings'
    families=sorted(set(d['families']));held=families[-2:];train=~np.isin(d['families'],held);test=~train
    x=prefix_features(d['paths'],d['completed']);mean=x[train].mean((0,1));std=x[train].std((0,1)).clip(.05)
    settings=dict(kind=kind,readout=readout,seed=seed,steps=steps,batch=32,lr=.0003,threads=threads,hypotheses=4,
        dataset_sha256=sha(dataset),mean=mean.tolist(),std=std.tolist(),source_commit=os.environ.get('CODE_COMMIT'),
        fit_families=[str(v) for v in families[:-2]],diagnostic_TRAIN_families=[str(v) for v in held],
        event_sigma='Fixed .005+.295sigmoid,4shared components, no search',
        likelihood='Censored observed stop factors, measured firstcross Gaussian position, complete-route clearance; categorical control additionally uses unweighted17classCE',
        scope='Known current whole requested path/observation; actual firstcross and prefix stop TRAIN labels only; no timing emissions')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    model=CrossingHead(kind,readout).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    t=lambda v:torch.as_tensor(v,device='cuda')
    tx=t(((x-mean)/std).astype(np.float32));tc=t(d['context'].astype(np.float32));paths=t(d['paths']);completed=t(d['completed'])
    target={key:t(d[key]) for key in ('prefix_hazard','prefix_observed','event_tip','event_present','prefix_valid','labels')}
    initial=tensor_state_digest(model.state_dict());history=[];stream='';start=0;tic=time.monotonic();indices=np.flatnonzero(train)
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    step=start
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.choice(indices,size=32,replace=True);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        model.train();pred=model(tx[ix],tc[ix],paths[ix],completed[ix]);loss=likelihood(pred,{k:v[ix] for k,v in target.items()})
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss)));atomic_checkpoint(out/'recovery.pt',state(step));print(history[-1],flush=True)
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));model.eval()
        with torch.no_grad():p={k:v.cpu().numpy() for k,v in model(tx,tc,paths,completed).items()}
        np.savez_compressed(out/'TRAIN_diagnostic_predictions.npz',**p,ids=d['ids'],slots=d['slots'],options=d['options'],families=d['families'],heldout=test)
        point=(d['event_tip']*d['event_present'][...,None]).sum(1)[...,1:];known=d['event_present'].any(1)
        weights=np.exp(p['log_weights']);chosen=weights.argmax(-1);mu=p['mean'][np.arange(len(paths)),chosen]
        proxy=(weights*(1/(1+np.exp(np.clip(p['hazard'],-40,40)))).prod(-1)/(1+np.exp(np.clip(-p['clear'],-40,40)))).sum(-1)
        report={}
        for label,mask in (('fit',train),('heldout',test)):
            report[label]=dict(firstcross_yz_rmse_m=float(np.sqrt(np.square(mu[mask]-point[mask])[known[mask]].mean())),
                successful_clear_proxy_brier=float(np.square(proxy[mask]-(d['labels'][mask]>0)).mean()),rows=int(mask.sum()))
        write(out/'SUMMARY.json',dict(settings=settings,seconds=time.monotonic()-tic,checkpoint_sha256=sha(out/'last.pt'),TRAIN_diagnostics=report,stream=stream,locked_access=False))

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);s=ck['settings'];model=CrossingHead(s['kind'],s['readout'],s['hypotheses']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);return model,ck

def predict(model,ck,paths,completed,context):
    x=prefix_features(paths,completed);s=ck['settings'];t=lambda v:torch.tensor(v,device='cuda',dtype=torch.float32)
    with torch.no_grad():p=model(t((x-np.asarray(s['mean']))/np.asarray(s['std'])),t(context),t(paths),t(completed))
    return {k:v.cpu().numpy() for k,v in p.items()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--kind',choices=['recurrent','nonrecurrent'],default='recurrent');p.add_argument('--readout',choices=['analytic','categorical_aux'],default='analytic');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2400);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--threads',type=int,choices=[1,4],default=1);fit(**vars(p.parse_args()))
