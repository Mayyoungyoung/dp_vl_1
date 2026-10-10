"""Predict finite-budget fixed-controller outcomes from legal current inputs."""
import argparse,hashlib,os,random,time,itertools
from pathlib import Path
import numpy as np
import torch
from torch import nn
from research_selective_repair_v1.io import RUN,SOURCE,read,write,sha
from scripts.run_observed_probability import torch_setup
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest

WORDS=['|'.join(x) for x in itertools.product(('gap0','gap1','gap2','over'),repeat=2)]

def natural_probabilities(logits,weights):
    # Weighted CE estimates w_y*p(y|x)/Z. Undo that known training reweighting
    # before using mass for expected coverage. This is not calibration evidence.
    return (logits-torch.as_tensor(weights,device=logits.device,dtype=logits.dtype).log()).softmax(-1)

def features(paths,completed):
    """26 relative features at24waypoints; no labels/true boxes/old route input."""
    p=np.asarray(paths,np.float32);c=np.asarray(completed,np.float32)
    assert p.shape[-2:]==(24,3) and c.shape[-2:]==(4,3)
    rel=(p[:,:,None]-c[:,None])/.1
    tangent=np.concatenate([np.zeros_like(p[:,:1]),np.diff(p,axis=1)],1)/.05
    clock=np.broadcast_to(np.linspace(0,1,24,dtype=np.float32)[None,:,None],p.shape[:-1]+(1,))
    goal=np.broadcast_to((p[:,-1:]-p[:,:1])/.3,p.shape)
    return np.concatenate([(p-p[:,:1])/.3,tangent,rel.reshape(len(p),24,12),np.linalg.norm(rel,axis=-1),clock,goal],-1).astype(np.float32)

class OutcomeHead(nn.Module):
    def __init__(self,kind='recurrent'):
        super().__init__();assert kind in ('recurrent','nonrecurrent');self.kind=kind
        self.context=nn.Sequential(nn.Linear(128,32),nn.SiLU())
        self.input=nn.Sequential(nn.Linear(26,64),nn.SiLU())
        if kind=='recurrent':self.sequence=nn.GRU(64,64,batch_first=True)
        else:self.sequence=nn.Sequential(nn.Linear(64,144),nn.SiLU(),nn.Linear(144,96),nn.SiLU(),nn.Linear(96,64),nn.SiLU())
        self.output=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Dropout(.1),nn.Linear(64,17))
    def forward(self,x,context):
        z=self.input(x)
        if self.kind=='recurrent':z,_=self.sequence(z);z=z[:,-1]
        else:z=self.sequence(z).mean(1)
        return self.output(torch.cat([z,self.context(context)],-1))

def fit(name,dataset,kind='recurrent',seed=0,steps=2400,resume=False,stop_after=None):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed immutable fit')
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'}
    families=sorted(set(d['families']));assert len(families)>=4
    # Last two registered TRAIN families are held out only for this fit diagnostic.
    held=families[-2:];train=~np.isin(d['families'],held);test=~train
    x=features(d['paths'],d['completed']);mean=x[train].mean((0,1));std=x[train].std((0,1)).clip(.05)
    y=d['labels'];count=np.bincount(y[train],minlength=17)
    weights=np.sqrt(max(1,count.max())/np.maximum(count,1)).clip(1,5).astype(np.float32)
    settings=dict(kind=kind,seed=seed,steps=steps,batch=32,lr=.0003,dataset_sha256=sha(dataset),mean=mean.tolist(),std=std.tolist(),class_weights=weights.tolist(),fit_families=[str(v) for v in families[:-2]],diagnostic_TRAIN_families=[str(v) for v in held],source_commit=os.environ.get('CODE_COMMIT'),outcomes=['budget_noncompletion_or_unclear_tip']+WORDS,scope='Actual successful-clear executed word. Controller RNG seed is nuisance, not forward input; teacher eight-route ranks do not constitute returned-four execution evidence.')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    model=OutcomeHead(kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    tx=torch.tensor((x-mean)/std,device='cuda');tc=torch.tensor(d['context'],device='cuda');ty=torch.tensor(y,device='cuda',dtype=torch.long);w=torch.tensor(weights,device='cuda')
    initial=tensor_state_digest(model.state_dict());history=[];stream='';start=0;tic=time.monotonic();indices=np.flatnonzero(train)
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.choice(indices,size=32,replace=True);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        model.train();pred=model(tx[ix],tc[ix]);loss=nn.functional.cross_entropy(pred,ty[ix],weight=w)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss)));atomic_checkpoint(out/'recovery.pt',state(step));print(history[-1],flush=True)
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));model.eval()
        with torch.no_grad():prob=natural_probabilities(model(tx,tc),weights).cpu().numpy()
        np.savez_compressed(out/'TRAIN_diagnostic_predictions.npz',probabilities=prob,labels=y,heldout=test,ids=d['ids'],slots=d['slots'],options=d['options'],families=d['families'])
        write(out/'SUMMARY.json',dict(settings=settings,seconds=time.monotonic()-tic,checkpoint_sha256=sha(out/'last.pt'),TRAIN_fit_accuracy=float((prob[train].argmax(-1)==y[train]).mean()),TRAIN_heldout_accuracy=float((prob[test].argmax(-1)==y[test]).mean()),TRAIN_heldout_success_brier=float(np.square((1-prob[test,0])-(y[test]>0)).mean()),TRAIN_heldout_rows=int(test.sum()),class_counts=count.tolist(),stream=stream,scope='TRAIN family heldout diagnostic only; not DEV or fixed returned-four execution'))

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);model=OutcomeHead(ck['settings']['kind']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);return model,ck

def predict(model,ck,paths,completed,context):
    x=features(paths,completed);s=ck['settings']
    tx=torch.tensor((x-np.asarray(s['mean']))/np.asarray(s['std']),device='cuda',dtype=torch.float32)
    tc=torch.tensor(context,device='cuda',dtype=torch.float32)
    with torch.no_grad():return natural_probabilities(model(tx,tc),s['class_weights']).cpu().numpy()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--kind',choices=['recurrent','nonrecurrent'],default='recurrent');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2400);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);fit(**vars(p.parse_args()))
