"""TRAIN-supervised post completion from current visible neutral surfaces.

Four sorted posts are a domain assumption, not arbitrary hidden-space inference.
No boxes, edit identity, old routes or checker result enter ConstraintHead.forward.
"""
import argparse, hashlib, os, random, time
import numpy as np
import torch
from torch import nn
from research_selective_repair_v1.core import *
from routeset.train_v2 import atomic_checkpoint, rng_state, restore_rng
from routeset.observed_training_audit import tensor_state_digest


def summaries(points, valid, priors):
    """Current point assignments to TRAIN-fitted ordered post anchors."""
    result=[];bases=[]
    for p,mask in zip(points,valid):
        p=p[mask];assignment=((p[:,None,:2]-priors[None,:,:2])**2).sum(-1).argmin(-1) if len(p) else []
        values=[];base=[]
        for j in range(4):
            q=p[np.asarray(assignment)==j] if len(p) else p
            if len(q):
                quant=np.quantile(q,[0,.1,.5,.9,1],axis=0)
                anchor=np.r_[(quant[0,:2]+quant[-1,:2])/2,quant[-1,2]]
                values.append(np.r_[(quant-anchor).flatten()/.1,anchor, np.std(q,axis=0)/.1, np.log1p(len(q))/8,1])
            else:
                anchor=priors[j].copy();values.append(np.r_[np.zeros(15),anchor,np.zeros(3),0,0])
            base.append(anchor)
        result.append(values);bases.append(base)
    return np.asarray(result,np.float32),np.asarray(bases,np.float32)


class ConstraintHead(nn.Module):
    def __init__(self):
        super().__init__();self.context=nn.Linear(128,32);self.slot=nn.Embedding(4,8)
        self.net=nn.Sequential(nn.Linear(23+32+8,96),nn.SiLU(),nn.Linear(96,96),nn.SiLU(),nn.Linear(96,3))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,context,features,anchors):
        slots=self.slot(torch.arange(4,device=features.device))[None].expand(len(context),-1,-1)
        value=torch.cat([features,self.context(context)[:,None].expand(-1,4,-1),slots],-1)
        return anchors+.08*self.net(value).tanh()


def boxes(completed,settings):
    """Physical completion proposal; its accuracy is measured after sealing."""
    base=settings['post_base'];halfxy=torch.tensor(settings['halfxy'],device=completed.device,dtype=completed.dtype)
    height=(completed[...,2]-base).clamp(.025,.4)
    centers=torch.cat([completed[...,:2],(base+height/2)[...,None]],-1)
    halves=torch.cat([halfxy[None].expand(len(completed),-1,-1),height[...,None]/2],-1)
    return centers,halves


def fit(name,seed=0,steps=2400,data='old_calibrated_targets_v1,interventions_calibrated_targets_v1',resume=False,stop_after=None):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed fit')
    files=[RUN/n/'samples.npz' for n in data.split(',')];parts=[]
    for file in files:
        with np.load(file) as z:
            ix=z['splits']=='TRAIN';parts.append({k:z[k][ix] for k in ('context','obstacle_centers','obstacle_halves','visible_points','visible_mask','floors')})
    d={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    targets=np.concatenate([d['obstacle_centers'][...,:2],(d['obstacle_centers'][...,2]+d['obstacle_halves'][...,2])[...,None]],-1)
    priors=np.median(targets,axis=0);x,anchors=summaries(d['visible_points'],d['visible_mask'],priors)
    settings=dict(seed=seed,steps=steps,batch=32,lr=.0003,data_sha256={str(f):sha(f) for f in files},priors=priors.tolist(),post_base=float(np.median(d['obstacle_centers'][...,2]-d['obstacle_halves'][...,2])),halfxy=np.median(d['obstacle_halves'][...,:2],axis=0).tolist(),floor=float(np.median(d['floors'])),source_commit=os.environ.get('CODE_COMMIT'),scope='TRAIN-fitted four-post ordered completion. Unknown geometry is predicted, never certified free; targets are absent from forward.')
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    model=ConstraintHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    t=[torch.tensor(v,dtype=torch.float32,device='cuda') for v in (d['context'],x,anchors,targets)]
    initial=tensor_state_digest(model.state_dict());history=[];stream='';start=0;tic=time.monotonic()
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.integers(len(x),size=32);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        pred=model(t[0][ix],t[1][ix],t[2][ix]);loss=1000*(pred-t[3][ix]).square().mean()
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss)));atomic_checkpoint(out/'recovery.pt',state(step));print(history[-1],flush=True)
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));write(out/'SUMMARY.json',dict(settings=settings,checkpoint_sha256=sha(out/'last.pt'),seconds=time.monotonic()-tic,stream=stream,initial_sha256=initial))


def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);model=ConstraintHead().cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);return model,ck


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2400);p.add_argument('--data',default='old_calibrated_targets_v1,interventions_calibrated_targets_v1');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);fit(**vars(p.parse_args()))
