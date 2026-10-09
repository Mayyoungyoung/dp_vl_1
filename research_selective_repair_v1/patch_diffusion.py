"""Adapted connected-local diffusion control, not a reproduction of joint-space CDM."""
import argparse,hashlib,os,random,time
import numpy as np
import torch
from torch import nn
from research_selective_repair_v1.core import *
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest

class LocalDiffusion(nn.Module):
    goal_tail=True
    def __init__(self,width,seed=0):
        super().__init__();self.seed=seed;self.context=nn.Linear(width,64);self.mode=nn.Embedding(16,16)
        self.net=nn.Sequential(nn.Conv1d(33+64+16+3+8,128,1),nn.SiLU(),nn.Conv1d(128,128,3,padding=1),nn.SiLU(),nn.Conv1d(128,128,3,padding=1),nn.SiLU(),nn.Conv1d(128,3,1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
        self.register_buffer('alpha',torch.cumprod(1-torch.linspace(.0001,.02,1000),0))
    def epsilon(self,context,modes,local,noisy,step):
        b,k,h,_=local.shape;ctx=self.context(context)[:,None,None].expand(-1,k,h,-1);mod=self.mode(modes)[:,:,None].expand(-1,-1,h,-1)
        phase=step.to(local.dtype)[:,None]/999*torch.tensor([1.,2.,4.,8.],device=local.device)[None]*np.pi
        timing=torch.cat([phase.sin(),phase.cos()],-1)[:,None,None].expand(-1,k,h,-1)
        value=torch.cat([local,ctx,mod,noisy,timing],-1).reshape(b*k,h,-1).transpose(1,2)
        return self.net(value).transpose(1,2).reshape(b,k,h,3)
    def forward(self,context,modes,drafts,local,hard=True,threshold=.025,scale=1.):
        b,k,h,_=drafts.shape
        near=(local[...,12:16].min(-1).values*.1<.045).float()
        near=torch.nn.functional.max_pool1d(near.reshape(b*k,1,h),5,stride=1,padding=2).reshape(b,k,h)
        support=masks(near,.5,12);support[...,0]=0
        diff=local[:,:,0,29:32]*.4;trigger=(diff.norm(dim=-1)>=threshold)&(local[:,:,0,32]>.5)
        s=torch.linspace(0,1,7,device=local.device)[1:];s=s*s*(3-2*s)
        support[:,:,18:]=torch.maximum(support[:,:,18:],trigger[...,None]*s)
        support[:,:,-1]=trigger.float()
        self.last_sampling_steps=0
        if scale==0 or not bool(support.any()):
            return drafts,dict(support=torch.zeros_like(support),prob=near,delta=torch.zeros_like(drafts),logits=torch.zeros_like(support))
        generator=torch.Generator(device=local.device);generator.manual_seed(self.seed)
        x=torch.randn(drafts.shape,device=local.device,dtype=local.dtype,generator=generator)
        steps=torch.linspace(999,0,12,device=local.device).long()
        for j,t in enumerate(steps):
            a=self.alpha[t];eps=self.epsilon(context,modes,local,x,t.expand(b));clean=((x-(1-a).sqrt()*eps)/a.sqrt()).clamp(-1,1)
            if j+1==len(steps):x=clean
            else:
                next_a=self.alpha[steps[j+1]];x=next_a.sqrt()*clean+(1-next_a).sqrt()*eps
        bound=torch.full_like(support,.08);bound[:,:,18:]=.4
        self.last_sampling_steps=12
        delta=x*bound[...,None];path=drafts+scale*support[...,None]*delta
        return path,dict(support=support,prob=near,delta=delta,logits=torch.zeros_like(support))

def train(name,seed=0,data='old_calibrated_targets_v1,interventions_calibrated_targets_v1',steps=2400,resume=False,stop_after=None):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed diffusion training')
    files=[RUN/x/'samples.npz' for x in data.split(',')];items=[]
    for file in files:
        with np.load(file) as z:
            ix=z['splits']=='TRAIN';items.append({k:z[k][ix] for k in ('context','modes','local','targets','drafts','usable','valid')})
    d={k:np.concatenate([x[k] for x in items]) for k in items[0]};t={k:torch.as_tensor(v,device='cuda') for k,v in d.items()}
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    center,_=center_model();model=LocalDiffusion(center.base.mode_embedding.embedding_dim,seed).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    settings=dict(arm='local_diffusion',goal_tail=True,seed=seed,steps=steps,batch=32,lr=.0003,DDIM_steps=12,noise_steps=1000,data_sha256={str(f):sha(f) for f in files},base_sha256=sha(BASE),source_commit=os.environ.get('CODE_COMMIT'),scope='Adaptation of connected segment local diffusion: current observed triggers, same found-repair/identity targets. Cartesian control; not joint-space CDM reproduction')
    history=[];stream='';tic=time.monotonic();start=0;initial=tensor_state_digest(model.state_dict())
    def state(step):return dict(repair=model.state_dict(),center=None,optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['repair']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);history=ck['history'];stream=ck['stream'];start=ck['step'];initial=ck['initial_sha256']
    else:write(out/'config.json',settings)
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.integers(len(d['context']),size=32);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest();target=t['targets'][ix]-t['drafts'][ix]
        bound=torch.full_like(target[...,:1],.08);bound[:,:,18:]=.4;clean=(target/bound).clamp(-1,1)
        level=torch.randint(1000,(32,),device='cuda');a=model.alpha[level][:,None,None,None];noise=torch.randn_like(clean);noisy=a.sqrt()*clean+(1-a).sqrt()*noise
        predicted=model.epsilon(t['context'][ix],t['modes'][ix],t['local'][ix],noisy,level)
        weight=t['usable'][ix].float()*(3-2*t['valid'][ix].float());error=(predicted-noise).square().mean((-1,-2));loss=(weight*error).sum()/weight.sum().clamp_min(1)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:history.append(dict(step=step,loss=float(loss)));print(history[-1],flush=True);atomic_checkpoint(out/'recovery.pt',state(step))
        if step in (600,1200,2400) or step==steps:atomic_checkpoint(out/('step%d.pt'%step),state(step))
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));write(out/'SUMMARY.json',dict(settings=settings,seconds=time.monotonic()-tic,checkpoint_sha256=sha(out/'last.pt'),stream=stream,initial_sha256=initial,trainable_parameters=sum(p.numel() for p in model.parameters())))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--data',default='old_calibrated_targets_v1,interventions_calibrated_targets_v1');p.add_argument('--steps',type=int,default=2400);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);train(**vars(p.parse_args()))
