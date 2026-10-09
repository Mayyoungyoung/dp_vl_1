"""Identical frozen-snapshot pool for simple success and net-set utility heads."""
import argparse
import hashlib
import os
import random
import time
import numpy as np
import torch
from torch.nn import functional as F
from research_realized_coverage_v1.core import RUN,read,write,sha,torch_setup
from research_realized_coverage_v1.allocator import SuccessHead,SetUtility,make_allocator
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng

def pack(folder):
    summary=read(folder/'SUMMARY.json');assert summary['manifest']['split']=='TRAIN'
    values={k:[] for k in ('context','modes','variants','utility','success','actual','base_m','base_v','base_u','gain','added_gain')};ids=[]
    for row in summary['files']:
        f=folder/(row['id']+'.npz');assert sha(f)==row['file_sha256']
        with np.load(f) as z:
            n=len(z['modes']);values['context'].extend(np.repeat(z['context'][None],n,0))
            for key in ('modes','variants','utility','gain'):values[key].extend(z[key])
            values['success'].extend((z['valid']&(z['words']==z['modes'])).astype(np.float32))
            values['actual'].extend(np.where(z['valid']&(z['words']>=0),z['words'],16))
            added=z['gain'].copy();added[:,0]=z['added'].sum(-1);values['added_gain'].extend(added)
            for key,source in [('base_m','modes'),('base_v','variants'),('base_u','utility')]:values[key].extend(np.repeat(z[source][:1],n,0))
            ids.extend([row['id']]*n)
    return {k:np.asarray(v) for k,v in values.items()},ids,summary['manifest']

def train(kind,name,feedback,seed,steps,resume=False,stop_after=None):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    assert not (out/'last.pt').exists()
    packs=[pack(RUN/f) for f in feedback.split(',')]
    d={k:np.concatenate([a[k] for a,_,_ in packs]) for k in packs[0][0]};ids=sum([b for _,b,_ in packs],[]);manifest=packs[0][2]
    assert all(c['generator_sha256']==manifest['generator_sha256'] for _,_,c in packs)
    rng=np.random.default_rng(seed)
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
    model=make_allocator(kind).cuda()
    data={k:torch.as_tensor(v,device='cuda') for k,v in d.items()}
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    settings=dict(kind=kind,feedback=feedback,feedback_sha256={f:sha(RUN/f/'SUMMARY.json') for f in feedback.split(',')},generator_sha256=manifest['generator_sha256'],
        seed=seed,steps=steps,batch_size=128,lr=.0003,source_commit=os.environ.get('CODE_COMMIT'))
    history=[];stream='';start=0;tic=time.monotonic()
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream)
    if resume:
        c=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert c['settings']==settings
        model.load_state_dict(c['model']);opt.load_state_dict(c['optimizer']);restore_rng(c['rng'],rng);start=c['step'];history=c['history'];stream=c['stream']
    else:write(out/'config.json',settings)
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.integers(len(ids),size=128);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        if kind=='success':
            logits=model(data['context'][ix]).gather(1,data['modes'][ix]);loss=F.binary_cross_entropy_with_logits(logits,data['success'][ix])
        else:
            pred=model(data['context'][ix],data['modes'][ix],data['variants'][ix])
            base=model(data['context'][ix],data['base_m'][ix],data['base_v'][ix])
            # Equal count-unit weights: all four outcomes matter, no oracle gradient.
            gain=data['added_gain'][ix] if kind=='added_only' else data['gain'][ix]
            utility=data['base_u'][ix]+gain if kind=='added_only' else data['utility'][ix]
            loss=F.mse_loss(pred,utility)+F.mse_loss(pred-base,gain)
            if kind in ('dense','no_peer'):
                _,logits=model.components(data['context'][ix],data['modes'][ix],data['variants'][ix])
                loss=loss+F.cross_entropy(logits.reshape(-1,17),data['actual'][ix].reshape(-1))
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss)));print(history[-1],flush=True)
        if step in (400,1200,2400) or step==steps:atomic_checkpoint(out/('step%d.pt'%step),state(step))
        if step%100==0 or step==min(steps,stop_after or steps):atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));write(out/'SUMMARY.json',dict(settings=settings,elapsed_seconds=time.monotonic()-tic,
            records=len(ids),parameters=sum(p.numel() for p in model.parameters()),last_sha256=sha(out/'last.pt'),stream=stream))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['success','net','dense','no_peer','added_only'],required=True);p.add_argument('--name',required=True)
    p.add_argument('--feedback',default='feedback_C_train');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2400)
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args();train(**vars(a))
