"""C continuation without displacement; matched ordinary and deployment-gap queries."""
import argparse
import hashlib
import os
import random
import time
import numpy as np
import torch
from research_realized_coverage_v1.core import RUN,BASE,PREP,VOCAB,load_generator,read,write,sha,torch_setup,variants
from research_realized_coverage_v1.allocator import SetUtility,SuccessHead,allocate,make_allocator
from scripts.mode_geometry_experiment import POOL
from routeset.mode_geometry import mode_loss,shared_allocation_loss
from routeset.segment_clearance import segment_clearance_loss
from routeset.paired_modes import workspace_floor_loss
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest

def train(name,arm,steps,seed,head=None,feedback='feedback_C_train',checkpoint=BASE,resume=False,stop_after=None):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=resume);assert not (out/'last.pt').exists()
    with np.load(PREP/'train.npz') as z:d={k:z[k] for k in z.files}
    with np.load(POOL) as z:s={k:z[k] for k in z.files}
    assert np.array_equal(d['ids'],s['ids']) and set(s['splits'])=={'TRAIN'}
    bm=[{VOCAB.index(w):np.flatnonzero(s['mask'][i]&(s['modes'][i]==w)) for w in sorted(set(s['modes'][i,s['mask'][i]]))} for i in range(len(d['ids']))]
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    model,ck=load_generator(checkpoint);model.train()
    t={k:torch.as_tensor(d[k],device='cuda') for k in ('context','anchor','current','evidence','centers','halves','floors')}
    proposals=[];pv=[];failures=[]
    if arm in ('gap','hard'):
        hs=torch.load(RUN/head,map_location='cpu',weights_only=False)
        assert hs['settings']['generator_sha256']==sha(checkpoint)
        allocator=make_allocator(hs['settings']['kind']).cuda();allocator.load_state_dict(hs['model']);allocator.eval()
        summary=read(RUN/feedback/'SUMMARY.json');assert summary['manifest']['split']=='TRAIN' and summary['manifest']['generator_sha256']==sha(checkpoint)
        for i,ident in enumerate(d['ids']):
            with np.load(RUN/feedback/(str(ident)+'.npz')) as z:
                base=z['modes'][0];vv=z['variants'][0]
                # All generated failures are realization outcomes, never physical negatives.
                rate={m:float((z['valid']&(z['words']==m)&(z['modes']==m)).sum()/max(1,(z['modes']==m).sum())) for m in bm[i]}
            with torch.no_grad():
                m,v,_=allocate(hs['settings']['kind'],allocator,t['context'][i:i+1],torch.as_tensor(base[None],device='cuda'),torch.as_tensor(vv[None],device='cuda'))
            proposals.append(m[0].cpu().numpy());pv.append(v[0].cpu().numpy());failures.append(rate)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0001)
    settings=dict(arm=arm,steps=steps,seed=seed,head=head,feedback=feedback,initial_sha256=sha(checkpoint),lr=.0001,
        prepared_sha256=sha(PREP/'train.npz'),support_sha256=sha(POOL),head_sha256=sha(RUN/head) if head else None,
        source_commit=os.environ.get('CODE_COMMIT'),displacement_weight=0.,route_normalization='Mean over witnessed query routes and coordinates; clearance/event unchanged')
    initial=tensor_state_digest(model.state_dict());frozen=tensor_state_digest({k:v for k,v in model.state_dict().items() if k.startswith(('geometry.','head.feature_encoder.','head.state_encoder.'))})
    history=[];stream='';start=0;tic=time.monotonic()
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,
        config=ck['config'],history=history,stream=stream,initial_tensor_sha256=initial,frozen_sha256=frozen)
    if resume:
        r=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert r['settings']==settings
        model.load_state_dict(r['model']);opt.load_state_dict(r['optimizer']);restore_rng(r['rng'],rng);history=r['history'];stream=r['stream'];start=r['step']
    else:write(out/'config.json',settings)
    for step in range(start+1,min(steps,stop_after or steps)+1):
        chosen=rng.integers(len(d['pairs']),size=16);ids=d['pairs'][chosen].reshape(-1);modes=[];vs=[];targets=[];masks=[]
        for j in chosen:
            common=rng.permutation(np.flatnonzero(d['pair_mask'][j]))[:6]
            for side,i in enumerate(d['pairs'][j]):
                extra=rng.permutation(sorted(set(bm[i])-set(common)))[:8-len(common)].tolist()
                while len(common)+len(extra)<8:extra.append(int(rng.choice(list(bm[i]))))
                conventional=np.r_[common,extra].astype(np.int64)
                # Half the draws preserve broad conventional coverage. Other half align
                # deployment context, injecting one witnessed failure with controlled mass.
                if arm in ('gap','hard') and step%2==0:
                    m=np.array(proposals[i]);v=np.array(pv[i]);words=list(bm[i]);weights=np.array([.1+1-failures[i][w] for w in words]);weights/=weights.sum()
                    forced=int(rng.choice(words,p=weights));slot=int(rng.integers(8));m[slot]=forced;v[slot]=0
                    if arm=='hard':
                        # Same request/forced word pressure, ordinary positive companions.
                        m=conventional.copy();m[slot]=forced;v=variants(m)
                else:m=conventional;v=variants(m)
                targets_i=[];mask=[]
                for k in m:
                    eligible=k in bm[i];mask.append(eligible)
                    targets_i.append(s['paths'][i,int(rng.choice(bm[i][k]))] if eligible else np.zeros((24,3),np.float32))
                modes.append(m);vs.append(v);targets.append(targets_i);masks.append(mask)
        m=np.array(modes);v=np.array(vs);target=np.array(targets,np.float32);mask=np.array(masks,np.float32)
        stream=hashlib.sha256(stream.encode()+ids.tobytes()+m.tobytes()+v.tobytes()+target.tobytes()).hexdigest()
        p,e,info=model.decode(t['context'][ids],t['anchor'][ids],t['current'][ids],torch.as_tensor(m,device='cuda'),variant_ids=torch.as_tensor(v,device='cuda'))
        weight=torch.as_tensor(mask,device='cuda');tx=torch.as_tensor(target,device='cuda')
        route=((p[:,:,1:]-tx[:,:,1:]).square().mean((-1,-2))*weight).sum()/weight.sum().clamp_min(1)
        clear=segment_clearance_loss(p,t['centers'][ids],t['halves'][ids])+workspace_floor_loss(p,t['floors'][ids])
        event=.01*(e-t['current'][ids,None,None,7]).square().mean();ml=mode_loss(info['mode_logits'],t['evidence'][ids])
        kl=shared_allocation_loss(info['mode_logits'],t['evidence'][ids]) if arm=='kl_only' else route*0
        loss=route+160*clear+event+.001*(ml+kl)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:history.append(dict(step=step,loss=float(loss),route=float(route),clear=float(clear),kl=float(kl)));print(history[-1],flush=True)
        if step in (600,1800,3600) or step==steps:atomic_checkpoint(out/('step%d.pt'%step),state(step))
        if step%100==0 or step==min(steps,stop_after or steps):atomic_checkpoint(out/'recovery.pt',state(step))
    assert frozen==tensor_state_digest({k:v for k,v in model.state_dict().items() if k.startswith(('geometry.','head.feature_encoder.','head.state_encoder.'))})
    if step==steps:atomic_checkpoint(out/'last.pt',state(step));write(out/'SUMMARY.json',dict(settings=settings,elapsed_seconds=time.monotonic()-tic,last_sha256=sha(out/'last.pt'),initial_tensor_sha256=initial,stream=stream))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--arm',choices=['ordinary','kl_only','gap','hard'],required=True)
    p.add_argument('--steps',type=int,default=3600);p.add_argument('--seed',type=int,default=0);p.add_argument('--head');p.add_argument('--feedback',default='feedback_C_train')
    p.add_argument('--checkpoint',type=type(BASE),default=BASE);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=p.parse_args();train(**vars(a))
