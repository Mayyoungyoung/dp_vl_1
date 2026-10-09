"""Matched real-draft supervision and actual RNG/optimizer recovery."""
import argparse,hashlib,os,random,time
import numpy as np
import torch
from research_selective_repair_v1.core import *
from research_selective_repair_v1.model import RepairHead
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest
from routeset.segment_clearance import path_segment_clearances

def train(name,arm,seed=0,steps=2400,data='prepared_v1',resume=False,stop_after=None,goal_tail=False):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed training')
    files=[RUN/n/'samples.npz' for n in data.split(',')]
    datasets=[]
    for file in files:
        with np.load(file) as z:
            ix=z['splits']=='TRAIN';datasets.append({k:z[k][ix] for k in z.files})
    d={k:np.concatenate([x[k] for x in datasets]) for k in datasets[0]}
    assert set(d['splits'])=={'TRAIN'}
    torch.manual_seed(seed);random.seed(seed);np.random.seed(seed);rng=np.random.default_rng(seed)
    center,baseck=center_model();head=RepairHead(center.base.mode_embedding.embedding_dim,arm,goal_tail).cuda()
    if arm=='recenter':
        center.corridor_blocks.requires_grad_(True);center.corridor_output.requires_grad_(True)
    trainable=list(head.parameters())+[p for p in center.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(trainable,lr=.0003,weight_decay=.0001)
    t={k:torch.as_tensor(v,device='cuda') for k,v in d.items() if v.dtype.kind not in 'USO'}
    settings=dict(arm=arm,seed=seed,steps=steps,batch=32,lr=.0003,goal_tail=goal_tail,data_sha256={str(f):sha(f) for f in files},base_sha256=sha(BASE),source_commit=os.environ.get('CODE_COMMIT'),loss='real-draft found repair + valid identity + support BCE; equal continuous geometry supervision on usable routes')
    history=[];stream='';start=0;initial=tensor_state_digest(head.state_dict());tic=time.monotonic()
    def state(step):return dict(repair=head.state_dict(),center=center.state_dict() if arm=='recenter' else None,optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        head.load_state_dict(ck['repair']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng)
        if ck['center'] is not None:center.load_state_dict(ck['center'])
        start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.integers(len(d['ids']),size=32);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        drafts=t['drafts'][ix]
        if arm=='recenter':drafts=center.corridor(t['context'][ix],t['anchor'][ix],t['current'][ix],t['modes'][ix],torch.zeros_like(t['modes'][ix]))[0]
        path,info=head(t['context'][ix],t['modes'][ix],drafts,t['local'][ix])
        usable=t['usable'][ix].float();valid=t['valid'][ix].float();weight=usable*(3-2*valid)
        error=(path-t['targets'][ix]).square().mean((-1,-2))
        route=(weight*error).sum()/weight.sum().clamp_min(1)
        bce=torch.nn.functional.binary_cross_entropy_with_logits(info['logits'],t['support'][ix],reduction='none').mean(-1)
        gate=(bce*usable).sum()/usable.sum().clamp_min(1)
        clear=path_segment_clearances(path,t['obstacle_centers'][ix],t['obstacle_halves'][ix])
        geom=(.021-clear).clamp_min(0).square().mean(-1)+(t['floors'][ix,None,None]-path[...,2]).clamp_min(0).square().mean(-1)
        geometry=(geom*usable).sum()/usable.sum().clamp_min(1)
        loss=1000*route+160*geometry+(.05*gate if arm=='selective' else 0)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(trainable,1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss),route=float(route),geometry=float(geometry),gate=float(gate)));print(history[-1],flush=True)
            atomic_checkpoint(out/'recovery.pt',state(step))
        if step in (600,1200) or step==steps:atomic_checkpoint(out/('step%d.pt'%step),state(step))
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));write(out/'SUMMARY.json',dict(settings=settings,stream=stream,initial_sha256=initial,seconds=time.monotonic()-tic,trainable_parameters=sum(p.numel() for p in trainable),checkpoint_sha256=sha(out/'last.pt')))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--arm',choices=['selective','residual','recenter'],required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2400);p.add_argument('--data',default='prepared_v1');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--goal-tail',action='store_true');train(**vars(p.parse_args()))
