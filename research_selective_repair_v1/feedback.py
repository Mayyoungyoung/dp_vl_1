"""Bind local repair to every current query, then reuse matched TRAIN feedback."""
import argparse
import numpy as np
import torch
from research_selective_repair_v1.core import *
from research_selective_repair_v1.model import load_repair
from research_selective_repair_v1.local import features,observed_points
from research_selective_repair_v1.grounding import predict
from research_realized_coverage_v1 import feedback as oldfeedback,train_allocation

class IntegratedRepair(torch.nn.Module):
    def __init__(self,center,repair,prototype,threshold=.5,scale=1.):
        super().__init__();self.center=center;self.repair=repair;self.prototype=prototype;self.threshold=threshold;self.scale=scale
    @property
    def mode_predictor(self):return self.center.mode_predictor
    def encode(self,**inp):
        self.inp=inp;self.points=observed_points(inp)
        return self.center.encode(**inp)
    def decode(self,context,anchor,current,mode_ids,variant_ids=None):
        drafts,events,info=self.center.decode(context,anchor,current,mode_ids,torch.zeros_like(mode_ids))
        p=drafts.detach().cpu().numpy();n=len(p)
        local=features(p.reshape(-1,24,3),self.points,self.observation).reshape(n,8,24,29)
        if self.repair.goal_tail:
            goal=self.goal if self.goal is not None else anchor[0].detach().cpu().numpy()
            relative=np.broadcast_to(((goal-p[:,:,-1])/.4)[:,:,None],(n,8,24,3))
            extra=np.concatenate([relative,np.full((n,8,24,1),self.goal is not None)],-1)
            local=np.concatenate([local,extra],-1)
        path,_=self.repair(context,mode_ids,drafts,torch.as_tensor(local,dtype=drafts.dtype,device=drafts.device),hard=True,threshold=self.threshold,scale=self.scale)
        return path,events,info

def collect(name,checkpoint,prototype,threshold=.5,scale=1.):
    model=None;pm=read(prototype)
    def load(path):
        nonlocal model
        center,repair,ck=load_repair(path)
        if ck.get('view'):
            assert ck['view']['prototype_sha256']==sha(prototype) and ck['view']['threshold']==threshold and ck['view']['scale']==scale
        model=IntegratedRepair(center,repair,pm,threshold,scale).cuda().eval();return model,ck
    original=oldfeedback.inputs_for
    def inputs(row,label,cache,torch,hashes):
        from scripts import observation_prototype_grounding as proto
        inp=original(row,label,cache,torch,hashes)
        with np.load(label['observation']) as z:model.observation={k:z[k] for k in ('depth','camera_intrinsics','camera_extrinsics')}
        (rgb,xyz,valid),_=proto.load_observation(DATA/'export',row,label['observation'])
        model.goal,_=predict(rgb,xyz,valid,row['instruction'],pm)
        return inp
    oldfeedback.RUN=RUN;oldfeedback.load_generator=load;oldfeedback.inputs_for=inputs
    oldfeedback.collect(name,'TRAIN',checkpoint=checkpoint)
    write(RUN/name/'REPAIR_VIEW.json',dict(threshold=threshold,scale=scale,prototype_sha256=sha(prototype),checkpoint_sha256=sha(checkpoint),current_observation_only=True))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--prototype',type=Path);p.add_argument('--pool');p.add_argument('--seed',type=int,default=0);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--scale',type=float,default=1.);a=p.parse_args()
    if a.checkpoint:collect(a.name,a.checkpoint,a.prototype,a.threshold,a.scale)
    else:train_allocation.RUN=RUN;train_allocation.train('success',a.name,a.pool,a.seed,2400)
