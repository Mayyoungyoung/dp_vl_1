"""Current RGB-D integrated decoder for matched feedback and online checks.

No stored old draft,truth geometry or controller query enters this module.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from research_selective_repair_v1.core import *
from research_selective_repair_v1.local import observed_points
from research_selective_repair_v1.constraints import load as load_boxes,summaries,boxes
from research_selective_repair_v1.constraint_operator import repair,configuration
from research_selective_repair_v1.body_forecast import load as load_outcomes,predict,WORDS
from research_selective_repair_v1.body_options import options
from research_selective_repair_v1.body_allocation import allocate
from research_selective_repair_v1.grounding import predict as goal_proposal
from scripts.research_v3_audit import mode
from routeset.train_v2 import atomic_checkpoint

class IntegratedBody(torch.nn.Module):
    def __init__(self,center,completion,completion_ck,outcome,outcome_ck,kind,protected=True):
        super().__init__();self.center=center;self.completion=completion;self.outcome=outcome
        self.completion_ck=completion_ck;self.outcome_ck=outcome_ck;self.kind=kind;self.protected=protected
        self.goal=None;self.query_counts=[]
    @property
    def mode_predictor(self):return self.center.mode_predictor
    def encode(self,**inp):
        self.inp=inp;points=observed_points(inp)
        self.visible=points[None];self.visible_mask=np.ones((1,len(points)),bool)
        return self.center.encode(**inp)
    def decode(self,context,anchor,current,mode_ids,variant_ids=None):
        drafts,events,info=self.center.decode(context,anchor,current,mode_ids,torch.zeros_like(mode_ids))
        n=len(context);settings=self.completion_ck['settings']
        x,a=summaries(self.visible,self.visible_mask,np.asarray(settings['priors']))
        t=lambda v:torch.tensor(v,device=drafts.device,dtype=drafts.dtype)
        completed=self.completion(context,t(np.repeat(x,n,0)),t(np.repeat(a,n,0)))
        centers,halves=boxes(completed,settings)
        goal=t(self.goal)[None].expand(n,-1) if self.goal is not None else anchor
        available=torch.full((n,),self.goal is not None,device=drafts.device,dtype=drafts.dtype)
        # Callers often wrap all decode in inference_mode. Adam requires normal
        # tensors and explicit local gradients,including crossing/floor checks.
        with torch.inference_mode(False),torch.enable_grad():
            p,geo=repair(drafts.detach().clone(),mode_ids.detach().clone(),goal.detach().clone(),available.clone(),centers.detach().clone(),halves.detach().clone(),settings,'global')
        paths=[]
        for i in range(n):
            c=completed[i].detach().cpu().numpy();candidate=options(p[i].cpu().numpy(),c)
            probabilities=np.zeros((8,3,17),np.float32);probabilities[:,:,0]=1
            static=self.kind in ('identity','lift','preserved')
            if not static:
                probabilities=predict(self.outcome,self.outcome_ck,candidate.reshape(24,24,3),np.broadcast_to(c,(24,4,3)),np.broadcast_to(context[i].detach().cpu().numpy(),(24,128))).reshape(8,3,17)
            planned=None
            if self.kind=='planned':
                cfg=configuration(centers[i].detach().cpu().numpy(),halves[i].detach().cpu().numpy(),settings)
                words=[mode(q,cfg) for q in candidate.reshape(24,24,3)];planned=np.array([WORDS.index(w)+1 if w in WORDS else 0 for w in words]).reshape(8,3)
            choice,count=allocate(probabilities,self.kind,self.protected,planned)
            paths.append(candidate[np.arange(8),choice]);self.query_counts.append(dict(final_candidates=8,internal_options=24,critic_route_forwards=0 if static else 24,allocation_objectives=count,geometry_steps=geo['steps']))
        return t(np.asarray(paths)),events,info

def bind(name,outcome,completion,prototype,kind='actual',protected=True):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    view=dict(center_path=str(BASE),center_sha256=sha(BASE),completion_path=str(completion),completion_sha256=sha(completion),outcome_path=str(outcome),outcome_sha256=sha(outcome),prototype_path=str(prototype),prototype_sha256=sha(prototype),kind=kind,protected=protected,internal_options=24,final_candidates=8,geometry_steps=64)
    atomic_checkpoint(out/'last.pt',dict(body_view=view));write(out/'VIEW.json',dict(**view,decoder_sha256=sha(out/'last.pt')))

def load_view(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);v=ck['body_view']
    for name in ('center','completion','outcome','prototype'):
        assert sha(Path(v[name+'_path']))==v[name+'_sha256'],'Frozen body decoder component changed:'+name
    center,_=center_model();completion,c=load_boxes(Path(v['completion_path']));outcome,o=load_outcomes(Path(v['outcome_path']))
    return IntegratedBody(center,completion,c,outcome,o,v['kind'],v['protected']).cuda().eval(),ck

def feedback(name,checkpoint,data,limit=None,resume=False):
    if limit is None:
        raise RuntimeError('Pilot forecast has target0-only TRAIN feedback. Expand actual controller feedback to all goals before formal matching-head collection; only bounded API checks are enabled now.')
    from research_realized_coverage_v1 import feedback as original
    from scripts import observation_prototype_grounding as proto
    model=None;v=torch.load(checkpoint,map_location='cpu',weights_only=False)['body_view'];pm=read(v['prototype_path'])
    def load(path):
        nonlocal model
        model,ck=load_view(path);return model,ck
    old_inputs=original.inputs_for
    def inputs(row,label,cache,torch,hashes):
        inp=old_inputs(row,label,cache,torch,hashes)
        (rgb,xyz,valid),_=proto.load_observation(data/'export',row,label['observation'])
        model.goal,_=goal_proposal(rgb,xyz,valid,row['instruction'],pm)
        return inp
    original.RUN=RUN;original.DATA=data;original.load_generator=load;original.inputs_for=inputs
    original.collect(name,'TRAIN',resume=resume,limit=limit,checkpoint=checkpoint)
    write(RUN/name/'BODY_COUNTS.json',dict(decode_counts=model.query_counts,current_observation_only=True,scope='Bounded TRAIN API check only; target0-only outcome pilot cannot yet support formal all-goal matching heads',set_dependent_companion_edits_possible=True,decoder_sha256=sha(checkpoint)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--outcome',type=Path);p.add_argument('--completion',type=Path);p.add_argument('--prototype',type=Path);p.add_argument('--kind',default='actual');p.add_argument('--unprotected',dest='protected',action='store_false');p.add_argument('--feedback',action='store_true');p.add_argument('--checkpoint',type=Path);p.add_argument('--data',type=Path);p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if a.feedback:feedback(a.name,a.checkpoint,a.data,a.limit,a.resume)
    else:bind(a.name,a.outcome,a.completion,a.prototype,a.kind,a.protected)
