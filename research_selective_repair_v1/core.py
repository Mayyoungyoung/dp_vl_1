from pathlib import Path
import json
import numpy as np
import torch
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha,lines,torch_setup
from research_realized_coverage_v1.core import Q,DATA,VOCAB,assess,SceneRunner
from scripts.evaluate_paired_modes import inputs_for,references,check_candidates
from research_feasible_space_v1.model import load_model
from research_realized_coverage_v1.allocator import SuccessHead,allocate

RUN=ROOT/'runs/selective_repair_v1'
POLICY=SOURCE/'configs/selective_repair_v1.json'
BASE=ROOT/read(POLICY)['base']
BASE_HEAD=ROOT/read(POLICY)['base_head']

def center_model():
    model,ck=load_model(BASE)
    # Skip both unused interior blocks and their output; this is the strong B0.
    def decode(context,anchor,current,mode_ids,variant_ids=None,**unused):
        if mode_ids.shape!=(len(context),8):raise ValueError('Eight drafts only')
        if variant_ids is None:variant_ids=torch.zeros_like(mode_ids)
        c,r,e,t=model.corridor(context,anchor,current,mode_ids,variant_ids)
        return c,e,dict(centers=c,radii=r)
    model.decode=decode
    model.requires_grad_(False)
    return model,ck

def base_queries(context,runner,head):
    m=torch.tensor(runner.base[None],device='cuda');v=torch.zeros_like(m)
    with torch.no_grad():m,v,_=allocate('success',head,context,m,v)
    return m,v.zero_()

def base_success():
    ck=torch.load(BASE_HEAD,map_location='cpu',weights_only=False)
    assert ck['settings']['generator_sha256']==sha(BASE)
    h=SuccessHead().cuda();h.load_state_dict(ck['model']);return h.eval()

def masks(scores,threshold=.5,max_nodes=12):
    """One contiguous interior edit window; no selected point means exact identity."""
    shape=scores.shape;s=scores.reshape(-1,24)
    result=torch.zeros_like(s)
    for i in range(len(s)):
        active=torch.nonzero(s[i,1:-1]>=threshold).flatten()+1
        if len(active):
            lo,hi=int(active[0]),int(active[-1])+1
            if hi-lo>max_nodes:
                sums=s[i,1:-1].unfold(0,max_nodes,1).sum(-1)
                lo=int(sums.argmax())+1;hi=lo+max_nodes
            result[i,lo:hi]=1
    return result.reshape(shape)

def wordset(paths,valid,cfg):
    from scripts.research_v3_audit import mode
    return {mode(p,cfg) for p,v in zip(paths,valid) if v}-{None}

def plain(v):
    if isinstance(v,np.ndarray):return v.tolist()
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,dict):return {str(k):plain(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [plain(x) for x in v]
    return v
