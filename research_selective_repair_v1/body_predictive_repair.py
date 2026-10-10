"""Bounded repair amplitudes using frozen predicted event measures.

Optimization operates inside the convex hull of the same three TRAIN recipes.
All candidate evaluations are charged. This is a conventional constrained
optimizer; it cannot turn a learned preservation prediction into a certificate.
"""
import numpy as np
import torch
from research_selective_repair_v1.body_options import options
from research_selective_repair_v1.body_feedback_data import WORDS
from research_selective_repair_v1.execution_semantics import word

def tensor_features(paths,completed,prefix=True):
    rel=(paths[:,:,None]-completed[:,None])/.1
    tangent=torch.cat([torch.zeros_like(paths[:,:1]),torch.diff(paths,dim=1)],1)/.05
    clock=torch.linspace(0,1,24,device=paths.device,dtype=paths.dtype)[None,:,None].expand(len(paths),-1,-1)
    goal=torch.zeros_like(paths) if prefix else ((paths[:,-1:]-paths[:,:1])/.3).expand(-1,24,-1)
    return torch.cat([(paths-paths[:,:1])/.3,tangent,rel.reshape(len(paths),24,12),torch.linalg.vector_norm(rel,dim=-1),clock,goal],-1)

def compose_tensor(p,cfg):
    occurrence=p['events'].sigmoid()
    prior=torch.cat([torch.ones_like(occurrence[:,:,:1]),torch.cumprod(1-occurrence,2)[:,:,:-1]],2)
    first=occurrence*prior;rows=[]
    for row in range(2):
        mean=p['mean'][...,row,:];scale=p['scale'][...,row,:]
        top=cfg['post_base_z']+cfg['post_heights'][row]+cfg['tip_clearance_m']
        below=torch.special.ndtr((top-mean[...,1])/scale[...,1])
        ys=cfg['post_y'][row];margin=.0175+cfg['tip_clearance_m']
        limits=[(-np.inf,ys[0]-margin),(ys[0]+margin,ys[1]-margin),(ys[1]+margin,np.inf)]
        mass=[]
        for lo,hi in limits:
            # Do not differentiate infinity/scale: PyTorch backward can create
            # inf*0. The corresponding CDF is a constant0 or1.
            upper=torch.ones_like(below) if np.isposinf(hi) else torch.special.ndtr((hi-mean[...,0])/scale[...,0])
            lower=torch.zeros_like(below) if np.isneginf(lo) else torch.special.ndtr((lo-mean[...,0])/scale[...,0])
            mass.append((first[...,row]*(upper-lower).clamp_min(0)*below).sum(-1))
        mass.append((first[...,row]*(1-below)).sum(-1));rows.append(torch.stack(mass,-1))
    joint=(rows[0][...,None]*rows[1][...,None,:]).flatten(-2)
    valid=p['log_weights'].exp()*(-p['hazard']).sigmoid().prod(-1)*p['clear'].sigmoid()
    positive=(valid[...,None]*joint).sum(1)
    return torch.cat([1-positive.sum(-1,keepdim=True),positive],-1)

def amplitude_caps(paths,completed,cfg):
    base=np.asarray(paths);basis=options(base,completed)[:,1:]-base[:,None]
    cap=np.ones(len(base));planned=[]
    for i,path in enumerate(base):
        label=word(path,cfg);planned.append(WORDS.index(label)+1 if label in WORDS else 0)
        if label is None:cap[i]=0;continue
        for r,x in enumerate(cfg['row_x']):
            if label.split('|')[r]=='over':continue
            hit=np.flatnonzero((path[:-1,0]<x)&(path[1:,0]>=x));assert len(hit)
            j=int(hit[0]);t=(x-path[j,0])/(path[j+1,0]-path[j,0])
            z=path[j,2]+t*(path[j+1,2]-path[j,2])
            delta=basis[i,0,j,2]+t*(basis[i,0,j+1,2]-basis[i,0,j,2])
            top=cfg['post_base_z']+cfg['post_heights'][r]+cfg['tip_clearance_m']
            if delta>1e-9:cap[i]=min(cap[i],max(0,(top-z-1e-6)/delta))
    return basis,cap,np.asarray(planned,np.int64)

def repair(model,ck,paths,completed,context,cfg,forecast='event',kind='actual',steps=32):
    assert forecast in ('event','binary') and kind in ('actual','planned','success')
    assert steps==32,'Frozen pilot budget; no optimizer sweep'
    basis,cap,planned=amplitude_caps(paths,completed,cfg)
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    t=lambda v:torch.tensor(v,device=device,dtype=dtype)
    base=t(paths);delta=t(basis);completed=t(np.broadcast_to(completed,(8,4,3)));context=t(np.broadcast_to(context,(8,128)))
    mean=t(ck['settings']['mean']);std=t(ck['settings']['std']);cap=t(cap)
    # Risk-only controls receive the identical geometry/basis/budget/optimizer.
    onehot=torch.nn.functional.one_hot(torch.tensor(planned,device=device),17).to(dtype)
    amplitudes=torch.zeros((8,2),device=device,dtype=dtype,requires_grad=True)
    opt=torch.optim.Adam([amplitudes],lr=.05);original=None;best_score=None;best=amplitudes.detach().clone();best_prob=None;feasible_states=0
    def probability(path):
        x=(tensor_features(path,completed,prefix=forecast=='event')-mean)/std
        if forecast=='event':p=compose_tensor(model(x,context,path),cfg)
        else:
            success=model(x,context).flatten().sigmoid();p=success[:,None]*onehot
            p=torch.cat([p[:,:1]+1-success[:,None],p[:,1:]],-1)
        if kind=='planned':
            success=1-p[:,0];p=success[:,None]*onehot
            p=torch.cat([p[:,:1]+1-success[:,None],p[:,1:]],-1)
        return p
    for step in range(steps+1):
        path=base+(amplitudes[:,:,None,None]*delta).sum(1)
        prob=probability(path);cov=1-(1-prob[:,1:]).prod(0)
        if original is None:original=(prob.detach(),cov.detach())
        eligible=bool((cov>=original[1]-.01).all() and (prob[:,0]<=original[0][:,0]+.01).all())
        score=(1-prob[:,0]).sum() if kind=='success' else cov.sum()+.05*(1-prob[:,0]).sum()
        score=score-.01*amplitudes.sum()
        if eligible:
            feasible_states+=1
            if best_score is None or float(score.detach())>best_score:
                best_score=float(score.detach());best=amplitudes.detach().clone();best_prob=prob.detach().clone()
        if step==steps:break
        opt.zero_grad();(-score).backward()
        assert torch.isfinite(amplitudes.grad).all()
        opt.step()
        with torch.no_grad():
            amplitudes[:,0].clamp_(min=0);amplitudes[:,0].copy_(torch.minimum(amplitudes[:,0],cap))
            amplitudes[:,1].clamp_(min=0);amplitudes[:,1].copy_(torch.minimum(amplitudes[:,1],1-amplitudes[:,0]))
    result=(base+(best[:,:,None,None]*delta).sum(1)).detach().cpu().numpy()
    assert np.array_equal(result[:,[0,-1]],np.asarray(paths)[:,[0,-1]])
    assert [word(p,cfg) for p in result]==[word(p,cfg) for p in paths]
    audit=np.stack([original[0].cpu().numpy(),best_prob.cpu().numpy()],1)
    return result,best.cpu().numpy(),audit,dict(internal_route_forwards=8*(steps+1),optimizer_steps=steps,
        internal_recipe_basis_curves=24,final_candidates=8,feasible_evaluated_states=feasible_states,
        current_NN_signature_calls=16,
        original_predicted_feasible_included=True,actual_controller_queries=0,
        Gaussian_CDF_evaluations=8*(steps+1)*4*23*2*7 if forecast=='event' else 0,
        scope='Predicted nondegradation only; current NN geometry caps; convex interpolation of same TRAIN recipes,not physical certificate')
