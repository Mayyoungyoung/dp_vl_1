"""One frozen TRAIN test of editable support versus inadequate lift-only span."""
import numpy as np
import torch
from research_selective_repair_v1.body_predictive_repair import tensor_features,input_gradient_mode
from research_selective_repair_v1.body_crossing_measure import compose_tensor
from research_selective_repair_v1.body_feedback_data import WORDS
from research_selective_repair_v1.execution_semantics import word,tip_clear

def support_from_stop(p,width=12):
    continuation=(-p['hazard']).sigmoid()
    preceding=torch.cat([torch.ones_like(continuation[:,:,:1]),continuation.cumprod(-1)[:,:,:-1]],-1)
    stop=(p['log_weights'].exp()[...,None]*preceding*(1-continuation)).sum(1)
    mask=torch.zeros((len(stop),24),device=stop.device,dtype=stop.dtype)
    # Node windows touch both adjacent segment endpoints. Starts/goals fixed.
    scores=torch.stack([stop[:,max(0,j-1):min(23,j+width)].sum(-1) for j in range(1,24-width)],-1)
    begin=scores.argmax(-1)+1
    for i,j in enumerate(begin.tolist()):mask[i,j:j+width]=1
    return mask,stop

def repair(model,ck,paths,completed,context,cfg,scope='local',forecast='crossing',kind='actual'):
    assert scope in ('local','global') and forecast in ('crossing','binary') and kind in ('actual','planned')
    assert scope=='global' or forecast=='crossing'
    with input_gradient_mode(model):return _repair(model,ck,paths,completed,context,cfg,scope,forecast,kind)

def _repair(model,ck,paths,completed,context,cfg,scope,forecast,kind):
    device=next(model.parameters()).device;dtype=next(model.parameters()).dtype
    t=lambda v:torch.tensor(v,device=device,dtype=dtype)
    base=t(paths);posts=t(np.broadcast_to(completed,(8,4,3)));ctx=t(np.broadcast_to(context,(8,128)))
    mean=t(ck['settings']['mean']);std=t(ck['settings']['std'])
    initial_words=[word(p,cfg) for p in paths];initial_clear=[tip_clear(p,cfg) for p in paths]
    typed=np.asarray([WORDS.index(w)+1 if w in WORDS else 0 for w in initial_words])
    onehot=torch.nn.functional.one_hot(torch.tensor(typed,device=device),17).to(dtype)
    mask=torch.ones((8,24),device=device,dtype=dtype);mask[:,[0,-1]]=0
    def prediction(path):
        x=(tensor_features(path,posts,prefix=forecast=='crossing')-mean)/std
        if forecast=='crossing':
            p=model(x,ctx,path,posts);prob=compose_tensor(p,cfg)
        else:
            p=None;success=model(x,ctx).flatten().sigmoid();positive=success[:,None]*onehot[:,1:]
            prob=torch.cat([1-positive.sum(-1,keepdim=True),positive],-1)
        if kind=='planned':
            success=1-prob[:,0];positive=success[:,None]*onehot[:,1:]
            prob=torch.cat([1-positive.sum(-1,keepdim=True),positive],-1)
        return prob,p
    initial,p=prediction(base)
    stop=None
    if scope=='local':mask,stop=support_from_stop(p)
    original=(initial.detach(),(1-(1-initial[:,1:]).prod(0)).detach())
    offset=torch.zeros_like(base,requires_grad=True);opt=torch.optim.Adam([offset],lr=.004)
    best=offset.detach().clone();best_prob=initial.detach().clone();best_score=None;feasible=0;protection_feasible=0;geometry_feasible=0
    for step in range(33):
        path=base+offset*mask[...,None];prob,_=prediction(path);coverage=1-(1-prob[:,1:]).prod(0)
        # Model protection and inferred geometric protection are distinct.
        predicted=bool((coverage>=original[1]-.01).all() and (prob[:,0]<=original[0][:,0]+.01).all())
        current=path.detach().cpu().numpy();words=[word(v,cfg) for v in current];clear=[tip_clear(v,cfg) for v in current]
        geometric=all((not initial_clear[i] or clear[i]) and (initial_words[i] is None or words[i]==initial_words[i]) for i in range(8))
        protection_feasible+=predicted;geometry_feasible+=geometric
        score=coverage.sum()+.05*(1-prob[:,0]).sum()-.01*((offset*mask[...,None])/.08).square().mean()
        if predicted and geometric:
            feasible+=1
            if best_score is None or float(score.detach())>best_score:
                best_score=float(score.detach());best=offset.detach().clone();best_prob=prob.detach().clone()
        if step==32:break
        opt.zero_grad();(-score).backward();assert torch.isfinite(offset.grad).all();opt.step()
        with torch.no_grad():offset.clamp_(-.08,.08);offset.mul_(mask[...,None])
    result=(base+best*mask[...,None]).detach().cpu().numpy()
    np.testing.assert_array_equal(result[:,[0,-1]],np.asarray(paths)[:,[0,-1]])
    np.testing.assert_array_equal(result[mask.cpu().numpy()==0],np.asarray(paths)[mask.cpu().numpy()==0])
    analytic=forecast=='crossing' and ck['settings']['readout']=='analytic'
    return result,best.cpu().numpy(),np.stack([initial.detach().cpu().numpy(),best_prob.cpu().numpy()],1),dict(
        support_scope=scope,editable_nodes_per_route=mask.sum(-1).int().cpu().tolist(),support_mask=mask.int().cpu().tolist(),
        initial_first_stop_mass=stop.detach().cpu().numpy().tolist() if stop is not None else None,
        optimizer_steps=32,internal_route_forwards=272,final_candidates=8,
        Gaussian_CDF_evaluations=272*4*2*7 if analytic else 0,
        current_NN_signature_calls=272,current_NN_tip_proxy_calls=272,
        predicted_protection_feasible_states=protection_feasible,inferred_geometry_feasible_states=geometry_feasible,feasible_evaluated_states=feasible,
        actual_controller_queries=0,maximum_coordinate_edit_m=float(np.abs(result-paths).max()),
        scope='One32step TRAIN-only spatial-span/support diagnostic; .08per-coordinate bound, fixed endpoints, same predicted/NN protections; no physical safety certification')
