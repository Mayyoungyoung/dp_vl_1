"""Finite edit operator using predicted constraints, never oracle boxes/checks."""
import numpy as np
import torch
from routeset.segment_clearance import path_segment_clearances
from research_selective_repair_v1.core import masks,VOCAB
from scripts.research_v3_audit import mode


def configuration(centers,halves,settings):
    return dict(row_x=[float(np.mean(centers[j:j+2,0])) for j in (0,2)],post_y=[centers[j:j+2,1].tolist() for j in (0,2)],post_heights=[float(np.max(2*halves[j:j+2,2])) for j in (0,2)],post_base_z=settings['post_base'],tip_clearance_m=.02)


def crossings(paths,rows):
    a=paths[:,:,:-1];b=paths[:,:,1:];rx=rows[:,None,None,:,None]
    hit=(a[...,None,0:1]<rx)&(b[...,None,0:1]>=rx)
    ix=hit[...,0].float().argmax(2)
    aa=a.gather(2,ix[...,None].expand(-1,-1,-1,3));bb=b.gather(2,ix[...,None].expand(-1,-1,-1,3))
    t=(rows[:,None]-aa[...,0])/(bb[...,0]-aa[...,0]).clamp_min(1e-5)
    return aa+t.clamp(0,1)[...,None]*(bb-aa),ix,hit[...,0].any(2)


def repair(drafts,modes,goals,available,centers,halves,settings,kind='protected',steps=64):
    """One contiguous <=16 interior window plus smooth goal tail; fixed starts.

    protected: retain one predicted feasible representative per operational word,
    release failed and redundant slots. global: same boxes, ordinary optimization.
    mode_global: same mode constraints, without representative/window protection.
    Predicted feasibility is model inference, not a checker validity certificate.
    """
    x=drafts.detach().clone();n,k,h,_=x.shape
    s=torch.linspace(0,1,7,device=x.device)[1:];s=s*s*(3-2*s)
    diff=(goals[:,None]-x[:,:,-1]).clamp(-.4,.4)*available[:,None,None]
    x[:,:,18:]+=s[None,None,:,None]*diff[:,:,None]
    with torch.no_grad():
        slack=path_segment_clearances(x,centers,halves)
        clear=(slack>=.023).all(-1)&(x[...,2]>=settings['floor']).all(-1)
        raw=[];target=[];released=[]
        for i in range(n):
            cfg=configuration(centers[i].cpu().numpy(),halves[i].cpu().numpy(),settings)
            words=[mode(p,cfg) for p in x[i].cpu().numpy()];typed=np.array([w in VOCAB for w in words])
            feasible=clear[i].cpu().numpy()&typed;keep=set();protected=np.zeros(k,bool)
            # Prefer a representative already matching its requested word.
            order=sorted(range(k),key=lambda j: (words[j]!=VOCAB[int(modes[i,j])],j))
            for j in order:
                if feasible[j] and words[j] not in keep:protected[j]=True;keep.add(words[j])
            release=~protected
            targets=[words[j] if words[j] in VOCAB and not feasible[j] else VOCAB[int(modes[i,j])] for j in range(k)]
            raw.append(words);target.append([[('gap0','gap1','gap2','over').index(v) for v in w.split('|')] for w in targets]);released.append(release)
        release=torch.tensor(np.asarray(released),device=x.device)
        wanted=torch.tensor(target,device=x.device)
        rows=centers.reshape(n,2,2,3)[...,0].mean(-1)
        points,ix,has=crossings(x,rows)
        node=torch.zeros_like(x[...,0]);bad=slack<.026
        node[:,:,:-1]=bad;node[:,:,1:]=torch.maximum(node[:,:,1:],bad.float())
        node=torch.nn.functional.max_pool1d(node.reshape(n*k,1,h),5,stride=1,padding=2).reshape(n,k,h)
        for i in range(n):
            for j in range(k):
                if release[i,j]:
                    for r in range(2):
                        actual=raw[i][j].split('|')[r] if raw[i][j] else None
                        if actual!=('gap0','gap1','gap2','over')[wanted[i,j,r]]:
                            q=int(ix[i,j,r]);node[i,j,max(1,q-2):min(23,q+4)]=1
        support=masks(node,.5,16) if kind=='protected' else torch.ones_like(node)
        support*=release[...,None] if kind=='protected' else (~clear)[...,None] if kind=='global' else torch.ones_like(release[...,None])
        support[:,:,0]=0;support[:,:,-1]=0
    if not bool(support.any()):return x,dict(support=support,steps=0,release=release)
    delta=torch.zeros_like(x,requires_grad=True);opt=torch.optim.Adam([delta],lr=.004)
    low=centers.reshape(n,2,2,3)[...,1]-halves.reshape(n,2,2,3)[...,1]-.026
    high=centers.reshape(n,2,2,3)[...,1]+halves.reshape(n,2,2,3)[...,1]+.026
    top=(centers.reshape(n,2,2,3)[...,2]+halves.reshape(n,2,2,3)[...,2]).max(-1).values+.026
    for _ in range(steps):
        p=x+support[...,None]*delta
        safe=(.026-path_segment_clearances(p,centers,halves)).clamp_min(0).square().mean(-1)
        safe+=(settings['floor']-p[...,2]).clamp_min(0).square().mean(-1)
        constraint=torch.zeros_like(safe)
        if kind!='global':
            point,_,has=crossings(p,rows);y,z=point[...,1],point[...,2]
            err=torch.where(wanted==0,(y-low[:,None,:,0]).clamp_min(0),torch.where(wanted==2,(high[:,None,:,1]-y).clamp_min(0),torch.where(wanted==1,(high[:,None,:,0]-y).clamp_min(0)+(y-low[:,None,:,1]).clamp_min(0),(top[:,None]-z).clamp_min(0))))
            constraint=(err.square()+.01*(~has).float()).mean(-1)
            if kind=='protected':constraint*=release
        bend=(p[:,:,2:]-2*p[:,:,1:-1]+p[:,:,:-2]-(x[:,:,2:]-2*x[:,:,1:-1]+x[:,:,:-2])).square().mean()
        loss=1000*(safe+constraint).mean()+.2*delta.square().mean()+.1*bend
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():delta.clamp_(-.12,.12)
    return (x+support[...,None]*delta).detach(),dict(support=support,steps=steps,release=release)
