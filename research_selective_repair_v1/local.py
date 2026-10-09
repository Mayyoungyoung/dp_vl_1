"""Observed relative features and finite geometry repair. No truth in forward."""
import numpy as np
import torch
from research_feasible_space_v1.observation_evidence import classify_probes
from research_selective_repair_v1.core import masks

def observed_points(inp,maximum=768):
    xyz=inp['world_xyz'][0].detach().cpu().numpy();rgb=inp['rgb'][0].detach().cpu().numpy()
    valid=inp['valid_mask'][0].detach().cpu().numpy().astype(bool)
    # Registered gray-post domain: visible neutral surfaces only, never truth mask.
    keep=valid&(rgb.max(-1)-rgb.min(-1)<.08)&(rgb.mean(-1)>.15)&(rgb.mean(-1)<.8)&(xyz[:,2]>.78)&(xyz[:,2]<1.1)&(xyz[:,0]>.07)&(xyz[:,0]<.42)
    p=xyz[keep]
    if len(p)>maximum:p=p[np.linspace(0,len(p)-1,maximum).astype(int)]
    return p.astype(np.float32)

def features(paths,points,observation):
    paths=np.asarray(paths);n=len(paths);points=np.asarray(points)
    if len(points):
        ds=((paths[:,:,None]-points[None,None])**2).sum(-1)
        k=min(4,len(points));idx=np.argsort(ds,axis=-1)[:,:,:k]
        rel=points[idx]-paths[:,:,None];dist=np.sqrt(np.take_along_axis(ds,idx,-1))
        if k<4:
            rel=np.pad(rel,((0,0),(0,0),(0,4-k),(0,0)));dist=np.pad(dist,((0,0),(0,0),(0,4-k)),constant_values=1)
    else:rel=np.zeros((n,24,4,3));dist=np.ones((n,24,4))
    vis=classify_probes(paths,observation['depth'],observation['camera_intrinsics'],observation['camera_extrinsics'])
    onehot=np.stack([vis==s for s in ('free_at_probe','observed_surface','unknown')],-1)
    tangent=np.gradient(paths,axis=1);curve=np.gradient(tangent,axis=1)
    time=np.broadcast_to(np.linspace(0,1,24)[None,:,None],(n,24,1))
    return np.concatenate([rel.reshape(n,24,12).clip(-.15,.15)/.1,dist.clip(0,.2)/.1,onehot,paths-paths[:,:1],tangent/.05,curve/.05,time],-1).astype(np.float32)

def features_torch(paths,points,point_mask,depth,intrinsics,extrinsics):
    """Refresh nearest surfaces and current ray categories on moving center drafts."""
    b,k,h,_=paths.shape;p=paths.reshape(b,k*h,3)
    squared=(p[:,:,None]-points[:,None]).square().sum(-1).masked_fill(~point_mask[:,None],float('inf'))
    distances,ix=squared.topk(4,largest=False);neighbors=points[:,None].expand(-1,k*h,-1,-1).gather(2,ix[...,None].expand(-1,-1,-1,3))
    usable=torch.isfinite(distances);rel=torch.where(usable[...,None],neighbors-p[:,:,None],torch.zeros_like(neighbors))
    distances=torch.where(usable,distances.clamp_min(0).sqrt(),torch.ones_like(distances))
    camera=(p-extrinsics[:,None,:3,3])@torch.linalg.inv(extrinsics[:,:3,:3]).transpose(-1,-2)
    projected=camera@intrinsics.transpose(-1,-2);positive=torch.isfinite(camera).all(-1)&(camera[...,2]>0)
    uv=projected[...,:2]/torch.where(positive,projected[...,2],torch.ones_like(projected[...,2]))[...,None]
    pixel=uv.round().long();height,width=depth.shape[-2:]
    on=positive&(pixel[...,0]>=0)&(pixel[...,0]<width)&(pixel[...,1]>=0)&(pixel[...,1]<height)
    seen=depth.flatten(1).gather(1,pixel[...,1].clamp(0,height-1)*width+pixel[...,0].clamp(0,width-1))
    known=on&torch.isfinite(seen)&(seen>0)&(seen<=10)
    free=known&(camera[...,2]<seen-.005);surface=known&((camera[...,2]-seen).abs()<=.005)
    onehot=torch.stack([free,surface,~(free|surface)],-1).to(paths.dtype)
    tangent=torch.gradient(paths,dim=2)[0];curve=torch.gradient(tangent,dim=2)[0]
    time=torch.linspace(0,1,h,device=paths.device,dtype=paths.dtype)[None,None,:,None].expand(b,k,-1,-1)
    return torch.cat([rel.reshape(b,k,h,12).clamp(-.15,.15)/.1,distances.reshape(b,k,h,4).clamp(0,.2)/.1,onehot.reshape(b,k,h,3),paths-paths[:,:,:1],tangent/.05,curve/.05,time],-1)

def optimize(drafts,centers=None,halves=None,floor=None,points=None,steps=64):
    """Oracle TRAIN diagnostic or separate observed-point baseline, same edit budget."""
    from routeset.segment_clearance import path_segment_clearances
    x=torch.as_tensor(drafts,dtype=torch.float32,device='cuda')
    if centers is not None:
        c=torch.as_tensor(centers,dtype=torch.float32,device='cuda')[None]
        h=torch.as_tensor(halves,dtype=torch.float32,device='cuda')[None]
        with torch.no_grad():bad=(path_segment_clearances(x[None],c,h)[0]<.023).float()
        node=torch.zeros_like(x[...,0]);node[:,:-1]=torch.maximum(node[:,:-1],bad);node[:,1:]=torch.maximum(node[:,1:],bad)
    else:
        if points is None or not len(points):return np.asarray(drafts).copy(),np.zeros((len(drafts),24))
        pts=torch.as_tensor(points,dtype=torch.float32,device='cuda')
        with torch.no_grad():node=(torch.cdist(x,pts[None].expand(len(x),-1,-1)).min(-1).values<.045).float()
    node=torch.nn.functional.max_pool1d(node[:,None],5,stride=1,padding=2)[:,0]
    mask=masks(node,.5,12);delta=torch.zeros_like(x,requires_grad=True);opt=torch.optim.Adam([delta],lr=.004)
    if not bool(mask.any()):return np.asarray(drafts).copy(),mask.cpu().numpy()
    for _ in range(steps):
        p=x+mask[...,None]*delta
        if centers is not None:
            slack=path_segment_clearances(p[None],c,h)[0]
            safety=(.024-slack).clamp_min(0).square().mean()
            if floor is not None:safety=safety+(float(floor)-p[...,2]).clamp_min(0).square().mean()
        else:
            probes=torch.cat([p,(p[:,1:]+p[:,:-1])/2],1)
            distance=torch.cdist(probes,pts[None].expand(len(x),-1,-1)).min(-1).values
            safety=(.032-distance).clamp_min(0).square().mean()
        bend=(p[:,2:]-2*p[:,1:-1]+p[:,:-2]-(x[:,2:]-2*x[:,1:-1]+x[:,:-2])).square().mean()
        loss=160*safety+delta.square().mean()+.1*bend
        opt.zero_grad();loss.backward();opt.step()
        with torch.no_grad():delta.clamp_(-.08,.08)
    return (x+mask[...,None]*delta).detach().cpu().numpy(),mask.cpu().numpy()
