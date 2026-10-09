"""Exact clearance of reachable cells conv(endpoint cubes with unequal widths)."""
import numpy as np
import torch

def tapered_segment_clearance(p0,p1,r0,r1,center,half):
    a=p0-center;d=p1-p0
    intercept=torch.cat((a-half,-a-half),-1)-r0[...,None]
    slope=torch.cat((d,-d),-1)-(r1-r0)[...,None]
    slope,intercept=torch.broadcast_tensors(slope,intercept)
    i,j=torch.triu_indices(6,6,1,device=p0.device);den=slope[...,i]-slope[...,j];moving=den.abs()>1e-12
    t=((intercept[...,j]-intercept[...,i])/torch.where(moving,den,torch.ones_like(den))).clamp(0,1)
    t=torch.where(moving,t,torch.zeros_like(t));t=torch.cat((torch.zeros_like(t[...,:1]),torch.ones_like(t[...,:1]),t),-1)
    return (intercept[...,None,:]+t[..., :,None]*slope[...,None,:]).max(-1).values.min(-1).values

def tapered_path_clearance(p,n,centers,halves):
    return tapered_segment_clearance(p[:,:,:-1,None,:],p[:,:,1:,None,:],n[:,:,:-1,None],n[:,:,1:,None],centers[:,None,None],halves[:,None,None]).min(-1).values

def tapered_clearance_numpy(paths,node_widths,centers,halves,floor):
    p=np.asarray(paths,dtype=np.float64);n=np.asarray(node_widths,dtype=np.float64)
    a=p[:,:-1,None]-np.asarray(centers);d=(p[:,1:]-p[:,:-1])[:,:,None]
    intercept=np.concatenate((a-halves,-a-halves),-1)-n[:,:-1,None,None]
    slope=np.broadcast_to(np.concatenate((d,-d),-1),intercept.shape)-(n[:,1:]-n[:,:-1])[:,:,None,None]
    i,j=np.triu_indices(6,1);den=slope[...,i]-slope[...,j];moving=np.abs(den)>1e-12;t=np.zeros_like(den)
    np.divide(intercept[...,j]-intercept[...,i],den,out=t,where=moving)
    t=np.concatenate((np.zeros_like(t[...,:1]),np.ones_like(t[...,:1]),np.clip(t,0,1)),-1)
    slack=(intercept[...,None,:]+t[..., :,None]*slope[...,None,:]).max(-1).min(-1).min(-1)
    floor_slack=np.minimum(p[:,:-1,2]-n[:,:-1],p[:,1:,2]-n[:,1:])-floor
    return np.minimum(slack,floor_slack)
