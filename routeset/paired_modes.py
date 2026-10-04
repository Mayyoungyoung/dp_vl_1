"""Training-only partial correspondence of witnessed geometric route families.

No oracle geometry enters the generator forward. Correspondence is between
families with positive witnesses on both sides, never between fixed query IDs.
"""
import numpy as np
import torch


def row_crossings(paths, row_x):
    """First forward crossing, piecewise differentiable in actual path points.

For a missing crossing use the nearest vertex; normal endpoint/route losses
remain responsible for reaching the goal. Hard indices carry no gradient.
"""
    out=[]
    a,b=paths[:,:-1],paths[:,1:]
    for x in row_x:
        hit=(a[...,0]<=x)&(b[...,0]>=x)&((b[...,0]-a[...,0])>1e-7)
        idx=hit.to(torch.int64).argmax(-1)
        rows=torch.arange(len(paths),device=paths.device)
        aa,bb=a[rows,idx],b[rows,idx]
        dx=bb[:,0]-aa[:,0]
        safe=torch.where(dx.abs()>1e-7,dx,torch.ones_like(dx))
        u=((x-aa[:,0])/safe).clamp(0,1)
        cross=aa+u[:,None]*(bb-aa)
        nearest=(paths[...,0]-x).abs().argmin(-1)
        cross=torch.where(hit.any(-1)[:,None],cross,paths[rows,nearest])
        out.append(cross[:,1:])
    return torch.stack(out,1)


def relation_cost(paths, config, modes):
    """M x C squared distance to TRAIN passage bands, in squared meters."""
    if not modes:return paths.new_empty((len(paths),0))
    crossing=row_crossings(paths,config['row_x']); bounds=[]
    for mode in modes:
        rows=[]
        for r,label in enumerate(mode):
            ys=config['post_y'][r]; top=config['post_base_z']+config['post_heights'][r]
            if label=='over':
                rows.append([[-1e3,top+.02],[1e3,1e3]])
            else:
                j=int(label[3:]);lo=-.45 if j==0 else ys[j-1]+.0375
                hi=.45 if j==len(ys) else ys[j]-.0375
                rows.append([[lo,config['post_base_z']+.02],[hi,top-.02]])
        bounds.append(rows)
    limits=paths.new_tensor(bounds)
    value=crossing[:,None]
    return ((limits[None,:,:,0]-value).clamp_min(0).square()
            +(value-limits[None,:,:,1]).clamp_min(0).square()).sum(-1).mean(-1)


def mode_descriptor(paths, config):
    crossing=row_crossings(paths,config['row_x'])
    offset=paths.new_tensor([[np.mean(ys),config['post_base_z']+h]
                           for ys,h in zip(config['post_y'],config['post_heights'])])
    # Translation-relative, metric descriptor: no path-coordinate equality.
    return crossing-offset[None]


def within_scene_loss(paths, configs, modes):
    values=[]
    for x,c,types in zip(paths,configs,modes):
        if not types:continue
        cost=relation_cost(x,c,types)
        values.append((cost.min(0).values.mean()+cost.min(1).values.mean())/2)
    return torch.stack(values).mean() if values else paths.sum()*0


def partial_pair_loss(paths, configs, modes, pairs):
    values=[]; matched=0
    for a,b in pairs:
        common=sorted(set(modes[a])&set(modes[b]))
        if not common:continue
        ca=relation_cost(paths[a],configs[a],common)
        cb=relation_cost(paths[b],configs[b],common)
        # Partial matching: only shared, positively witnessed relation classes.
        # Deleted/new/unknown classes have no edge and are learned per-scene.
        ia=ca.detach().argmin(0);ib=cb.detach().argmin(0)
        da=mode_descriptor(paths[a],configs[a])[ia]
        db=mode_descriptor(paths[b],configs[b])[ib]
        values.append((da-db).square().mean())
        matched+=len(common)
    return (torch.stack(values).mean() if values else paths.sum()*0),matched


def class_weights(paths, mask, configs, signature):
    weights=np.zeros(mask.shape,dtype=np.float32); modes=[]
    for i,(refs,valid,cfg) in enumerate(zip(paths,mask,configs)):
        groups={}
        for j in np.flatnonzero(valid):
            mode=signature(refs[j],cfg)
            key=tuple(mode) if mode is not None else ('unknown_reference',int(j))
            groups.setdefault(key,[]).append(int(j))
        known=sorted(k for k in groups if k[0]!='unknown_reference');modes.append(known)
        for indices in groups.values():weights[i,indices]=1/(len(groups)*len(indices))
    return weights,modes
