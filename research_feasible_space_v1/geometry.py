"""TRAIN-only labels and analytic corridor certificate; no learned forward imports."""
import numpy as np
from routeset.verified_route_set import signed_clearances

def segment_radii(paths, centers, halves, floor, cap=.06, factor=.8):
    p=np.asarray(paths,dtype=np.float64)
    slack=signed_clearances(p,centers,np.asarray(halves)+.02)
    vertical=np.minimum(p[:,:-1,2],p[:,1:,2])-floor
    r=factor*np.minimum(slack,vertical)
    return np.minimum(cap,np.maximum(0,r)).astype(np.float32),slack

def node_radii_numpy(r):
    r=np.asarray(r);zero=np.zeros_like(r[...,:1])
    return np.concatenate((zero,np.minimum(r[...,:-1],r[...,1:]),zero),-1)

def prototypes(paths, check, events, count=2):
    """Deterministic2cluster means; independently invalid means fall back to medoid."""
    p=np.asarray(paths);flat=p.reshape(len(p),-1)
    mean=flat.mean(0);first=int(((flat-mean)**2).sum(1).argmin())
    second=int(((flat-flat[first])**2).sum(1).argmax());c=flat[[first,second]].copy()
    for _ in range(12):
        assignment=((flat[:,None]-c[None])**2).mean(-1).argmin(1)
        new=np.array([flat[assignment==j].mean(0) if (assignment==j).any() else c[j] for j in range(count)])
        if np.array_equal(new,c):break
        c=new
    c=c.reshape(count,24,3);ok,_=check(c,np.repeat(events[:1],count,0));fallback=0
    for j in range(count):
        if not ok[j]:
            ids=np.flatnonzero(assignment==j)
            if not len(ids):ids=np.arange(len(p))
            c[j]=p[ids[((p[ids]-c[j])**2).mean((1,2)).argmin()]];fallback+=1
    return c.astype(np.float32),assignment,fallback
