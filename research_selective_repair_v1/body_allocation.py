"""Commit one correction per slot using predicted actual outcome distributions.

Finite expected coverage is a conventional optimizer,not the claimed novelty.
"""
import itertools
import numpy as np

COMBINATIONS=np.asarray(list(itertools.product(range(3),repeat=8)),np.int64)

def coverage(p):return 1-np.prod(1-p[...,1:],axis=-2)

def objectives(prob,choices,protected=True):
    p=prob[np.arange(8)[None],choices];cov=coverage(p);original=coverage(prob[:,0])
    # Same hard predicted preservation constraint for exact and ordinary controls.
    eligible=((cov>=original-.01).all(-1)&(p[...,0]<=prob[:,0,0][None]+.01).all(-1)) if protected else np.ones(len(choices),bool)
    utility=cov.sum(-1)+.05*(1-p[...,0]).sum(-1)-.01*(choices!=0).sum(-1)
    return np.where(eligible,utility,-np.inf)

def allocate(prob,kind='actual',protected=True,planned=None):
    p=np.asarray(prob,np.float64);assert p.shape==(8,3,17)
    assert np.isfinite(p).all() and np.allclose(p.sum(-1),1,atol=1e-5)
    if kind=='identity':return np.zeros(8,np.int64),0
    if kind=='lift':return np.ones(8,np.int64),0
    if kind=='preserved':return np.full(8,2,np.int64),0
    if kind=='success':return (1-p[...,0]-.002*np.array([0,1,1])).argmax(-1),24
    if kind=='planned':
        assert planned is not None and np.asarray(planned).shape==(8,3)
        p=np.zeros_like(p);p[...,0]=prob[...,0]
        for i in range(8):
            for j in range(3):
                w=int(planned[i,j]);p[i,j,w if w>0 else 0]+=1-prob[i,j,0]
    if kind in ('actual','planned'):
        score=objectives(p,COMBINATIONS,protected);return COMBINATIONS[int(score.argmax())].copy(),len(COMBINATIONS)
    assert kind=='coordinate'
    choice=np.zeros(8,np.int64);queries=0
    for _ in range(5):
        changed=False
        for slot in range(8):
            trials=np.broadcast_to(choice,(3,8)).copy();trials[:,slot]=np.arange(3)
            best=int(objectives(p,trials,protected).argmax());queries+=3
            changed|=choice[slot]!=best;choice=trials[best]
        if not changed:break
    return choice,queries
