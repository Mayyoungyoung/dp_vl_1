"""Finite observed body correction recipes, without simulator or truth input."""
import numpy as np

OPTIONS=('identity','lift','crossing_preserved_lift')

def signed_options(paths,completed,lowering_m):
    """Same24 internal alternatives: replace preserved lift with TRAIN bias pull.

    lowering_m is a frozen learned scalar, not an observed actual event input.
    First/last nodes remain fixed; invalid body/geometry results stay unknown
    until the corresponding predictor or independent evaluation observes them.
    """
    assert -.08<=float(lowering_m)<0
    p=np.asarray(paths);assert p.shape[-2:]==(24,3)
    factor=np.sin(np.pi*np.linspace(0,1,24));factor[[0,-1]]=0
    out=np.broadcast_to(p[:,None],(len(p),3,24,3)).copy()
    out[:,1,:,2]+=.08*factor;out[:,2,:,2]+=float(lowering_m)*factor
    return out

def options(paths,completed,lift=.08):
    """Return3 internal alternatives per slot; caller commits exactly one per slot.

    All alternatives and model evaluations count as internal correction compute.
    These are never a24-candidate scored deployment pool.
    """
    p=np.asarray(paths);assert p.shape[-2:]==(24,3)
    factor=np.broadcast_to(np.sin(np.pi*np.linspace(0,1,24)),p.shape[:-1]).copy()
    factor[...,[0,-1]]=0
    protected=factor.copy();rows=np.asarray(completed)[...,0].reshape(2,2).mean(-1)
    for j,path in enumerate(p):
        for row in rows:
            crossing=np.flatnonzero((path[:-1,0]<row)&(path[1:,0]>=row))
            if not len(crossing):protected[j]=0;break
            q=int(crossing[0]);protected[j,max(0,q-1):min(24,q+3)]=0
    out=np.broadcast_to(p[:,None],(len(p),3,24,3)).copy()
    out[:,1,:,2]+=lift*factor;out[:,2,:,2]+=lift*protected
    return out
