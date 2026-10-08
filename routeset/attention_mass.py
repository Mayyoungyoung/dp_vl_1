"""Conventional fixed-scale Gaussian attention aggregation on observed points."""
def mass_anchor(points, weights, sigma=.025, chunk=256):
    """Exact Gaussian attention mass at every observed point, chunked in memory."""
    import torch
    if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(weights):
        raise ValueError('point/weight shape mismatch')
    if len(points) == 0 or sigma <= 0 or chunk < 1:
        raise ValueError('nonempty points and positive scale/chunk required')
    scores=[]
    for start in range(0, len(points), chunk):
        d2=((points[start:start+chunk,None]-points[None])**2).sum(-1)
        scores.append(torch.exp(-d2/(2*sigma*sigma)) @ weights)
    scores=torch.cat(scores)
    return points[scores.argmax()], scores.max()
