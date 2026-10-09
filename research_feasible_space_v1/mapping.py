"""Pure differentiable construction, no truth, verifier or hidden scene inputs."""
import torch

def node_radii(radii):
    if radii.shape[-1]!=23:raise ValueError('23 cells required')
    zero=torch.zeros_like(radii[...,:1])
    return torch.cat((zero,torch.minimum(radii[...,:-1],radii[...,1:]),zero),-1)

def construct(centers,radii,parameters,kind='bounded'):
    if centers.shape[-2:]!=(24,3) or parameters.shape!=centers.shape:raise ValueError('24XYZ nodes required')
    if (radii<0).any() or not torch.isfinite(radii).all():raise ValueError('Nonnegative finite radii required')
    radius=node_radii(radii)[...,None]
    if kind=='bounded':offset=radius*parameters.tanh()
    elif kind=='relative':offset=radius*parameters
    elif kind=='xyz':offset=.06*parameters
    elif kind=='projection':offset=(.06*parameters).clamp(-radius,radius)
    elif kind=='center':offset=torch.zeros_like(parameters)
    else:raise ValueError(kind)
    offset=offset.clone();offset[...,[0,-1],:]=0
    return centers+offset
