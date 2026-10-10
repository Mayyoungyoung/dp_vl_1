"""Bounded internal public-FK conditioning,never native IK/planning queries."""
import torch

def pose_jacobian(q,qref,base,relative):
    frame=base.expand(q.shape[:-1]+(4,4));origins=[];axes=[];delta=q-qref
    for j in range(7):
        origins.append(frame[...,:3,3]);axes.append(frame[...,:3,2])
        c,s=delta[...,j].cos(),delta[...,j].sin();zero=torch.zeros_like(c);one=torch.ones_like(c)
        rotation=torch.stack([c,-s,zero,zero,s,c,zero,zero,zero,zero,one,zero,zero,zero,zero,one],-1).reshape(q.shape[:-1]+(4,4))
        frame=frame@rotation@relative[j]
    axis=torch.stack(axes,-2);origin=torch.stack(origins,-2)
    linear=torch.cross(axis,frame[...,:3,3][...,None,:]-origin,dim=-1).transpose(-1,-2)
    return frame,linear,axis.transpose(-1,-2)

def terminal(q,target,rotation,robot,steps=4):
    qref,base,relative=robot
    identity=torch.eye(6,device=q.device,dtype=q.dtype)
    for _ in range(steps):
        frame,linear,angular=pose_jacobian(q,qref,base,relative)
        orientation=.5*torch.cross(frame[...,:3,:3].transpose(-1,-2),rotation.transpose(-1,-2),dim=-1).sum(-2)
        # .2m/rad follows the fixed .02m/.1rad state likelihood scales.
        jac=torch.cat([linear,.2*angular],-2)
        residual=torch.cat([target-frame[...,:3,3],.2*orientation],-1)
        # Same damped linear system; solve_ex avoids one CUDA/CPU barrier per
        # tiny system. Positive damping makes the matrix strictly positive
        # definite; finite loss/gradient tests still check the whole model.
        solution=torch.linalg.solve_ex(jac@jac.transpose(-1,-2)+.0001*identity,residual[...,None],check_errors=False)[0]
        update=jac.transpose(-1,-2)@solution
        q=q+update[...,0].clamp(-.35,.35)
    return q

def crossing(q,row_x,robot,steps=4):
    qref,base,relative=robot
    for _ in range(steps):
        frame,jac,_=pose_jacobian(q,qref,base,relative);direction=jac[...,0,:]
        residual=row_x-frame[...,0,3]
        update=residual[...,None]*direction/(direction.square().sum(-1,keepdim=True)+.0001)
        q=q+update.clamp(-.35,.35)
    return q
