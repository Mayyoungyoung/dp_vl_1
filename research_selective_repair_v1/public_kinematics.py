"""Public serial revolute-chain FK; no scene/IK/collision queries."""
import numpy as np

def chain(joint_world,tip_world):
    joint_world=np.asarray(joint_world,dtype=np.float64)
    tip_world=np.asarray(tip_world,dtype=np.float64)
    assert joint_world.shape==(7,4,4) and tip_world.shape==(4,4)
    frames=np.concatenate([joint_world,tip_world[None]],0)
    relative=np.linalg.solve(frames[:-1],frames[1:])
    return frames[0],relative

def forward(q,q0,base,relative):
    q=np.asarray(q);delta=q-np.asarray(q0)
    assert q.shape[-1]==7
    result=np.broadcast_to(base,q.shape[:-1]+(4,4)).copy()
    for j in range(7):
        c,s=np.cos(delta[...,j]),np.sin(delta[...,j])
        rotation=np.broadcast_to(np.eye(4),result.shape).copy()
        rotation[...,0,0]=rotation[...,1,1]=c
        rotation[...,0,1]=-s;rotation[...,1,0]=s
        result=result@rotation@relative[j]
    return result[...,:3,3]

def load(path):
    with np.load(path) as z:q0=z['q0'];base,relative=chain(z['joint_world'],z['tip_world'])
    return q0,base,relative

def verify(model,trace):
    q0,base,relative=load(model)
    with np.load(trace) as z:q=z['arm_joint_positions'];tip=z['gripper_pose'][:,:3]
    error=np.linalg.norm(forward(q,q0,base,relative)-tip,axis=-1)
    return dict(samples=len(q),max_tip_error_m=float(error.max()),mean_tip_error_m=float(error.mean()))
