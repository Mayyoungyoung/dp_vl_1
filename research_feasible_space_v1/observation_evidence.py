"""Current-camera ray evidence for probes. Never certifies an entire cell."""
import numpy as np

def classify_probes(xyz,depth,intrinsics,camera_to_world,tolerance=.005):
    p=np.asarray(xyz,dtype=np.float64);shape=p.shape[:-1];p=p.reshape(-1,3)
    k=np.asarray(intrinsics,dtype=np.float64);pose=np.asarray(camera_to_world,dtype=np.float64);d=np.asarray(depth)
    if k.shape!=(3,3) or pose.shape not in ((3,4),(4,4)) or d.ndim!=2:raise ValueError('Current calibrated depth required')
    camera=(p-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T
    projected=camera@k.T;positive=np.isfinite(camera).all(1)&(camera[:,2]>0)
    uv=np.zeros((len(p),2));np.divide(projected[:,:2],projected[:,2,None],out=uv,where=positive[:,None])
    pixel=np.rint(uv).astype(np.int64);h,w=d.shape;on=positive&(pixel[:,0]>=0)&(pixel[:,0]<w)&(pixel[:,1]>=0)&(pixel[:,1]<h)
    seen=np.full(len(p),np.nan);seen[on]=d[pixel[on,1],pixel[on,0]];known=on&np.isfinite(seen)&(seen>0)&(seen<=10)
    status=np.full(len(p),'unknown',dtype='<U18')
    status[known&(camera[:,2]<seen-tolerance)]='free_at_probe'
    status[known&(np.abs(camera[:,2]-seen)<=tolerance)]='observed_surface'
    # Behind a visible depth return is unknown, not occupied and not free.
    return status.reshape(shape)
