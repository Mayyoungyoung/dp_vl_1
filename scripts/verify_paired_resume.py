"""Compare actual continuous and interrupted/resumed 100-update executions."""
import numpy as np
from scripts.paired_modes_data import RUN
from scripts.run_observed_probability import write,sha


def equal(a,b,path=''):
    import torch
    if isinstance(a,torch.Tensor):torch.testing.assert_close(a,b,rtol=0,atol=0,msg=path)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b,err_msg=path)
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:equal(a[k],b[k],path+'/'+str(k))
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'/'+str(i))
    else:assert a==b,path


if __name__=='__main__':
    import torch
    paths=[RUN/n/'last.pt' for n in ('profile_R2_continuous','profile_R2_resumed')]
    a,b=[torch.load(p,map_location='cpu',weights_only=False) for p in paths]
    keys=('model','optimizer','scheduler','rng','loss_rng','sampler','config','step','coefficients')
    for k in keys:equal(a[k],b[k],k)
    write(RUN/'resume_verification.json',dict(exact_equal=list(keys),steps=a['step'],
        files={str(p):sha(p) for p in paths},scope='Actual model updates, optimizer, scheduler, all RNG and sampled input sequence'))
