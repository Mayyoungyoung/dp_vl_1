import numpy as np
from scripts.run_observed_probability import ROOT,write,torch_setup


def same(a,b):
    import torch
    if torch.is_tensor(a):torch.testing.assert_close(a,b,rtol=0,atol=0)
    elif isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):same(x,y)
    else:assert a==b,(a,b)


if __name__=='__main__':
    torch=torch_setup();r=ROOT/'runs/factored_q_v1'
    a=torch.load(r/'profile_continuous/last.pt',map_location='cpu',weights_only=False)
    b=torch.load(r/'profile_resumed/last.pt',map_location='cpu',weights_only=False)
    keys=('model','optimizer','scheduler','rng','sampler','settings','normalization','step','best','history')
    for k in keys:same(a[k],b[k])
    write(r/'resume_check.json',dict(exact=True,compared=list(keys),continuous_steps=100,split_steps=[50,50]))
    print('Exact resume verified',flush=True)
