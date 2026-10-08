"""Actual continuous-versus-resumed checkpoint comparison, no tolerance."""
import argparse
from pathlib import Path
import numpy as np
import torch
from scripts.run_observed_probability import ROOT,write,sha


def equal(a,b,path='root'):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor);torch.testing.assert_close(a,b,rtol=0,atol=0);return
    if isinstance(a,np.ndarray):np.testing.assert_array_equal(a,b);return
    if isinstance(a,dict):
        assert set(a)==set(b),path
        for k in a:equal(a[k],b[k],path+'.'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):equal(x,y,path+'.'+str(i))
    else:assert a==b,path


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--a',required=True);p.add_argument('--b',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    a,b=(torch.load(x,map_location='cpu',weights_only=False) for x in (args.a,args.b))
    equal(a,b)
    write(args.output,dict(exact=True,checkpoint_sha256=[sha(args.a),sha(args.b)],
        fields=['model','optimizer','scheduler','rng','loss_rng','sampler','mode_exposure','unique_routes','settings','step','history']))
