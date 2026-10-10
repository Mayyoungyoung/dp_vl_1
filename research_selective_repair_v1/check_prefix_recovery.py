"""Short exact RNG/optimizer recovery audit before the registered full fits."""
import argparse
from pathlib import Path
import torch
from research_selective_repair_v1.io import RUN,write,sha
from research_selective_repair_v1.body_prefix_forecast import fit

def compare(a,b,key='root'):
    if isinstance(a,torch.Tensor):
        assert isinstance(b,torch.Tensor);assert torch.equal(a.cpu(),b.cpu()),key
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),key
        for k in a:compare(a[k],b[k],key+'/'+str(k))
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b),key
        for j,(x,y) in enumerate(zip(a,b)):compare(x,y,key+'/'+str(j))
    elif hasattr(a,'shape'):
        import numpy as np
        assert np.array_equal(a,b),key
    else:assert a==b,key

def run(name,dataset):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    full=name+'_uninterrupted';partial=name+'_interrupted'
    fit(full,dataset,seed=206015,steps=120)
    fit(partial,dataset,seed=206015,steps=120,stop_after=60)
    assert not (RUN/partial/'last.pt').exists()
    fit(partial,dataset,seed=206015,steps=120,resume=True)
    a=torch.load(RUN/full/'last.pt',map_location='cpu',weights_only=False)
    b=torch.load(RUN/partial/'last.pt',map_location='cpu',weights_only=False)
    compare(a,b)
    write(out/'SUMMARY.json',dict(exact_tensor_optimizer_RNG_stream_history=True,
        uninterrupted_sha256=sha(RUN/full/'last.pt'),resumed_sha256=sha(RUN/partial/'last.pt'),
        steps=120,interrupt_at=60,role='TRAIN technical recovery check,not research seed',locked_access=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',required=True,type=Path);run(**vars(p.parse_args()))
