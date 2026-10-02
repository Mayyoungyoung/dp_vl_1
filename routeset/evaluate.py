"""Evaluate raw exactly-K outputs: no oversampling, repair, or oracle selection."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import load_data, seed_all, sha256, write_json
from .diffusion import DiffusionSchedule
from .geometry import route_metrics, route_modes, path_validity
from .train import build_model, predict


def aggregate(paths,scenes):
    # Flatten repeats into evaluations, keeping scene groups for confidence intervals.
    n,r,k,h,d=paths.shape
    repeated=np.repeat(scenes,r,axis=0)
    flat=paths.reshape(n*r,k,h,d)
    metrics=route_metrics(flat,repeated)
    valid=np.stack([path_validity(p,s)['valid'] for p,s in zip(flat,repeated)]).reshape(n,r,k)
    modes=np.stack([route_modes(p,s) for p,s in zip(flat,repeated)]).reshape(n,r,k)
    unique=np.array([[len(set(modes[i,j][valid[i,j]].tolist())-{-1}) for j in range(r)] for i in range(n)])
    blocked=[]
    for mode in range(3):
        blocked.append((valid & (modes!=mode) & (modes>=0)).any(-1))
    # Symmetric predeclared task rules: exclude one crossing-plane mode.
    # This is a categorical constraint test, not physical obstacle insertion.
    closure=np.stack(blocked,-1).mean(axis=(1,2))
    point={key:float(val) for key,val in metrics.items() if np.isscalar(val)}
    point['mode_exclusion_survival']=float(closure.mean())
    point['raw_candidates_per_condition']=k
    by_scene={'unique_valid':unique.mean(1),'valid_rate':valid.mean((1,2)),
              'success':valid.any(-1).mean(1),'coverage':unique.mean(1)/3,
              'mode_exclusion_survival':closure}
    rng=np.random.default_rng(918)
    boot=rng.integers(0,n,size=(2000,n))
    ci={key:np.percentile(values[boot].mean(1),[2.5,97.5]).tolist() for key,values in by_scene.items()}
    return point,ci,by_scene,valid,modes


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/routes.npz')
    p.add_argument('--features')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--splits',nargs='+',default=['val','test','ood'])
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--sample-steps',type=int,default=40)
    p.add_argument('--seed',type=int,default=2048)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    torch.set_num_threads(4)
    seed_all(a.seed)
    data=load_data(a.data,a.features)
    cp=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    assert cp['config']['dataset_sha256']==sha256(a.data),'wrong dataset checkpoint'
    assert cp['config']['cond_dim']==data['condition'].shape[-1],'wrong feature cache'
    assert cp['config'].get('features_sha256')==(sha256(a.features) if a.features else None),'wrong feature cache contents'
    model=build_model(cp['config'],data['condition'].shape[-1]).to(a.device).eval()
    model.load_state_dict(cp['model'])
    schedule=DiffusionSchedule(steps=100,device=a.device)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    all_results={}
    for split in a.splits:
        ids=np.flatnonzero(data['splits']==split)
        if split=='train': ids=ids[:128]
        repeats=1 if cp['config']['model']=='regressor' else a.repeats
        pred=[]
        if a.device.startswith('cuda'): torch.cuda.synchronize()
        begin=time.perf_counter()
        with torch.inference_mode():
            for repeat in range(repeats):
                gen=torch.Generator(device=a.device).manual_seed(a.seed+repeat)
                batches=[]
                for batch in np.array_split(ids,max(1,(len(ids)+31)//32)):
                    c=torch.as_tensor(data['condition'][batch],device=a.device)
                    s=torch.as_tensor(data['scenes'][batch],device=a.device)
                    batch_pred=predict(model,schedule,c,s,cp['config']['model'],cp['config']['candidates'],a.sample_steps,gen)
                    batches.append(batch_pred.cpu().numpy())
                pred.append(np.concatenate(batches))
        if a.device.startswith('cuda'): torch.cuda.synchronize()
        elapsed=time.perf_counter()-begin
        pred=np.stack(pred,axis=1)
        point,ci,by_scene,valid,modes=aggregate(pred,data['scenes'][ids])
        point['generation_ms_per_set']=1000*elapsed/(len(ids)*repeats)
        point['repeats']=repeats
        point['checkpoint_step']=cp['step']
        all_results[split]={'metrics':point,'scene_bootstrap_ci95':ci}
        np.savez_compressed(out/(split+'_predictions.npz'),paths=pred,scene_ids=data['scene_ids'][ids],indices=ids,valid=valid,modes=modes,**{'per_scene_'+key:val for key,val in by_scene.items()})
        print(json.dumps({split:all_results[split]}),flush=True)
    write_json(out/'metrics.json',all_results)


if __name__=='__main__':
    main()
