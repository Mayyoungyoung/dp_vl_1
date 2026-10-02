"""Strict K=4 self-draft rollout: generate 2, then complete 2, no discarded route."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from routeset.common import decode_paths, sha256, write_json
from routeset.completion import CompletionRegressor, draft_validity
from routeset.multigate import load_dataset, route_metrics, path_validity


@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True); p.add_argument('--runs',required=True)
    p.add_argument('--output',required=True); p.add_argument('--device',default='cpu')
    a=p.parse_args(); torch.set_num_threads(1)
    data=load_dataset(a.data,'DEV_MODEL'); out=Path(a.output); out.mkdir(parents=True,exist_ok=True)
    results=[]
    for mechanism in ['attention','coverage']:
        cp=Path(a.runs)/mechanism/'best.pt'
        checkpoint=torch.load(cp,map_location=a.device,weights_only=False); conf=checkpoint['config']
        model=CompletionRegressor(conf['cond_dim'],conf['horizon'],conf['width'],conf['depth'],mechanism).to(a.device).eval()
        model.load_state_dict(checkpoint['model'])
        for mode in ['joint4','self2_then2']:
            predictions=[]; times=[]; selectors=[]
            for row,scene in enumerate(data['scenes']):
                tic=time.perf_counter(); c=torch.as_tensor(scene[None],device=a.device)
                empty=torch.zeros((1,2,conf['horizon'],3),device=a.device)
                none=torch.zeros((1,2),dtype=torch.bool,device=a.device)
                if mode=='joint4':
                    paths=decode_paths(model(c,empty,none,k=4),c).cpu().numpy()
                else:
                    first=decode_paths(model(c,empty,none,k=2),c).cpu().numpy()
                    gate=draft_validity(first,scene[None],np.ones((1,2),bool))
                    second=decode_paths(model(c,torch.as_tensor(first,device=a.device),torch.as_tensor(gate,device=a.device),k=2),c).cpu().numpy()
                    paths=np.concatenate([first,second],axis=1)
                checks=path_validity(paths[0],scene)
                selectors.append(checks['valid'].astype(float)*100-checks['lengths'])
                times.append((time.perf_counter()-tic)*1000)
                predictions.append(paths[0])
            predictions=np.stack(predictions)
            metrics=route_metrics(predictions,data['scenes'],data['modes'],data['path_mask'],scores=np.stack(selectors))
            summary={k:float(v) for k,v in metrics.items() if np.isscalar(v)}
            summary.update(mechanism=mechanism,mode=mode,K=4,forward_passes=1 if mode=='joint4' else 2,
                           total_generated_routes=4,discarded_routes=0,latency_scope='CPU batch1 conditioning transfer + generation + gate + output checks; no VLM',
                           median_ms=float(np.median(times[5:])),p95_ms=float(np.percentile(times[5:],95)),
                           checkpoint_sha256=sha256(cp),split='DEV_MODEL',seed=conf['seed'])
            results.append(summary)
            folder=out/(mechanism+'_'+mode);folder.mkdir(exist_ok=True)
            np.savez_compressed(folder/'predictions.npz',paths=predictions,scene_ids=data['scene_ids'])
            per_scene=[]
            for row in range(len(predictions)):
                m=route_metrics(predictions[row:row+1],data['scenes'][row:row+1],data['modes'][row:row+1],data['path_mask'][row:row+1])
                per_scene.append(dict(scene_id=str(data['scene_ids'][row]),**{k:float(v) for k,v in m.items() if np.isscalar(v)}))
            write_json(folder/'per_scene.json',per_scene)
    write_json(out/'results.json',results);print(json.dumps(results))


if __name__=='__main__':main()
