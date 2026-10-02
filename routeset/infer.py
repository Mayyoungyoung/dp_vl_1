"""Export one task's raw candidate paths, calibrated confidence, and geometry."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .common import encode_paths, load_data, sha256, write_json
from .diffusion import DiffusionSchedule
from .geometry import camera_matrices, path_validity, project_points, route_modes
from .models import Critic
from .train import build_model, predict


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/routes.npz')
    p.add_argument('--features')
    p.add_argument('--index',type=int,default=896,help='Default is the first held-out test scene')
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--critic')
    p.add_argument('--output',default='reports/demo_candidates.json')
    p.add_argument('--seed',type=int,default=2048)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    data=load_data(a.data,a.features)
    cp=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    assert cp['config']['dataset_sha256']==sha256(a.data)
    assert cp['config'].get('features_sha256')==(sha256(a.features) if a.features else None)
    model=build_model(cp['config'],data['condition'].shape[-1]).to(a.device).eval()
    model.load_state_dict(cp['model'])
    scene=torch.as_tensor(data['scenes'][a.index:a.index+1],device=a.device)
    condition=torch.as_tensor(data['condition'][a.index:a.index+1],device=a.device)
    schedule=DiffusionSchedule(100,device=a.device)
    gen=torch.Generator(device=a.device).manual_seed(a.seed)
    with torch.inference_mode():
        paths=predict(model,schedule,condition,scene,cp['config']['model'],cp['config']['candidates'],40,gen)
        confidence=None
        if a.critic:
            ck=torch.load(a.critic,map_location=a.device,weights_only=False)
            assert ck['config']['dataset_sha256']==sha256(a.data)
            assert ck['config'].get('features_sha256')==(sha256(a.features) if a.features else None)
            critic=Critic(cond_dim=ck['config']['cond_dim'],horizon=ck['config']['horizon'],width=ck['config']['width']).to(a.device).eval()
            critic.load_state_dict(ck['model'])
            logits=critic(encode_paths(paths,scene),condition)
            confidence=torch.sigmoid(logits*ck['platt_slope']+ck['platt_bias'])[0].cpu().numpy()
    xyz=paths[0].cpu().numpy()
    geom=data['scenes'][a.index]
    check=path_validity(xyz,geom)
    modes=route_modes(xyz,geom)
    paired=project_points(xyz)
    payload={'scene_id':str(data['scene_ids'][a.index]),'split':str(data['splits'][a.index]),
             'instruction':str(data['instructions'][a.index]),'scene':geom.tolist(),
             'model':cp['config']['model'],'candidate_budget':len(xyz),'generation_calls':1 if cp['config']['model']=='regressor' else 40,
             'geometry_input':'Privileged synthetic obstacle and task anchors',
             'confidence_scope':'Probability of satisfying toy task-layer checker; not robot success',
             'camera_matrices':camera_matrices().tolist(),'candidates':[]}
    for i,path in enumerate(xyz):
        payload['candidates'].append({'xyz':path.tolist(),'paired_2d_pixels':paired[:,i].tolist(),
                                      'confidence':float(confidence[i]) if confidence is not None else None,
                                      'checker_valid':bool(check['valid'][i]),'passage_mode':int(modes[i]),
                                      'minimum_clearance':float(check['clearance'][i]),'length':float(check['lengths'][i])})
    write_json(a.output,payload)
    print(json.dumps({'output':str(Path(a.output).resolve()),'valid':[c['checker_valid'] for c in payload['candidates']],
                      'confidence':[c['confidence'] for c in payload['candidates']]}),flush=True)


if __name__=='__main__': main()
