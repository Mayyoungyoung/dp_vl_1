"""TRAIN oracle-corridor mechanism diagnostic; never deployment numbers."""
import argparse,json
import numpy as np
import torch
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.model import load_model
from scripts.mode_geometry_experiment import PREP,POOL,DATA
from research_realized_coverage_v1.core import torch_setup,write,lines,sha,VOCAB
from scripts.evaluate_paired_modes import references
from scripts.train_verified_set import checker

def main(name,checkpoints):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    with np.load(PREP/'train.npz') as z:d={k:z[k] for k in z.files}
    with np.load(RUN/'prepared/corridors.npz') as z:c={k:z[k] for k in z.files}
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='TRAIN'}
    models={str(p):load_model(p)[0].requires_grad_(False) for p in checkpoints}
    counts={k:dict(routes=0,valid=0,correct_mode=0,length=[]) for k in ['center']+list(models)};pools={k:[] for k in counts}
    # Fixed first128TRAIN requests, no model-based selection.
    for i in range(128):
        modes=np.flatnonzero(c['mask'][i])[:8].tolist()
        while len(modes)<8:modes.append(modes[len(modes)%len(modes)])
        m=torch.tensor([modes],device='cuda');v=torch.zeros_like(m)
        centers=torch.tensor(c['centers'][i,modes,0][None],device='cuda');r=torch.tensor(c['radii'][i,modes,0][None],device='cuda')
        current=torch.tensor(d['current'][i:i+1],device='cuda');ref=references(labels[str(d['ids'][i])]);check=checker(ref)
        predictions={'center':(centers.cpu().numpy()[0],np.repeat(d['current'][i,7][None,None],8*24).reshape(8,24))}
        with torch.inference_mode():
            for k,model in models.items():
                p,e,_=model.decode(torch.tensor(d['context'][i:i+1],device='cuda'),torch.tensor(d['anchor'][i:i+1],device='cuda'),current,m,v,reference_corridor=(centers,r))
                predictions[k]=(p.cpu().numpy()[0],e.cpu().numpy()[0])
        for k,(p,e) in predictions.items():
            ok,w=check(p,e);counts[k]['routes']+=8;counts[k]['valid']+=int(ok.sum());counts[k]['correct_mode']+=sum(a==VOCAB[b] and x for a,b,x in zip(w,modes,ok))
            counts[k]['length'].extend(np.linalg.norm(np.diff(p,axis=1),axis=-1).sum(1).tolist());pools[k].append(p)
    for k,val in counts.items():val['mean_length']=float(np.mean(val.pop('length')));val['validity']=val['valid']/val['routes'];np.savez_compressed(out/('paths_'+str(list(counts).index(k))+'.npz'),paths=pools[k])
    report=dict(scope='TRAIN128,oracle supplied corridor; NOT observation deployment',models=counts,checkpoints={str(p):sha(p) for p in checkpoints},locked_access=False)
    write(out/'RESULTS.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoints',nargs='+',type=type(RUN),required=True);main(**vars(p.parse_args()))
