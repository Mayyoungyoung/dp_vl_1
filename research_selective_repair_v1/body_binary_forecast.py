"""Separately trained binary successful-clear risk control, matched feedback."""
import argparse,hashlib,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from research_selective_repair_v1.io import RUN,write,sha
from research_selective_repair_v1.body_forecast import OutcomeHead,features
from scripts.run_observed_probability import torch_setup
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest

class BinaryHead(OutcomeHead):
    def __init__(self):
        super().__init__('recurrent');self.output[-1]=nn.Linear(64,1)

def fit(name,dataset,seed=0,steps=2400,resume=False,stop_after=None,threads=4):
    torch_setup();assert threads in (1,4);torch.set_num_threads(threads);out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed immutable fit')
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'}
    families=sorted(set(d['families']));held=families[-2:];train=~np.isin(d['families'],held);test=~train
    x=features(d['paths'],d['completed']);mean=x[train].mean((0,1));std=x[train].std((0,1)).clip(.05)
    y=(d['labels']>0).astype(np.float32)
    settings=dict(seed=seed,steps=steps,batch=32,lr=.0003,threads=threads,dataset_sha256=sha(dataset),mean=mean.tolist(),std=std.tolist(),
        fit_families=[str(v) for v in families[:-2]],diagnostic_TRAIN_families=[str(v) for v in held],source_commit=os.environ.get('CODE_COMMIT'),
        scope='Unweighted binary successful-clear BCE,actual same TRAIN feedback; no categorical word supervision')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    model=BinaryHead().cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    tx=torch.tensor((x-mean)/std,device='cuda');tc=torch.tensor(d['context'],device='cuda');ty=torch.tensor(y,device='cuda')
    initial=tensor_state_digest(model.state_dict());history=[];stream='';start=0;tic=time.monotonic();indices=np.flatnonzero(train)
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    step=start
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.choice(indices,size=32,replace=True);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        model.train();loss=nn.functional.binary_cross_entropy_with_logits(model(tx[ix],tc[ix]).flatten(),ty[ix])
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss)));atomic_checkpoint(out/'recovery.pt',state(step));print(history[-1],flush=True)
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));model.eval()
        with torch.no_grad():prob=model(tx,tc).sigmoid().flatten().cpu().numpy()
        np.savez_compressed(out/'TRAIN_diagnostic_predictions.npz',success=prob,labels=y,heldout=test,ids=d['ids'],slots=d['slots'],options=d['options'],families=d['families'])
        write(out/'SUMMARY.json',dict(settings=settings,seconds=time.monotonic()-tic,checkpoint_sha256=sha(out/'last.pt'),TRAIN_fit_brier=float(np.square(prob[train]-y[train]).mean()),TRAIN_heldout_brier=float(np.square(prob[test]-y[test]).mean()),TRAIN_heldout_accuracy=float(((prob[test]>=.5)==y[test]).mean()),stream=stream,locked_access=False))

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);model=BinaryHead().cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);return model,ck

def predict(model,ck,paths,completed,context):
    x=features(paths,completed);s=ck['settings'];t=lambda v:torch.tensor(v,device='cuda',dtype=torch.float32)
    with torch.no_grad():p=model(t((x-np.asarray(s['mean']))/np.asarray(s['std'])),t(context)).sigmoid()
    return p.flatten().cpu().numpy()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',required=True,type=Path);p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2400);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--threads',type=int,choices=[1,4],default=4);fit(**vars(p.parse_args()))
