"""Counterfactual finite-controller state transport from current legal inputs.

Teacher prefix states exist only inside TRAIN likelihood evaluation. Deployment
rolls out all hypotheses from public initial joints without controller queries.
"""
import argparse,hashlib,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_forecast import features
from research_selective_repair_v1.public_kinematics import load as load_robot
from scripts.run_observed_probability import torch_setup
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest

def prefix_features(paths,completed):
    x=features(paths,completed)
    # A future requested endpoint is not an input to earlier controller calls.
    x[:,:,23:]=0
    return x

def tip_forward(q,qref,base,relative):
    result=base.expand(q.shape[:-1]+(4,4))
    delta=q-qref
    for j in range(7):
        c,s=delta[...,j].cos(),delta[...,j].sin()
        zero=torch.zeros_like(c);one=torch.ones_like(c)
        rot=torch.stack([c,-s,zero,zero,s,c,zero,zero,zero,zero,one,zero,zero,zero,zero,one],-1).reshape(q.shape[:-1]+(4,4))
        result=result@rot@relative[j]
    return result[...,:3,3]

class PrefixHead(nn.Module):
    def __init__(self,robot,kind='recurrent',hypotheses=4):
        super().__init__();assert kind in ('recurrent','nonrecurrent');self.kind=kind;self.hypotheses=hypotheses
        q,b,r=robot
        for name,value in zip(('qref','base','relative'),(q,b,r)):
            self.register_buffer(name,torch.as_tensor(value,dtype=torch.float32))
        self.context=nn.Sequential(nn.Linear(128,32),nn.SiLU())
        self.input=nn.Sequential(nn.Linear(26,64),nn.SiLU())
        self.latent=nn.Embedding(hypotheses,16)
        self.initial=nn.Sequential(nn.Linear(96,64),nn.Tanh())
        if kind=='recurrent':self.cell=nn.GRUCell(94,64)
        else:self.cell=nn.Sequential(nn.Linear(94,160),nn.SiLU(),nn.Linear(160,80),nn.SiLU(),nn.Linear(80,64),nn.SiLU())
        self.output=nn.Linear(64,45);self.weights=nn.Linear(96,hypotheses)
    def forward(self,x,context,q0,teacher=None,forcing=0.):
        b=len(x);k=self.hypotheses;z=self.input(x)
        # Fixed executor processes requested waypoints sequentially. Future
        # route nodes cannot change a predicted already executed prefix.
        cx=torch.cat([z[:,0],self.context(context)],-1)
        h=self.initial(cx)[:,None].expand(-1,k,-1).reshape(b*k,64)
        latent=self.latent(torch.arange(k,device=x.device))[None].expand(b,-1,-1).reshape(b*k,16)
        previous=q0[:,None].expand(-1,k,-1).reshape(b*k,7)
        states=[];hazards=[];event_states=[];event_logits=[]
        for j in range(23):
            inp=torch.cat([z[:,j+1,None].expand(-1,k,-1).reshape(b*k,64),previous.sin(),previous.cos(),latent],-1)
            h=self.cell(inp,h) if self.kind=='recurrent' else self.cell(inp)
            prediction=self.output(h)
            q=previous[:,None]+np.pi*prediction[:,:28].reshape(b*k,4,7).tanh()
            states.append(q.reshape(b,k,4,7));hazards.append(prediction[:,28].reshape(b,k))
            event=previous[:,None]+np.pi*prediction[:,29:43].reshape(b*k,2,7).tanh()
            event_states.append(event.reshape(b,k,2,7));event_logits.append(prediction[:,43:].reshape(b,k,2))
            previous=q[:,-1]
            if teacher is not None and forcing>0:
                use=(torch.rand((b,1,1),device=x.device)<forcing)&teacher['valid'][:,j,None,None]
                previous=torch.where(use,teacher['q'][:,j,-1,None].expand(-1,k,-1),previous.reshape(b,k,7)).reshape(b*k,7)
        q=torch.stack(states,2);hazard=torch.stack(hazards,2)
        tip=tip_forward(q,self.qref,self.base,self.relative)
        event_q=torch.stack(event_states,2)
        return dict(q=q,tip=tip,hazard=hazard,event_q=event_q,
            event_tip=tip_forward(event_q,self.qref,self.base,self.relative),
            events=torch.stack(event_logits,2),log_weights=self.weights(cx).log_softmax(-1))

def likelihood(pred,q,tip,valid,hazard,observed,event=None):
    # Fixed Gaussian scales .1rad and.02m; no suffix supervision or word index.
    qloss=((pred['q']-q[:,None])/.1).square().mean((-1,-2))
    ploss=((pred['tip']-tip[:,None])/.02).square().mean((-1,-2))
    state=((qloss+ploss)*valid[:,None]).sum(-1)/valid.sum(-1).clamp_min(1)[:,None]
    target=hazard[:,None].expand_as(pred['hazard'])
    risk=nn.functional.binary_cross_entropy_with_logits(pred['hazard'],target,reduction='none')
    # Censored first-failure likelihood: product of continuation probabilities
    # until observed stop,then failure. Averaging by prefix length incorrectly
    # changes that distribution while deployment still multiplies all factors.
    risk=(risk*observed[:,None]).sum(-1)
    if event is not None:
        qevent=((pred['event_q']-event['q'][:,None])/.1).square().mean(-1)
        pevent=((pred['event_tip']-event['tip'][:,None])/.02).square().mean(-1)
        mask=event['present'][:,None]
        state+=((qevent+pevent)*mask).sum((-1,-2))/mask.sum((-1,-2)).clamp_min(1)
        truth=event['present'][:,None].expand_as(pred['events']).float()
        ce=nn.functional.binary_cross_entropy_with_logits(pred['events'],truth,reduction='none')
        mask=event['observed'][:,None]
        risk=risk+(ce*mask).sum((-1,-2))
    return -torch.logsumexp(pred['log_weights']-.5*state-risk,dim=-1).mean()

def fit(name,dataset,kind='recurrent',seed=0,steps=2400,resume=False,stop_after=None,threads=4):
    torch_setup();assert threads in (1,4);torch.set_num_threads(threads);out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed immutable fit')
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'}
    manifest=read(dataset.parent/'MANIFEST.json');assert manifest['no_DEV_feedback'] and manifest['max_FK_tip_error_m']<=.005
    families=sorted(set(d['families']));held=families[-2:];train=~np.isin(d['families'],held);test=~train
    x=prefix_features(d['paths'],d['completed']);mean=x[train].mean((0,1));std=x[train].std((0,1)).clip(.05)
    robot=dataset.parent/'public_robot.npz'
    settings=dict(kind=kind,seed=seed,steps=steps,batch=32,lr=.0003,threads=threads,hypotheses=4,joint_sigma_rad=.1,tip_sigma_m=.02,
        dataset_sha256=sha(dataset),robot_sha256=sha(robot),mean=mean.tolist(),std=std.tolist(),
        fit_families=[str(v) for v in families[:-2]],diagnostic_TRAIN_families=[str(v) for v in held],
        source_commit=os.environ.get('CODE_COMMIT'),forcing='linear1to0infirst1200updates;free deployment',
        event_supervision='Exact first crossings from full trace; uniform4states alone alias7/85 TRAIN successes',
        hazard_likelihood='Sum observed conditional stop/continue log factors,censored unknown suffix',
        event_likelihood='Sum firstcross/preceding noncross log factors,censored after firstcross/stop',
        scope='Current observation/public initial q only in free forward; actual prefix/event/hazard TRAIN likelihood only')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    model=PrefixHead(load_robot(robot),kind).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    t=lambda v:torch.as_tensor(v,device='cuda')
    tx=t(((x-mean)/std).astype(np.float32));tc=t(d['context'].astype(np.float32));tq0=t(d['initial_q'].astype(np.float32))
    tq=t(d['prefix_q']);tp=t(d['prefix_tip']);tv=t(d['prefix_valid']);th=t(d['prefix_hazard']);to=t(d['prefix_observed'])
    teq=t(d['event_q']);tep=t(d['event_tip']);tev=t(d['event_present']);teo=t(d['event_observed'])
    initial=tensor_state_digest(model.state_dict());history=[];stream='';start=0;tic=time.monotonic();indices=np.flatnonzero(train)
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    step=start
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.choice(indices,size=32,replace=True);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        model.train();pred=model(tx[ix],tc[ix],tq0[ix],dict(q=tq[ix],valid=tv[ix]),max(0,1-step/1200))
        loss=likelihood(pred,tq[ix],tp[ix],tv[ix],th[ix],to[ix],dict(q=teq[ix],tip=tep[ix],present=tev[ix],observed=teo[ix]));assert torch.isfinite(loss)
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss),forcing=max(0,1-step/1200)));atomic_checkpoint(out/'recovery.pt',state(step));print(history[-1],flush=True)
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));model.eval();predictions=[]
        with torch.no_grad():
            for start in range(0,len(x),32):
                sl=slice(start,start+32);p=model(tx[sl],tc[sl],tq0[sl]);predictions.append({k:v.cpu().numpy() for k,v in p.items()})
        p={k:np.concatenate([v[k] for v in predictions]) for k in predictions[0]}
        np.savez_compressed(out/'TRAIN_diagnostic_predictions.npz',**p,ids=d['ids'],slots=d['slots'],options=d['options'],families=d['families'],heldout=test)
        report={}
        for label,mask in (('fit',train),('heldout',test)):
            chosen=p['log_weights'][mask].argmax(-1);ix=np.flatnonzero(mask)
            tip=p['tip'][ix,chosen];joint=p['q'][ix,chosen];valid=d['prefix_valid'][mask]
            # Posterior-oracle component metric separated from deployed MAP.
            err=np.linalg.norm(p['tip'][ix]-d['prefix_tip'][mask,None],axis=-1).mean(-1)
            oracle=(err*valid[:,None]).sum(-1)/valid.sum(-1).clip(1)[:,None]
            success=np.sum(np.exp(p['log_weights'][mask])*np.prod(1/(1+np.exp(np.clip(p['hazard'][mask],-40,40))),-1),-1)
            event_tip=p['event_tip'][ix,chosen];event_valid=d['event_present'][mask]
            event_prob=np.sum(np.exp(p['log_weights'][mask,:,None,None])/(1+np.exp(np.clip(-p['events'][mask],-40,40))),1)
            report[label]=dict(MAP_prefix_tip_rmse_m=float(np.sqrt(np.square(tip-d['prefix_tip'][mask])[valid].mean())),
                MAP_prefix_joint_rmse_rad=float(np.sqrt(np.square(joint-d['prefix_q'][mask])[valid].mean())),
                MAP_first_crossing_tip_rmse_m=float(np.sqrt(np.square(event_tip-d['event_tip'][mask])[event_valid].mean())),
                first_crossing_occurrence_brier=float(np.square(event_prob-d['event_present'][mask])[d['event_observed'][mask]].mean()),
                oracle_component_mean_tip_error_m=float(oracle.min(-1).mean()),
                completion_proxy_vs_successful_clear_brier=float(np.square(success-(d['labels'][mask]>0)).mean()),rows=int(mask.sum()),
                scope='Free rollout; successful-prefix errors omit unknown suffix; oracle component is diagnostic only')
        write(out/'SUMMARY.json',dict(settings=settings,seconds=time.monotonic()-tic,checkpoint_sha256=sha(out/'last.pt'),TRAIN_diagnostics=report,stream=stream,locked_access=False))

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);s=ck['settings'];m=ck['model']
    robot=tuple(m[k].numpy() for k in ('qref','base','relative'))
    model=PrefixHead(robot,s['kind'],s['hypotheses']).cuda().eval().requires_grad_(False);model.load_state_dict(m)
    return model,ck

def predict(model,ck,paths,completed,context,q0=None):
    x=prefix_features(paths,completed);s=ck['settings'];t=lambda v:torch.tensor(v,device='cuda',dtype=torch.float32)
    q0=np.broadcast_to(model.qref.cpu().numpy(),(len(x),7)) if q0 is None else q0
    with torch.no_grad():p=model(t((x-np.asarray(s['mean']))/np.asarray(s['std'])),t(context),t(q0))
    return {k:v.cpu().numpy() for k,v in p.items()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',required=True,type=Path);p.add_argument('--kind',choices=['recurrent','nonrecurrent'],default='recurrent');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2400);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);p.add_argument('--threads',type=int,choices=[1,4],default=4);fit(**vars(p.parse_args()))
