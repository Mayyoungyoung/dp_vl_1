"""Bounded native-joint hypotheses feed a causal requested-path recurrence.

No FK, IK, solver, future measured joints, or teacher forcing in forward.
Auxiliary control receives identical labels/parameters but masks state feedback.
The direct event readout and mixture optimizer remain established components.
"""
import argparse,hashlib,os,random,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from research_selective_repair_v1.io import RUN,read,write,sha
from research_selective_repair_v1.body_prefix_forecast import prefix_features
from research_selective_repair_v1.body_crossing_measure import anchors,likelihood as event_loss,compose,compose_tensor
from scripts.run_observed_probability import torch_setup
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest

class NativeBranchHead(nn.Module):
    def __init__(self,robot,feedback=True,hypotheses=4):
        super().__init__();self.feedback=feedback;self.hypotheses=hypotheses
        for key in ('lower','upper','q0'):self.register_buffer(key,torch.tensor(robot[key],dtype=torch.float32))
        self.context=nn.Sequential(nn.Linear(128,32),nn.SiLU());self.input=nn.Sequential(nn.Linear(26,64),nn.SiLU())
        self.initial=nn.Sequential(nn.Linear(96,64),nn.Tanh());self.latent=nn.Embedding(hypotheses,16)
        self.cell=nn.GRUCell(64+32+7+16,64);self.joint=nn.Linear(64,14)
        self.hazard_one=nn.Linear(64,1)
        self.positions=nn.Sequential(nn.Linear(96,64),nn.SiLU(),nn.Linear(64,8))
        self.clear=nn.Linear(96,1);self.weights=nn.Linear(96,hypotheses)
    def forward(self,x,context,paths,completed,initial_q=None):
        b=len(x);k=self.hypotheses;z=self.input(x);cx=self.context(context);start=torch.cat([z[:,0],cx],-1)
        h=self.initial(start)[:,None].expand(-1,k,-1).reshape(b*k,64)
        mid=(self.upper+self.lower)/2;half=(self.upper-self.lower)/2
        q=self.q0.expand(b,7) if initial_q is None else initial_q
        assert q.shape==(b,7)
        qn=((q-mid)/half).clamp(-1+1e-5,1-1e-5)[:,None].expand(-1,k,-1).reshape(b*k,7)
        qstart=qn;latent=self.latent(torch.arange(k,device=x.device))[None].expand(b,-1,-1).reshape(b*k,16)
        cc=cx[:,None].expand(-1,k,-1).reshape(b*k,32);states=[];scales=[];hazards=[]
        for j in range(23):
            requested=z[:,j+1,None].expand(-1,k,-1).reshape(b*k,64)
            h=self.cell(torch.cat([requested,cc,qn if self.feedback else qstart,latent],-1),h)
            raw=self.joint(h);qn=(torch.atanh(qn.clamp(-1+1e-5,1-1e-5))+raw[:,:7]).tanh()
            states.append(qn.reshape(b,k,7));scales.append((.01+.49*raw[:,7:].sigmoid()).reshape(b,k,7))
            hazards.append(self.hazard_one(h).reshape(b,k))
        summary=torch.cat([h,cc],-1);p=self.positions(summary).reshape(b,k,8);a,has=anchors(paths,completed)
        return dict(joint_mean=torch.stack(states,2),joint_scale=torch.stack(scales,2),
            hazard=torch.stack(hazards,2),mean=a[:,None]+.1*p[...,:4].reshape(b,k,2,2),
            scale=.005+.295*p[...,4:].reshape(b,k,2,2).sigmoid(),clear=self.clear(summary).reshape(b,k),
            log_weights=self.weights(start).log_softmax(-1),has_requested_crossings=has)

def likelihood(p,d,lower,upper):
    # Keep the existing censored event objective. Joint regression supplies an
    # explicitly averaged auxiliary density to avoid counting seven coordinates
    # as seven times the evidence. Same weighting in both feedback arms.
    component=event_loss(p,d,return_components=True)
    mid=(upper+lower)/2;half=(upper-lower)/2
    target=((d['prefix_q'][:,:,-1]-mid)/half).clamp(-1+1e-5,1-1e-5)
    mu=p['joint_mean'];sigma=p['joint_scale'];error=(target[:,None]-mu)/sigma
    cdf=lambda v:.5*(1+torch.erf(v/np.sqrt(2)))
    mass=(cdf((1-mu)/sigma)-cdf((-1-mu)/sigma)).clamp_min(1e-8)
    nll=(.5*error.square()+sigma.log()+mass.log()+.5*np.log(2*np.pi)).mean(-1)
    valid=d['joint_label_valid'][:,None]
    per_component=(nll*valid).sum(-1)/valid.sum(-1).clamp_min(1)
    return -torch.logsumexp(p['log_weights']-component-per_component,-1).mean()

def fit(name,dataset,feedback=True,seed=0,steps=2400,resume=False,stop_after=None,threads=4):
    torch_setup();torch.set_num_threads(threads)
    out=RUN/name;out.mkdir(parents=True,exist_ok=resume);assert not(out/'last.pt').exists()
    manifest=read(dataset.parent/'MANIFEST.json');assert manifest['native_joint_labels'] and manifest['no_DEV_feedback'] and sha(dataset)==manifest['samples_sha256']
    with np.load(dataset) as z:d={k:z[k] for k in z.files}
    assert set(d['roles'])=={'TRAIN'};families=sorted(set(d['families']));held=np.isin(d['families'],families[-2:]);train=~held
    x=prefix_features(d['paths'],d['completed']);mean=x[train].mean((0,1));std=x[train].std((0,1)).clip(.05)
    settings=dict(feedback=feedback,seed=seed,steps=steps,batch=32,lr=.0003,threads=threads,hypotheses=4,
        robot=manifest['public_joint_contract'],dataset_sha256=sha(dataset),mean=mean.tolist(),std=std.tolist(),source_commit=os.environ.get('CODE_COMMIT'),
        fit_families=list(map(str,families[:-2])),diagnostic_TRAIN_families=list(map(str,families[-2:])),
        scope='Native endpoint joint labels only in averaged truncated-normal auxiliary loss; no FK/IK/teacher forcing. Exact shared control masks predicted-joint feedback.',
        joint_density='Normalized bounded coordinate truncated diagonal Gaussian; average 7coordinates and observed segments, fixed weight1')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    model=NativeBranchHead(settings['robot'],feedback).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
    t=lambda v:torch.as_tensor(v,device='cuda');tx=t(((x-mean)/std).astype(np.float32));tc=t(d['context'].astype(np.float32));paths=t(d['paths']);completed=t(d['completed']);q0=t(d['initial_q'])
    keys=('prefix_hazard','prefix_observed','event_tip','event_present','prefix_valid','labels','prefix_q','joint_label_valid')
    target={key:t(d[key]) for key in keys};initial=tensor_state_digest(model.state_dict());history=[];stream='';start=0;tic=time.monotonic();indices=np.flatnonzero(train)
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,history=history,stream=stream,initial_sha256=initial)
    if resume:
        ck=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert ck['settings']==settings
        model.load_state_dict(ck['model']);opt.load_state_dict(ck['optimizer']);restore_rng(ck['rng'],rng);start=ck['step'];history=ck['history'];stream=ck['stream']
    else:write(out/'config.json',settings)
    step=start
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ix=rng.choice(indices,size=32,replace=True);stream=hashlib.sha256(stream.encode()+ix.tobytes()).hexdigest()
        model.train();p=model(tx[ix],tc[ix],paths[ix],completed[ix],q0[ix]);loss=likelihood(p,{key:v[ix] for key,v in target.items()},model.lower,model.upper)
        assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:history.append(dict(step=step,loss=float(loss)));atomic_checkpoint(out/'recovery.pt',state(step));print(history[-1],flush=True)
    atomic_checkpoint(out/'recovery.pt',state(step))
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));model.eval();parts={}
        with torch.no_grad():
            for start in range(0,len(x),64):
                ix=slice(start,start+64)
                for key,v in model(tx[ix],tc[ix],paths[ix],completed[ix],q0[ix]).items():parts.setdefault(key,[]).append(v.cpu().numpy())
        p={key:np.concatenate(v) for key,v in parts.items()};np.savez_compressed(out/'TRAIN_diagnostic_predictions.npz',**p,ids=d['ids'],slots=d['slots'],options=d['options'],families=d['families'],heldout=held)
        lo=np.asarray(settings['robot']['lower']);hi=np.asarray(settings['robot']['upper']);q=p['joint_mean']*(hi-lo)/2+(hi+lo)/2;choose=p['log_weights'].argmax(-1);best=q[np.arange(len(q)),choose]
        report={}
        for label,mask in [('fit',train),('held_TRAIN',held)]:
            valid=d['joint_label_valid'][mask];error=best[mask]-d['prefix_q'][mask,:,-1]
            report[label]=dict(rows=int(mask.sum()),joint_endpoint_RMSE_rad=float(np.sqrt(np.square(error)[valid].mean())),valid_joint_endpoints=int(valid.sum()),predicted_joint_bound_violations=int(((q[mask]<lo-1e-6)|(q[mask]>hi+1e-6)).sum()))
        write(out/'SUMMARY.json',dict(settings=settings,seconds=time.monotonic()-tic,checkpoint_sha256=sha(out/'last.pt'),TRAIN_diagnostics=report,stream=stream,locked_access=False))

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);s=ck['settings'];model=NativeBranchHead(s['robot'],s['feedback'],s['hypotheses']).cuda().eval().requires_grad_(False);model.load_state_dict(ck['model']);return model,ck

def predict(model,ck,paths,completed,context):
    x=prefix_features(paths,completed);s=ck['settings'];t=lambda v:torch.tensor(v,device='cuda',dtype=torch.float32)
    with torch.no_grad():p=model(t((x-np.asarray(s['mean']))/np.asarray(s['std'])),t(context),t(paths),t(completed))
    return {key:v.cpu().numpy() for key,v in p.items() if not key.startswith('joint_')}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--aux-only',action='store_true');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=2400);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);a=vars(p.parse_args());a['feedback']=not a.pop('aux_only');fit(**a)
