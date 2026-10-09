"""Equal information/draws/init and full optimizer/RNG state; oracle labels TRAIN only."""
import argparse,hashlib,os,random,time
import numpy as np
import torch
from scripts.mode_geometry_experiment import POOL,PREP
from research_realized_coverage_v1.core import BASE,load_generator,torch_setup,sha,write,read
from research_feasible_space_v1.prepare import RUN
from research_feasible_space_v1.model import FeasibleSpaceHead
from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
from routeset.observed_training_audit import tensor_state_digest
from routeset.segment_clearance import segment_clearance_loss
from routeset.paired_modes import workspace_floor_loss

def train(name,arm,steps=2400,seed=0,peer_boundaries=False,oracle=False,resume=False,stop_after=None):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=resume)
    if (out/'last.pt').exists():raise FileExistsError('Completed checkpoint')
    with np.load(PREP/'train.npz') as z:d={k:z[k] for k in z.files}
    with np.load(POOL) as z:s={k:z[k] for k in z.files}
    with np.load(RUN/'prepared/corridors.npz') as z:c={k:z[k] for k in z.files}
    assert np.array_equal(d['ids'],s['ids']) and np.array_equal(c['ids'],s['ids']) and set(s['splits'])=={'TRAIN'}
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    base,baseck=load_generator();model=FeasibleSpaceHead(base,arm,peer_boundaries).cuda();model.train()
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=.0001)
    initial=tensor_state_digest(model.state_dict());frozen=tensor_state_digest(model.base.state_dict())
    t={k:torch.as_tensor(d[k],device='cuda') for k in ('context','anchor','current','centers','halves','floors')}
    tc={k:torch.as_tensor(c[k],device='cuda') for k in ('centers','radii')}
    from routeset.mode_geometry import VOCAB
    bm=[{VOCAB.index(w):np.flatnonzero(s['mask'][i]&(s['modes'][i]==w)) for w in sorted(set(s['modes'][i,s['mask'][i]]))} for i in range(len(d['ids']))]
    settings=dict(arm=arm,steps=steps,seed=seed,peer_boundaries=peer_boundaries,oracle=oracle,lr=.0001,batch_size=32,
        generator_sha256=sha(BASE),support_sha256=sha(POOL),prepared_sha256=sha(PREP/'train.npz'),corridors_sha256=sha(RUN/'prepared/corridors.npz'),
        source_commit=os.environ.get('CODE_COMMIT'),loss='route + corridor_center + radius_mse +160clear +.01event; oracle excludes corridor losses')
    history=[];stream='';start=0;tic=time.monotonic()
    def state(step):return dict(model=model.state_dict(),optimizer=opt.state_dict(),rng=rng_state(rng),step=step,settings=settings,
        config=baseck['config'],history=history,stream=stream,initial_tensor_sha256=initial,frozen_sha256=frozen)
    if resume:
        saved=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert saved['settings']==settings
        model.load_state_dict(saved['model']);opt.load_state_dict(saved['optimizer']);restore_rng(saved['rng'],rng);history=saved['history'];stream=saved['stream'];start=saved['step']
    else:write(out/'config.json',settings)
    for step in range(start+1,min(steps,stop_after or steps)+1):
        ids=rng.integers(len(d['ids']),size=32);ms=[];vs=[];targets=[]
        for i in ids:
            m=rng.permutation(sorted(bm[i]))[:8].tolist()
            while len(m)<8:m.append(int(rng.choice(sorted(bm[i]))))
            choices=[int(rng.choice(bm[i][j])) for j in m]
            ms.append(m);vs.append(c['assignment'][i,choices]);targets.append(s['paths'][i,choices])
        m=np.asarray(ms,np.int64);v=np.asarray(vs,np.int64);target=np.asarray(targets,np.float32)
        stream=hashlib.sha256(stream.encode()+ids.tobytes()+m.tobytes()+v.tobytes()+target.tobytes()).hexdigest()
        mt=torch.as_tensor(m,device='cuda');vt=torch.as_tensor(v,device='cuda');ii=torch.as_tensor(ids,device='cuda')[:,None]
        center=tc['centers'][ii,mt,vt];radius=tc['radii'][ii,mt,vt];truth=torch.as_tensor(target,device='cuda')
        p,e,info=model.decode(t['context'][ids],t['anchor'][ids],t['current'][ids],mt,variant_ids=vt,reference_corridor=(center,radius) if oracle else None)
        route=(p[:,:,1:]-truth[:,:,1:]).square().mean()
        center_loss=(info['predicted_centers'][:,:,1:]-center[:,:,1:]).square().mean()
        radius_loss=(info['predicted_radii']-radius).square().mean()
        clear=segment_clearance_loss(p,t['centers'][ids],t['halves'][ids])+workspace_floor_loss(p,t['floors'][ids])
        event=(e-t['current'][ids,None,None,7]).square().mean()
        loss=route+160*clear+.01*event+(center_loss+radius_loss if not oracle else 0)
        if step==1:
            trainable=[p for p in model.parameters() if p.requires_grad]
            norms={}
            for key,value in [('route',route),('weighted_clear',160*clear),('corridor',center_loss+radius_loss)]:
                g=torch.autograd.grad(value,trainable,allow_unused=True,retain_graph=True)
                norms[key]=float(torch.sqrt(sum(x.square().sum() for x in g if x is not None)))
            write(out/'INITIAL_GRADIENTS.json',norms)
        opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step()
        if step%100==0:
            history.append(dict(step=step,loss=float(loss),route=float(route),clear=float(clear),center=float(center_loss),radius=float(radius_loss)))
            print(history[-1],flush=True)
        if step in (600,1200,2400) or step==steps:atomic_checkpoint(out/('step%d.pt'%step),state(step))
        if step%100==0 or step==min(steps,stop_after or steps):atomic_checkpoint(out/'recovery.pt',state(step))
    assert frozen==tensor_state_digest(model.base.state_dict())
    if step==steps:
        atomic_checkpoint(out/'last.pt',state(step));write(out/'SUMMARY.json',dict(settings=settings,elapsed_seconds=time.monotonic()-tic,
            last_sha256=sha(out/'last.pt'),initial_tensor_sha256=initial,stream=stream,trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--arm',choices=['xyz','relative','bounded'],required=True)
    p.add_argument('--steps',type=int,default=2400);p.add_argument('--seed',type=int,default=0);p.add_argument('--peer-boundaries',action='store_true');p.add_argument('--oracle',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int);train(**vars(p.parse_args()))
