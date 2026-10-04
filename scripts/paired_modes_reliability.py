"""Separate route-validity scoring on frozen actual generator outputs."""
import argparse
import copy
import json
import random
import time
import numpy as np
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha,torch_setup
from scripts.paired_modes_data import RUN
from scripts.train_paired_modes import source_identity


def pool(arm,generator_seed,role):
    folder=RUN/'evaluation'/role/('%s_seed%d'%(arm,generator_seed))
    receipt=read(folder/'receipt.json');assert sha(folder/'pool.npz')==receipt['pool_sha256']
    with np.load(folder/'pool.npz') as a:data={k:a[k] for k in a.files}
    return data,receipt['pool_sha256']


def location(arm,generator_seed):return RUN/'reliability'/('%s_seed%d'%(arm,generator_seed))


def fit(arm,generator_seed,seed,resume=False):
    torch=torch_setup()
    from routeset.observed_probability import RouteValidityHead
    from routeset.train_v2 import atomic_checkpoint,rng_state,restore_rng
    from routeset.observed_training_audit import new_stream_audit,append_indices
    from torch.nn import functional as F
    tr,th=pool(arm,generator_seed,'SCORE_TRAIN');dv,dh=pool(arm,generator_seed,'DEV_SCORE')
    assert not set(tr['parents'])&set(dv['parents'])
    cfg=read(SOURCE/'configs/observed_probability_v1.json')['scorer']
    out=location(arm,generator_seed)/('q_seed%d'%seed);out.mkdir(parents=True,exist_ok=resume)
    if (out/'summary.json').exists():raise FileExistsError('Completed scorer')
    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed);rng=np.random.default_rng(seed)
    norm={}
    for key in ('nodes','context'):
        axes=tuple(range(tr[key].ndim-1));norm[key+'_mean']=tr[key].mean(axes);norm[key+'_std']=np.maximum(tr[key].std(axes),.01)
    def pack(data):return [torch.tensor((data[k]-norm[k+'_mean'])/norm[k+'_std'],device='cuda') for k in ('nodes','context')]
    tx,dx=pack(tr),pack(dv);ty,dy=[torch.tensor(d['labels'],dtype=torch.float32,device='cuda') for d in (tr,dv)]
    model=RouteValidityHead(tr['nodes'].shape[-1],tr['context'].shape[-1],cfg['width']).cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:1.)
    audit=new_stream_audit(model,rng.bit_generator.state,torch.get_rng_state())
    settings=dict(arm=arm,generator_seed=generator_seed,seed=seed,config=cfg,pool_sha256={'SCORE_TRAIN':th,'DEV_SCORE':dh},source_sha256=source_identity())
    start=0;best=float('inf');history=[];tic=time.monotonic()
    if resume:
        s=torch.load(out/'recovery.pt',map_location='cpu',weights_only=False);assert s['settings']==settings
        model.load_state_dict(s['model']);optimizer.load_state_dict(s['optimizer']);scheduler.load_state_dict(s['scheduler'])
        restore_rng(s['rng'],rng);audit=s['sampler'];start=s['step'];best=s['best'];history=s['history']
    else:write(out/'config.json',settings)
    def state(step):return dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(rng),
        sampler=audit,step=step,best=best,history=history,settings=settings,normalization=norm)
    for step in range(start+1,cfg['steps']+1):
        ids=rng.integers(len(ty),size=cfg['batch_size']);audit=append_indices(audit,ids)
        model.train();loss=F.binary_cross_entropy_with_logits(model(tx[0][ids],tx[1][ids]),ty[ids])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite q loss')
        optimizer.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step();scheduler.step()
        if step%cfg['eval_every']==0:
            model.eval()
            with torch.inference_mode():val=float(F.binary_cross_entropy_with_logits(model(*dx),dy))
            history.append(dict(step=step,train_bce=float(loss),dev_score_bce=val))
            if val<best:best=val;atomic_checkpoint(out/'best.pt',state(step))
        if step%1000==0 or step==cfg['steps']:atomic_checkpoint(out/'recovery.pt',state(step))
    atomic_checkpoint(out/'last.pt',state(cfg['steps']))
    saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(saved['model']);model.eval()
    with torch.inference_mode():logits=model(*dx).cpu().numpy()
    metrics=reliability_metrics(logits,dv)
    np.savez_compressed(out/'dev_score_predictions.npz',logits=logits,labels=dv['labels'],ids=dv['ids'])
    write(out/'summary.json',dict(metrics=metrics,train_prevalence=float(tr['labels'].mean()),
        constant_train_prevalence_brier=float(((dv['labels']-tr['labels'].mean())**2).mean()),
        best_step=saved['step'],best_sha256=sha(out/'best.pt'),last_sha256=sha(out/'last.pt'),
        elapsed_seconds=time.monotonic()-tic,peak_allocated_bytes=torch.cuda.max_memory_allocated(),updates=cfg['steps'],history=history))


def reliability_metrics(logits,data,temperature=1.):
    from routeset.observed_probability import selection_metrics
    parents=data['parents']
    if str(parents[0]).startswith('paired_family_'):parents=np.array([x.rsplit('_',1)[0] for x in parents])
    result=selection_metrics(logits,data['labels'],data['paths'],parents,temperature)
    z=np.asarray(logits,dtype=float)/temperature;y=data['labels'].astype(float);q=1/(1+np.exp(-np.clip(z,-60,60)))
    selected=z.argmax(1);rows=np.arange(len(y));sq=q[rows,selected];sy=y[rows,selected]
    result['nll']=float((np.logaddexp(0,z)-y*z).mean())
    result['selected_nll']=float((np.logaddexp(0,z[rows,selected])-sy*z[rows,selected]).mean())
    def bins(p,y):
        b=np.minimum((p*10).astype(int),9);out=[]
        for i in range(10):
            keep=b==i;out.append(dict(lower=i/10,n=int(keep.sum()),confidence=float(p[keep].mean()) if keep.any() else None,valid=float(y[keep].mean()) if keep.any() else None))
        return out
    result['reliability_bins']=bins(q,y);result['selected_reliability_bins']=bins(sq,sy)
    for key in ('reliability_bins','selected_reliability_bins'):
        result[key+'_ece']=sum(r['n']*abs(r['confidence']-r['valid']) for r in result[key] if r['n'])/sum(r['n'] for r in result[key])
    result['confident_08']=dict(candidates=int((q>=.8).sum()),observed_validity=float(y[q>=.8].mean()) if np.any(q>=.8) else None,
        selected_requests=int((sq>=.8).sum()),selected_observed_validity=float(sy[sq>=.8].mean()) if np.any(sq>=.8) else None)
    order=np.argsort(-sq,kind='stable');risk=np.cumsum(1-sy[order])/np.arange(1,len(sy)+1)
    result['selected_aurc']=float(risk.mean())
    result['risk_coverage']=[dict(coverage=float(i/len(sy)),risk=float(risk[i-1])) for i in range(1,len(sy)+1)]
    return result


def calibrate(arm,generator_seed,seed):
    torch=torch_setup()
    from routeset.observed_probability import RouteValidityHead
    from scipy.optimize import minimize_scalar
    root=location(arm,generator_seed);saved=torch.load(root/('q_seed%d'%seed)/'best.pt',map_location='cpu',weights_only=False)
    cal,ch=pool(arm,generator_seed,'CALIBRATION');norm=saved['normalization']
    model=RouteValidityHead(cal['nodes'].shape[-1],cal['context'].shape[-1],saved['settings']['config']['width']).cuda().eval()
    model.load_state_dict(saved['model'])
    def predict(data):
        x=[torch.tensor((data[k]-norm[k+'_mean'])/norm[k+'_std'],device='cuda') for k in ('nodes','context')]
        with torch.inference_mode():return model(*x).cpu().numpy()
    clog=predict(cal)
    def nll(t):
        z=clog/np.exp(t);return float((np.logaddexp(0,z)-cal['labels']*z).mean())
    fit=minimize_scalar(nll,bounds=(-3,3),method='bounded');assert fit.success
    temperature=float(np.exp(fit.x));out=root/('calibration_seed%d'%seed);out.mkdir(exist_ok=False)
    results={}
    for split in ('paired_dev','old_dev','dev32'):
        d,h=pool(arm,generator_seed,split);logits=predict(d)
        np.savez_compressed(out/(split+'.npz'),logits=logits,q=1/(1+np.exp(-logits/temperature)),labels=d['labels'],ids=d['ids'],parents=d['parents'])
        frozen=np.log(np.clip(d['q'],1e-7,1-1e-7)/(1-np.clip(d['q'],1e-7,1-1e-7)))
        results[split]=dict(pool_sha256=h,frozen_transfer=reliability_metrics(frozen,d),
            uncalibrated=reliability_metrics(logits,d),calibrated=reliability_metrics(logits,d,temperature))
    write(out/'summary.json',dict(temperature=temperature,calibration_pool_sha256=ch,calibration_nll_before=nll(0),calibration_nll_after=nll(fit.x),
        scorer_sha256=sha(root/('q_seed%d'%seed)/'best.pt'),results=results,
        scope='Temperature fitted on existing CALIBRATION only. Paired DEV is layout-shift generalization, without new-layout recalibration.'))


def package(arm,generator_seed,seed):
    torch=torch_setup()
    from routeset.observed_probability import RouteValidityHead,ProbabilisticGeometryRouteHead,ScoredRoutePlanner,load_scored_planner
    from scripts.evaluate_paired_modes import inputs_for,dataset
    root=location(arm,generator_seed);scorerfile=root/('q_seed%d'%seed)/'best.pt';state=torch.load(scorerfile,map_location='cpu',weights_only=False)
    calibration=read(root/('calibration_seed%d'%seed)/'summary.json');assert calibration['scorer_sha256']==sha(scorerfile)
    genfile=RUN/('%s_seed%d'%(arm,generator_seed))/'last.pt';genstate=torch.load(genfile,map_location='cpu',weights_only=False)
    generator=ProbabilisticGeometryRouteHead(**genstate['config']['head_options']).cuda().eval();generator.load_state_dict(genstate['model'])
    scorer=RouteValidityHead().cuda().eval();scorer.load_state_dict(state['model'])
    planner=ScoredRoutePlanner(generator,scorer,state['normalization'],calibration['temperature']).cuda().eval()
    for role in ('SCORE_TRAIN','DEV_SCORE','CALIBRATION'):
        receipt=read(RUN/'evaluation'/role/('%s_seed%d'%(arm,generator_seed))/'receipt.json')
        assert receipt['generator_checkpoint_sha256']==sha(genfile)
    folder,_=dataset('paired_dev');from scripts.run_observed_probability import lines
    row=next(r for r in lines(folder/'observations.jsonl') if r['split']=='DEV_MODEL')
    label=next(r for r in lines(folder/'supervision.jsonl') if r['id']==row['id'])
    inp=inputs_for(row,label,folder/'qwen_cache',torch,{})
    with torch.inference_mode():pred=planner(**inp,return_k=4)
    d,_=pool(arm,generator_seed,'paired_dev');idx=int(np.flatnonzero(d['ids']==row['id'])[0])
    with np.load(root/('calibration_seed%d'%seed)/'paired_dev.npz') as a:expected=a['q'][idx]
    np.testing.assert_allclose(pred['q'][0].cpu().numpy(),expected,rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(pred['paths'][0].cpu().numpy(),d['paths'][idx],rtol=0,atol=0)
    out=root/('deployment_seed%d'%seed);out.mkdir(exist_ok=False)
    torch.save(dict(planner_state=planner.state_dict(),feature_dim=4096,horizon=24,M=8,anchor_mode='straight_through_peak',
        temperature=calibration['temperature'],has_pi=True,pi_trained=False,public_outputs=['paths','events','q','selected_indices'],
        frozen_qwen_revision='89644892e4d85e24eaac8bacfd4f463576704203'),out/'planner.pt')
    restored=load_scored_planner(out/'planner.pt','cuda')
    with torch.inference_mode():replayed=restored(**inp,return_k=4)
    for k in ('paths','events','q','selected_indices'):torch.testing.assert_close(pred[k],replayed[k],rtol=0,atol=0)
    np.savez_compressed(out/'example.npz',**{k:pred[k][0].cpu().numpy() for k in ('paths','events','q','selected_indices')})
    torch.save({k:v.cpu() for k,v in inp.items()},out/'example_observed_inputs.pt')
    write(out/'manifest.json',dict(generator_sha256=sha(genfile),scorer_sha256=sha(scorerfile),planner_sha256=sha(out/'planner.pt'),
        calibration_sha256=sha(root/('calibration_seed%d'%seed)/'summary.json'),example_id=row['id'],reload_exact=True,
        reference_to_saved_pool_exact=True,public_pi=False,complete_path_states=16,
        scope='High-level tip path validity; no full-arm/controller guarantee; q scores need not sum to one.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','calibrate','package']);p.add_argument('--arm',required=True)
    p.add_argument('--generator-seed',type=int,default=0);p.add_argument('--seed',type=int,default=0);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if a.stage=='fit':fit(a.arm,a.generator_seed,a.seed,a.resume)
    elif a.stage=='calibrate':calibrate(a.arm,a.generator_seed,a.seed)
    else:package(a.arm,a.generator_seed,a.seed)
