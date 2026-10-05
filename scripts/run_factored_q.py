"""Frozen R1 pools -> task/geometry factors, with role-separated calibration."""
import argparse
import copy
import math
import os
import random
import time
import numpy as np
from scripts.run_observed_probability import ROOT, SOURCE, read, write, sha, torch_setup
from scripts.train_paired_modes import source_identity

RUN = ROOT/'runs/factored_q_v1'
OLD = ROOT/'runs/paired_modes_v1'
POLICY = SOURCE/'configs/factored_q_v1.json'
ROLES = ('SCORE_TRAIN', 'DEV_SCORE', 'CALIBRATION', 'paired_dev', 'old_dev', 'dev32')


def load_pool(seed, role):
    if role not in ROLES:
        raise ValueError('Unregistered role')
    from routeset.factored_q import factor_labels
    base = OLD/'evaluation'/role/('R1_seed%d' % seed)
    receipt = read(base/'receipt.json')
    digest = sha(base/'pool.npz')
    if digest != receipt['pool_sha256'] or receipt['generator_checkpoint_sha256'] != sha(OLD/('R1_seed%d' % seed)/'last.pt'):
        raise ValueError('Original frozen pool or generator changed')
    with np.load(base/'pool.npz') as z:
        data = {k:z[k] for k in z.files}
    scenes = read(base/'per_scene.json')
    if [s['id'] for s in scenes] != list(data['ids']):
        raise ValueError('Checker/pool order differs')
    t, f = zip(*(factor_labels(s['candidates']) for s in scenes))
    data.update(task=np.array(t), feas=np.array(f))
    np.testing.assert_array_equal(data['task'] & data['feas'], data['labels'])
    if role in ('paired_dev', 'old_dev', 'dev32'):
        rows = {r['id']:r for r in read(base/'rows.json')}
        data['words'] = [rows[str(i)]['words'] for i in data['ids']]
    return data, dict(pool=digest, factor_source=sha(base/'per_scene.json'),
                      receipt=sha(base/'receipt.json'), generator=receipt['generator_checkpoint_sha256'])


def folder(arm, gs, ss):
    return RUN/('%s_g%d_s%d' % (arm, gs, ss))


def fit(arm, gs, ss, resume=False, name=None, steps=None, stop_after=None):
    torch = torch_setup()
    from routeset.factored_q import FactorRouteScorer, training_loss, joint_nll
    from routeset.train_v2 import atomic_checkpoint, rng_state, restore_rng
    from routeset.observed_training_audit import new_stream_audit, append_indices, tensor_state_digest
    cfg = read(POLICY)
    if steps is not None:
        cfg = dict(cfg, steps=steps)
    tr, th = load_pool(gs, 'SCORE_TRAIN'); dv, dh = load_pool(gs, 'DEV_SCORE')
    assert not set(tr['parents']) & set(dv['parents'])
    out = RUN/name if name else folder(arm, gs, ss)
    out.mkdir(parents=True, exist_ok=resume)
    if (out/'last.pt').exists():
        raise FileExistsError('Completed run')
    torch.manual_seed(ss); np.random.seed(ss); random.seed(ss); rng = np.random.default_rng(ss)
    norm = {}
    for k in ('nodes', 'context'):
        axes = tuple(range(tr[k].ndim-1))
        norm[k+'_mean'] = tr[k].mean(axes)
        norm[k+'_std'] = np.maximum(tr[k].std(axes), .01)
    def tensor(d):
        return [torch.tensor((d[k]-norm[k+'_mean'])/norm[k+'_std'], device='cuda') for k in ('nodes', 'context')]
    tx, dx = tensor(tr), tensor(dv)
    tt, tf = [torch.tensor(tr[k], dtype=torch.float32, device='cuda') for k in ('task', 'feas')]
    dy = torch.tensor(dv['labels'], dtype=torch.float32, device='cuda')
    model = FactorRouteScorer(arm, cfg['width']).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step:1.)
    initial = tensor_state_digest(model.state_dict())
    settings = dict(arm=arm, generator_seed=gs, scorer_seed=ss, config=cfg, pool_hashes=dict(train=th, dev=dh),
                    source_sha256=source_identity(), initial_sha256=initial)
    audit = new_stream_audit(model, rng.bit_generator.state, torch.get_rng_state())
    start=0; best=float('inf'); history=[]; tic=time.monotonic()
    if resume:
        state=torch.load(out/'recovery.pt', map_location='cpu', weights_only=False)
        assert settings == state['settings']
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scheduler.load_state_dict(state['scheduler'])
        restore_rng(state['rng'], rng); audit=state['sampler']; start=state['step']; best=state['best']; history=state['history']
    else:
        write(out/'config.json', settings)
    def snapshot(step):
        return dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    rng=rng_state(rng), sampler=audit, step=step, best=best, history=history,
                    settings=settings, normalization=norm, source_commit=os.environ.get('CODE_COMMIT'))
    end = min(cfg['steps'], stop_after or cfg['steps'])
    for step in range(start+1, end+1):
        ids = rng.integers(len(tt), size=cfg['batch_size']); audit=append_indices(audit, ids)
        model.train(); z=model(tx[0][ids], tx[1][ids]); loss=training_loss(z, tt[ids], tf[ids], arm)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite factor loss')
        optimizer.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step(); scheduler.step()
        if step % cfg['eval_every'] == 0 or step == cfg['steps']:
            model.eval()
            with torch.inference_mode():
                value=float(joint_nll(model(*dx), dy, arm))
            history.append(dict(step=step, train_loss=float(loss), dev_joint_nll=value))
            if value < best:
                best=value; atomic_checkpoint(out/'best.pt', snapshot(step))
        if step % cfg['checkpoint_every'] == 0 or step == end:
            atomic_checkpoint(out/'recovery.pt', snapshot(step))
    if end < cfg['steps']:
        write(out/'interruption.json', dict(step=end, elapsed_seconds=time.monotonic()-tic))
        return
    atomic_checkpoint(out/'last.pt', snapshot(end))
    assert initial != tensor_state_digest(model.state_dict())
    write(out/'fit_summary.json', dict(elapsed_seconds=time.monotonic()-tic, updates=end,
          parameters=sum(p.numel() for p in model.parameters()), peak_allocated_bytes=torch.cuda.max_memory_allocated(),
          best_joint_nll=best, best_sha256=sha(out/'best.pt'), last_sha256=sha(out/'last.pt'),
          sampler=audit, task_prevalence=float(tr['task'].mean()), feasibility_prevalence=float(tr['feas'].mean()),
          joint_prevalence=float(tr['labels'].mean()), history=history))
    print('FIT', out.name, best, flush=True)


def metrics(q, data):
    from scripts.paired_modes_reliability import reliability_metrics
    from scripts.analyze_paired_selection import select
    z = np.log(np.clip(q, 1e-7, 1-1e-7)) - np.log1p(-np.clip(q, 1e-7, 1-1e-7))
    result = reliability_metrics(z, data)
    parents = data['parents']
    if str(parents[0]).startswith('paired_family_'):
        parents=np.array([p.rsplit('_', 1)[0] for p in parents])
    per=[]
    for i in range(len(q)):
        idx=select(data['paths'][i], q[i], 4); y=data['labels'][i,idx]
        modes={data['words'][i][j] for j in idx if data['labels'][i,j]}
        accepted=np.flatnonzero(q[i]>=.8)
        amodes={data['words'][i][j] for j in accepted if data['labels'][i,j]}
        top=int(np.argmax(q[i]))
        per.append(dict(brier=float(((q[i]-data['labels'][i])**2).mean()),
                        nll=float((np.logaddexp(0,z[i])-data['labels'][i]*z[i]).mean()),
                        top1=float(data['labels'][i,top]), k4_valid=float(y.mean()), k4_all=float(y.all()),
                        k4_modes=len(modes), k4_two=float(len(modes)>=2), accepted08=len(accepted),
                        accepted08_valid=int(data['labels'][i,accepted].sum()), accepted08_modes=len(amodes)))
    result['families']={p:{k:float(np.mean([v[k] for i,v in enumerate(per) if parents[i]==p])) for k in per[0]}
                        for p in sorted(set(parents))}
    result['set_metrics']={k:float(np.mean([v[k] for v in per])) for k in per[0]}
    return result


def binary_metrics(q, y, mask=None):
    if mask is not None:
        q,y=q[mask],y[mask]
    q=np.clip(q.astype(float),1e-7,1-1e-7); y=y.astype(float)
    return dict(count=int(y.size), brier=float(((q-y)**2).mean()),
                nll=float(-(y*np.log(q)+(1-y)*np.log1p(-q)).mean()), mean_q=float(q.mean()), prevalence=float(y.mean()))


def fit_joint_platt(logits, labels):
    from scipy.optimize import minimize
    z=np.asarray(logits,dtype=np.float64);y=np.asarray(labels,dtype=np.float64)
    def objective(ab):
        slope=np.exp(ab[0]);p=slope*z+ab[1]
        residual=1/(1+np.exp(-np.clip(p,-700,700)))-y
        return float((np.logaddexp(0,p)-y*p).mean()),np.array([(residual*slope*z).mean(),residual.mean()])
    fit=minimize(objective,[0.,0.],jac=True,method='L-BFGS-B',bounds=[(-3,3),(-5,5)])
    if not fit.success:raise RuntimeError(str(fit.message))
    return dict(slope=float(np.exp(fit.x[0])),intercept=float(fit.x[1]))


def evaluate(arm, gs, ss, version='evaluation'):
    torch=torch_setup()
    from routeset.factored_q import FactorRouteScorer, apply_calibration
    from scipy.optimize import minimize, minimize_scalar
    out=folder(arm,gs,ss); saved=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    norm=saved['normalization']; model=FactorRouteScorer(arm,saved['settings']['config']['width']).cuda().eval()
    model.load_state_dict(saved['model'])
    def predict(d):
        x=[torch.tensor((d[k]-norm[k+'_mean'])/norm[k+'_std'],device='cuda') for k in ('nodes','context')]
        with torch.inference_mode():
            return model(*x).cpu().numpy()
    cal,ch=load_pool(gs,'CALIBRATION'); cz=predict(cal).astype(np.float64)
    for role in ('SCORE_TRAIN','DEV_SCORE'):
        d,_=load_pool(gs,role); assert not set(d['parents']) & set(cal['parents'])
    def bce(z,y):return float((np.logaddexp(0,z)-y*z).mean())
    if arm in ('marginal','conditional'):
        temps=[]
        for j,key in enumerate(('task','feas')):
            mask=cal['task'] if j==1 and arm=='conditional' else np.ones_like(cal['task'])
            z,y=cz[...,j][mask],cal[key][mask]
            fit=minimize_scalar(lambda t:bce(z/np.exp(t),y),bounds=(-3,3),method='bounded')
            assert fit.success; temps.append(float(np.exp(fit.x)))
        calibration=dict(temperatures=temps)
        identity=dict(temperatures=[1.,1.])
    else:
        z=cz[...,0] if arm=='single' else cz.sum(-1)/math.sqrt(2.)
        calibration=fit_joint_platt(z,cal['labels'])
        identity=dict(slope=1.,intercept=0.)
    target=out/version; target.mkdir(exist_ok=False)
    results={}
    for role in ('paired_dev','old_dev','dev32'):
        d,dh=load_pool(gs,role); z=predict(d)
        with torch.inference_mode():
            raw,rf=apply_calibration(torch.tensor(z),arm,identity)
            q,f=apply_calibration(torch.tensor(z),arm,calibration)
        q=q.numpy(); raw=raw.numpy(); f=None if f is None else f.numpy()
        results[role]=dict(raw=metrics(raw,d),calibrated=metrics(q,d),input_hashes=dh)
        arrays=dict(logits=z,q=q,raw_q=raw,labels=d['labels'],task=d['task'],feas=d['feas'],ids=d['ids'],parents=d['parents'])
        if f is not None:
            arrays.update(q_task=f[...,0],q_feas=f[...,1])
            results[role]['factors']=dict(task=binary_metrics(f[...,0],d['task']),
                  feas_target_population=binary_metrics(f[...,1],d['feas'],d['task'] if arm=='conditional' else None))
            np.testing.assert_allclose(q,f[...,0]*f[...,1],rtol=0,atol=0)
        np.savez_compressed(target/(role+'.npz'),**arrays)
    write(out/(version+'.json'),dict(arm=arm,generator_seed=gs,scorer_seed=ss,calibration=calibration,
          calibration_pool_hashes=ch,scorer_sha256=sha(out/'best.pt'),results=results,
          calibration_arithmetic='float64 optimizer objective; finite-difference Platt optimization must not operate on float32 logits',
          claim='Upper-level task AND geometry only; product probability interpretation requires conditional arm and empirical calibration. No execution or independent final-test claim.'))
    print('EVAL',out.name,{r:{k:results[r]['calibrated'][k] for k in ('brier','selected_valid','nll')} for r in results},flush=True)


def package(gs=0,ss=0,version='evaluation',deployment='deployment'):
    torch=torch_setup()
    from routeset.factored_q import FactorRouteScorer,FactoredRoutePlanner,load_factored_planner
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    arm='conditional'; base=folder(arm,gs,ss); target=base/deployment; target.mkdir(exist_ok=False)
    saved=torch.load(base/'best.pt',map_location='cpu',weights_only=False); evaluation=read(base/(version+'.json'))
    genfile=OLD/('R1_seed%d'%gs)/'last.pt'; gen=torch.load(genfile,map_location='cpu',weights_only=False)
    generator=ProbabilisticGeometryRouteHead(**gen['config']['head_options']).cuda().eval();generator.load_state_dict(gen['model'])
    scorer=FactorRouteScorer(arm).cuda().eval();scorer.load_state_dict(saved['model'])
    planner=FactoredRoutePlanner(generator,scorer,saved['normalization'],evaluation['calibration']).cuda().eval()
    original=OLD/'reliability'/('R1_seed%d'%gs)/('deployment_seed%d'%gs)
    inp={k:v.cuda() for k,v in torch.load(original/'example_observed_inputs.pt',map_location='cpu',weights_only=False).items()}
    with torch.inference_mode():pred=planner(**inp,return_k=4)
    d,_=load_pool(gs,'paired_dev'); ident=read(original/'manifest.json')['example_id'];idx=list(d['ids']).index(ident)
    with np.load(base/version/'paired_dev.npz') as z:
        np.testing.assert_allclose(pred['q'][0].cpu().numpy(),z['q'][idx],rtol=1e-5,atol=1e-6)
    np.testing.assert_array_equal(pred['paths'][0].cpu().numpy(),d['paths'][idx])
    torch.save(dict(model=planner.state_dict(),head_options=gen['config']['head_options'],arm=arm,width=64,
                    normalization=saved['normalization'],calibration=evaluation['calibration'],
                    generator_sha256=sha(genfile),scorer_sha256=sha(base/'best.pt')),target/'planner.pt')
    loaded=load_factored_planner(target/'planner.pt','cuda')
    with torch.inference_mode():replayed=loaded(**inp,return_k=4)
    for k in pred:
        torch.testing.assert_close(pred[k],replayed[k],rtol=0,atol=0)
    np.savez_compressed(target/'example.npz',**{k:v[0].cpu().numpy() for k,v in pred.items()})
    torch.save({k:v.cpu() for k,v in inp.items()},target/'example_observed_inputs.pt')
    write(target/'manifest.json',dict(generator_sha256=sha(genfile),scorer_sha256=sha(base/'best.pt'),
          planner_sha256=sha(target/'planner.pt'),example_id=ident,reload_exact=True,original_paths_exact=True,
          product_exact=True,public_outputs=['paths','events','q_task','q_feas','q','selected_indices']))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['fit','evaluate','run','package']);p.add_argument('--arm',default='conditional')
    p.add_argument('--generator-seed',type=int,default=0);p.add_argument('--scorer-seed',type=int,default=0)
    p.add_argument('--resume',action='store_true');p.add_argument('--name');p.add_argument('--steps',type=int);p.add_argument('--stop-after',type=int)
    p.add_argument('--evaluation-version',default='evaluation')
    p.add_argument('--deployment-version',default='deployment')
    a=p.parse_args()
    if a.stage in ('fit','run'):fit(a.arm,a.generator_seed,a.scorer_seed,a.resume,a.name,a.steps,a.stop_after)
    if a.stage in ('evaluate','run'):evaluate(a.arm,a.generator_seed,a.scorer_seed,a.evaluation_version)
    if a.stage=='package':package(a.generator_seed,a.scorer_seed,a.evaluation_version,a.deployment_version)
