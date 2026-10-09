"""TRAIN-only forced-mode/scene-input interventions and actual resume audit."""
import argparse
import json
import numpy as np
from scripts.mode_geometry_experiment import RUN,PREP,DATA,read,write,sha,lines,torch_setup
from scripts.research_v3_audit import plain


def diagnose(names,output):
    torch=torch_setup()
    from routeset.mode_geometry import ModeGeometryHead,VOCAB
    from scripts.train_verified_set import checker
    from scripts.evaluate_paired_modes import references
    with np.load(PREP/'train.npz') as z:d={k:z[k] for k in z.files}
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='TRAIN'}
    # First128 recorded TRAIN pairs with a changed-coordinate witness, no model-based selection.
    pairs=np.flatnonzero(d['pair_changed'].any(-1))[:128]
    report={}
    for name in names:
        ck=torch.load(RUN/name/'last.pt',map_location='cpu',weights_only=False)
        model=ModeGeometryHead(**ck['config']['head_options'],conditional=ck['config']['conditional'],decoder_communication=ck['config'].get('decoder_communication',True)).cuda()
        model.load_state_dict(ck['model']);model.eval();model.requires_grad_(False)
        counts=dict(opportunities=0,valid_intended=0,source_context_valid_intended=0,mode_selected=0,
                    companion_changed_valid_intended=0)
        errors=[];static=[];sensitivity=[];pred_delta=[];target_delta=[];companion_change=[]
        for pair in pairs:
            a,b=d['pairs'][pair];m=np.flatnonzero(d['pair_changed'][pair])
            # Replicate first mode to fill eight; report only one canonical occurrence per queried mode.
            modes=np.resize(m,8)[None];mi=torch.as_tensor(modes,device='cuda')
            def run(idx,ctxidx):
                vals=[torch.as_tensor(d[k][[ctxidx if k=='context' else idx]],device='cuda') for k in ('context','anchor','current')]
                return model.decode(*vals,mi)
            with torch.inference_mode():
                pa,_,_=run(a,a);pb,eb,db=run(b,b);swap,es,_=run(b,a)
                original_mi=mi
                # Keep all reported mode queries, alter only unreported companions.
                others=[v for v in range(16) if v not in m][:8-len(m)]
                mi=torch.as_tensor(np.r_[m,others][None],device='cuda')
                alt,alte,_=run(b,b);mi=original_mi
            x=pb[0].cpu().numpy();e=eb[0].cpu().numpy();sw=swap[0].cpu().numpy();ev=es[0].cpu().numpy()
            ref=references(labels[str(d['ids'][b])]);check=checker(ref);v,w=check(x,e);sv,ww=check(sw,ev)
            av,aw=check(alt[0].cpu().numpy(),alte[0].cpu().numpy())
            k=len(m);counts['opportunities']+=k
            counts['valid_intended']+=sum(v[i] and w[i]==VOCAB[m[i]] for i in range(k))
            counts['source_context_valid_intended']+=sum(sv[i] and ww[i]==VOCAB[m[i]] for i in range(k))
            counts['companion_changed_valid_intended']+=sum(av[i] and aw[i]==VOCAB[m[i]] for i in range(k))
            companion_change.extend(np.linalg.norm(x[:k]-alt[0,:k].cpu().numpy(),axis=-1).mean(-1))
            top=db['mode_logits'][0].topk(8).indices.cpu().numpy();counts['mode_selected']+=len(set(top)&set(m))
            delta=(pb-pa)[0,:k].cpu().numpy();truth=d['pair_target'][pair,m]-d['pair_source'][pair,m]
            errors.extend(np.square(delta-truth).mean((1,2)));static.extend(np.square(truth).mean((1,2)))
            sensitivity.extend(np.linalg.norm(x[:k]-sw[:k],axis=-1).mean(-1))
            pred_delta.extend(np.linalg.norm(delta,axis=-1).mean(-1));target_delta.extend(np.linalg.norm(truth,axis=-1).mean(-1))
        report[name]=dict(counts,displacement_mse=float(np.mean(errors)),zero_displacement_mse=float(np.mean(static)),
            context_change_mean_m=float(np.mean(sensitivity)),predicted_change_mean_m=float(np.mean(pred_delta)),target_change_mean_m=float(np.mean(target_delta)))
        report[name]['companion_change_mean_m']=float(np.mean(companion_change))
    report=plain(report)
    write(RUN/output,dict(models=report,scope='TRAIN forced-mode diagnostic only, source-context swap is an intervention not deployed inference',pairs=pairs.tolist(),prepared_sha256=sha(PREP/'train.npz')))
    print(json.dumps(report),flush=True)


def resume_audit(a,b,output):
    torch=torch_setup()
    x=torch.load(RUN/a/'last.pt',map_location='cpu',weights_only=False)
    y=torch.load(RUN/b/'last.pt',map_location='cpu',weights_only=False)
    def equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,np.ndarray):return np.array_equal(a,b)
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(u,v) for u,v in zip(a,b))
        return a==b
    result={k:equal(x[k],y[k]) for k in ('model','optimizer','rng','stream','step','history','settings')}
    assert all(result.values()),result
    write(RUN/output,dict(exact_equal=result,a_sha256=sha(RUN/a/'last.pt'),b_sha256=sha(RUN/b/'last.pt')))
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['diagnose','resume']);p.add_argument('--names',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if a.stage=='diagnose':diagnose(a.names,a.output)
    else:resume_audit(*a.names,a.output)
