"""Read-only exact-forward versus surrogate-backward TRAIN diagnostic."""
import argparse
import copy
import numpy as np
from scripts.run_observed_probability import torch_setup,read,write,sha
from scripts.research_v3_frequency import RUN,SUPPORT


def main(output):
    torch=torch_setup()
    from scripts.train_paired_modes import load_data
    from scripts.train_observed_geometry import batch_inputs
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    from routeset.segment_clearance import path_segment_clearances
    from routeset.observed_training_audit import tensor_state_digest
    out=RUN/output;out.mkdir(exist_ok=False)
    with np.load(SUPPORT/'support.npz') as z:support={k:z[k] for k in ('ids','splits')}
    data,geo,_,centers,halves,_,_=load_data();index={str(k):j for j,k in enumerate(data['scene_ids'])}
    train=np.flatnonzero(support['splits']=='TRAIN');mapped=np.array([index[str(support['ids'][j])] for j in train])
    assert len(train)==1152 and all(data['splits'][j]=='TRAIN' for j in mapped)
    checkpoint=RUN/'safety_mean/last.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    fp32=ProbabilisticGeometryRouteHead(**saved['config']['head_options']).cuda();fp32.load_state_dict(saved['model']);fp32.eval()
    assert fp32.geometry.anchor_mode=='straight_through_peak'
    model=copy.deepcopy(fp32).double();before=tensor_state_digest(model.state_dict())
    named=list(model.named_parameters());params=[p for _,p in named]
    scale_index=next(i for i,(n,_) in enumerate(named) if n=='geometry.log_attention_scale')
    scale=model.geometry.log_attention_scale;original=scale.detach().clone();base_scale=float(original)
    hs=(1e-3,1e-4);clamped=bool(np.exp(base_scale+max(hs))>=100)
    rng=np.random.default_rng(0);torch.manual_seed(0)
    torch.save(dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),sampler=rng.bit_generator.state),out/'rng_before.pt')
    rows=[]
    if not clamped:
        for batch in range(4):
            ids=mapped[rng.integers(len(train),size=32)];inp32=batch_inputs(data,geo,ids,'cuda')
            inp={k:v.double() if v.is_floating_point() else v for k,v in inp32.items()}
            cs=torch.tensor(centers[ids],dtype=torch.double,device='cuda');hh=torch.tensor(halves[ids],dtype=torch.double,device='cuda')
            with torch.no_grad():p32,e32,d32=fp32(**inp32)
            def objective():
                xyz,event,details=model(**inp)
                gap=(.02-path_segment_clearances(xyz,cs,hh)).clamp_min(0)
                return 160*gap.square().mean(),xyz,event,details
            analytic={};vectors={};reference=None
            for mode in ('straight_through_peak','hard_peak'):
                model.geometry.anchor_mode=mode;loss,xyz,event,details=objective()
                current=(xyz.detach().clone(),event.detach().clone(),details['anchor_xyz'].detach().clone(),details['attention'].argmax(-1).detach().clone())
                if reference is None:reference=current
                else:
                    for a,b in zip(reference,current):assert torch.equal(a,b)
                grads=torch.autograd.grad(loss,params,allow_unused=True)
                vec=torch.cat([(torch.zeros_like(p) if g is None else g).detach().reshape(-1) for p,g in zip(params,grads)])
                vectors[mode]=vec
                analytic[mode]=dict(loss=float(loss),scale_derivative=float(grads[scale_index]),gradient_norm=float(torch.linalg.vector_norm(vec)))
                del loss,xyz,event,details,grads
            finite=[]
            try:
                for h in hs:
                    values=[]
                    for sign in (-1,1):
                        with torch.no_grad():
                            scale.copy_(original+sign*h);loss,_,_,details=objective()
                            assert torch.equal(details['attention'].argmax(-1),reference[3])
                            assert torch.equal(details['anchor_xyz'],reference[2])
                            values.append(float(loss))
                    finite.append(dict(h=h,loss_minus=values[0],loss_plus=values[1],derivative=(values[1]-values[0])/(2*h)))
            finally:
                with torch.no_grad():scale.copy_(original)
                model.geometry.anchor_mode='straight_through_peak'
            a,b=(vectors[k] for k in ('straight_through_peak','hard_peak'))
            fd=[r['derivative'] for r in finite];tol=1e-9+.01*max(map(abs,fd))
            stable=abs(fd[0]-fd[1])<=tol
            hard_ok=all(abs(d-analytic['hard_peak']['scale_derivative'])<=tol for d in fd)
            ste_bad=all(abs(d-analytic['straight_through_peak']['scale_derivative'])>1e-9+.1*abs(d) for d in fd)
            rows.append(dict(batch=batch,ids=[str(data['scene_ids'][j]) for j in ids],analytic=analytic,finite=finite,
                anchors=reference[2].cpu().tolist(),anchor_indices=reference[3].cpu().tolist(),forward_modes_exact=True,
                fp32_vs_fp64_max_path_difference_m=float((p32.double()-reference[0]).abs().max()),
                fp32_vs_fp64_max_event_difference=float((e32.double()-reference[1]).abs().max()),
                fp32_vs_fp64_anchor_id_changes=int((d32['attention'].argmax(-1)!=reference[3]).sum()),
                gradient_difference_fraction=float(torch.linalg.vector_norm(a-b)/(torch.linalg.vector_norm(a)+1e-30)),
                gradient_cosine=float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)+1e-30)),
                finite_difference_stable=stable,hard_derivative_agrees=hard_ok,ste_derivative_differs=ste_bad))
            print(dict(batch=batch,analytic=analytic,finite=finite,stable=stable,hard_ok=hard_ok,ste_differs=ste_bad),flush=True)
    assert tensor_state_digest(model.state_dict())==before
    result=dict(rows=rows,scale_parameter=base_scale,scale_value=float(np.exp(base_scale)),clamp_blocks_probe=clamped,
        checkpoint_sha256=sha(checkpoint),data_fingerprint=geo['fingerprint'],support_sha256=sha(SUPPORT/'support.npz'),
        parameters_unchanged=True,comparison_precision='float64; original fp32 deviations explicitly recorded',
        numerical_tolerance='finite difference agreement:1e-9+1% relative; STE mismatch:1e-9+10% relative; descriptive, no training gate',
        scope='Four fixed TRAIN batches, no updates; intentional surrogate derivatives may differ without being harmful. No DEV or scoring loss used.')
    write(out/'RESULTS.json',result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
