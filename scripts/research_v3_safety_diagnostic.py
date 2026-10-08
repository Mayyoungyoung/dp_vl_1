"""Four fixed TRAIN batches: mean versus whole-route collision objective."""
import argparse
import numpy as np
from scripts.run_observed_probability import torch_setup,read,write,sha
from scripts.research_v3_frequency import RUN,SUPPORT,POLICY,pad_targets,match_loss


def main(output):
    torch=torch_setup()
    from scripts.train_paired_modes import load_data
    from scripts.train_observed_geometry import batch_inputs
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.segment_clearance import path_segment_clearances
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    with np.load(SUPPORT/'support.npz') as z:support={k:z[k] for k in z.files}
    data,geo,configs,centers,halves,_,_=load_data()
    index={str(k):j for j,k in enumerate(data['scene_ids'])}
    train=np.flatnonzero(support['splits']=='TRAIN');mapped=np.array([index[str(support['ids'][j])] for j in train])
    assert all(data['splits'][j]=='TRAIN' for j in mapped)
    checkpoint=RUN/'frequency_set_matching/last.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ProbabilisticGeometryRouteHead(**saved['config']['head_options']).cuda();model.load_state_dict(saved['model']);model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    rng=np.random.default_rng(0);lrng=np.random.default_rng(6100);torch.manual_seed(0)
    centers=torch.tensor(centers,dtype=torch.float32,device='cuda');halves=torch.tensor(halves,dtype=torch.float32,device='cuda')
    rows=[];ratios=[];totalbad=0;onebad=0;total=0
    for batch in range(4):
        local=rng.integers(len(train),size=32);si=train[local];ids=mapped[local]
        inp=batch_inputs(data,geo,ids,'cuda');xyz,event,details=model(**inp)
        tags=[support['modes'][s,support['mask'][s]] for s in si]
        target,targete,valid=pad_targets([support['paths'][s,support['mask'][s]] for s in si],
            [support['events'][s,support['mask'][s]] for s in si])
        tx=torch.tensor(target,device='cuda');te=torch.tensor(targete,device='cuda')
        pred=torch.cat([xyz[:,:,1:],event[:,:,1:,None]*.2],-1);truth=torch.cat([tx[:,:,1:],te[:,:,1:,None]*.2],-1)
        regression=torch.stack([match_loss(pred[i:i+1],truth[i:i+1,:len(tags[i])],np.array([tags[i]]),lrng) for i in range(len(pred))]).mean()
        ground=.02*positive_endpoint_attention_loss(details['attention'],inp['world_xyz'],inp['valid_mask'],tx[:,:,-1],torch.tensor(valid,device='cuda'),.025)
        clearance=path_segment_clearances(xyz,centers[ids],halves[ids]);gap=(.02-clearance).clamp_min(0)
        mean=gap.square().mean();worst=gap.square().amax(-1).mean()
        terms=dict(regression=regression,grounding=ground,mean_collision=160*mean,worst_unscaled=worst)
        grads={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True) for k,v in terms.items()}
        norms={k:float(torch.sqrt(sum((g.square().sum() for g in gs if g is not None),xyz.new_zeros(())))) for k,gs in grads.items()}
        if norms['worst_unscaled']>1e-12:ratios.append(norms['mean_collision']/norms['worst_unscaled'])
        cos={}
        for a,b in [('mean_collision','regression'),('worst_unscaled','regression'),('mean_collision','grounding'),('mean_collision','worst_unscaled')]:
            cos[a+'_vs_'+b]=float(sum((g*h).sum() for g,h in zip(grads[a],grads[b]) if g is not None and h is not None))/(norms[a]*norms[b]+1e-30)
        counts=(gap>0).sum(-1).detach().cpu().numpy();bad=counts>0
        values=gap.amax(-1).detach().cpu().numpy();total+=counts.size;totalbad+=int(bad.sum());onebad+=int((counts==1).sum())
        rows.append(dict(batch=batch,ids=[str(data['scene_ids'][j]) for j in ids],collision_paths=int(bad.sum()),
            one_bad_segment=int((counts==1).sum()),violating_segment_counts=counts.tolist(),
            maximum_clearance_deficit_m=values.tolist(),mean_squared=float(mean),worst_squared=float(worst),gradient_norms=norms,gradient_cosines=cos))
    coefficient=float(np.median(ratios)) if ratios else None
    # Sparse violations make per-segment averaging a falsifiable explanation.
    sparse=sum(np.sum((np.array(r['violating_segment_counts'])>0)&(np.array(r['violating_segment_counts'])<=5)) for r in rows)
    gate=totalbad/total>=.05 and sparse/max(totalbad,1)>=.5 and coefficient is not None and coefficient>0
    write(out/'RESULTS.json',dict(rows=rows,total_path_states=total,colliding_paths=totalbad,
        at_most5_bad_segments=int(sparse),one_bad_segment=onebad,worst_collision_coefficient=coefficient,
        ordinary_safety_control_gate=bool(gate),checkpoint_sha256=sha(checkpoint),support_sha256=sha(SUPPORT/'support.npz'),
        scope='Four fixed TRAIN batches, no updates; coefficient matches initial gradient norm of historical160*mean collision loss; floor unchanged',
        explanation='Local gradient evidence is not a causal generalization result; max reduction is a conventional control, not novelty'))
    print(dict(total=total,colliding=totalbad,sparse=int(sparse),coefficient=coefficient,gate=bool(gate)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
