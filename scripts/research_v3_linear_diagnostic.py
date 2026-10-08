"""TRAIN-only gradient normalization for a conventional linear margin loss."""
import argparse
import numpy as np
from scripts.run_observed_probability import torch_setup,read,write,sha
from scripts.research_v3_frequency import RUN,SUPPORT,pad_targets,match_loss


def main(output):
    torch=torch_setup()
    from scripts.train_paired_modes import load_data
    from scripts.train_observed_geometry import batch_inputs
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.segment_clearance import path_segment_clearances
    out=RUN/output;out.mkdir(exist_ok=False)
    assert sha(SUPPORT/'support.npz')==read(SUPPORT/'receipt.json')['support_sha256']
    with np.load(SUPPORT/'support.npz') as z:support={k:z[k] for k in z.files}
    data,geo,configs,centers,halves,_,_=load_data()
    index={str(k):j for j,k in enumerate(data['scene_ids'])}
    train=np.flatnonzero(support['splits']=='TRAIN');mapped=np.array([index[str(support['ids'][j])] for j in train])
    assert len(train)==1152 and all(data['splits'][j]=='TRAIN' for j in mapped)
    checkpoint=RUN/'safety_mean/last.pt';saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ProbabilisticGeometryRouteHead(**saved['config']['head_options']).cuda();model.load_state_dict(saved['model']);model.train()
    params=[p for p in model.parameters() if p.requires_grad]
    rng=np.random.default_rng(0);lrng=np.random.default_rng(6100);torch.manual_seed(0)
    centers=torch.tensor(centers,dtype=torch.float32,device='cuda');halves=torch.tensor(halves,dtype=torch.float32,device='cuda')
    rows=[];ratios=[];cosines=[];totalbad=0
    for batch in range(4):
        local=rng.integers(len(train),size=32);si=train[local];ids=mapped[local]
        inp=batch_inputs(data,geo,ids,'cuda');xyz,event,details=model(**inp)
        tags=[support['modes'][s,support['mask'][s]] for s in si]
        target,targete,valid=pad_targets([support['paths'][s,support['mask'][s]] for s in si],[support['events'][s,support['mask'][s]] for s in si])
        tx=torch.tensor(target,device='cuda');te=torch.tensor(targete,device='cuda')
        pred=torch.cat([xyz[:,:,1:],event[:,:,1:,None]*.2],-1);truth=torch.cat([tx[:,:,1:],te[:,:,1:,None]*.2],-1)
        regression=torch.stack([match_loss(pred[i:i+1],truth[i:i+1,:len(tags[i])],np.array([tags[i]]),lrng) for i in range(len(pred))]).mean()
        ground=.02*positive_endpoint_attention_loss(details['attention'],inp['world_xyz'],inp['valid_mask'],tx[:,:,-1],torch.tensor(valid,device='cuda'),.025)
        gap=(.02-path_segment_clearances(xyz,centers[ids],halves[ids])).clamp_min(0)
        terms=dict(regression=regression,grounding=ground,quadratic=160*gap.square().mean(),linear=gap.mean())
        grads={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True) for k,v in terms.items()}
        norms={k:float(torch.sqrt(sum((g.square().sum() for g in gs if g is not None),xyz.new_zeros(())))) for k,gs in grads.items()}
        cos={}
        for a,b in [('quadratic','linear'),('quadratic','regression'),('linear','regression'),('linear','grounding')]:
            cos[a+'_vs_'+b]=float(sum((g*h).sum() for g,h in zip(grads[a],grads[b]) if g is not None and h is not None))/(norms[a]*norms[b]+1e-30)
        if norms['linear']>1e-12 and norms['quadratic']>1e-12:
            ratios.append(norms['quadratic']/norms['linear']);cosines.append(cos['quadratic_vs_linear'])
        count=int((gap.amax(-1)>0).sum());totalbad+=count
        rows.append(dict(batch=batch,ids=[str(data['scene_ids'][j]) for j in ids],colliding_paths=count,
            maximum_deficit_m=gap.amax(-1).detach().cpu().tolist(),gradient_norms=norms,gradient_cosines=cos))
    coefficient=float(np.median(ratios)) if ratios else None
    gate=bool(totalbad>=10 and len(ratios)>=3 and coefficient is not None and np.isfinite(coefficient) and coefficient>0 and np.median(cosines)>=.5)
    result=dict(rows=rows,colliding_paths=totalbad,path_states=1024,linear_coefficient=coefficient,
        defined_ratios=len(ratios),median_quadratic_linear_cosine=float(np.median(cosines)) if cosines else None,
        ordinary_linear_gate=gate,checkpoint_sha256=sha(checkpoint),support_sha256=sha(SUPPORT/'support.npz'),
        data_fingerprint=geo['fingerprint'],scope='Four fixed TRAIN batches, no updates, finite gradient scale gate only; standard hinge loss, no novelty.')
    write(out/'RESULTS.json',result);print({k:v for k,v in result.items() if k!='rows'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
