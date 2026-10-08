"""Eight isolated one-step TRAIN updates, no deployment model modification."""
import argparse
import copy
import numpy as np
from scripts.run_observed_probability import torch_setup,read,write,sha
from scripts.research_v3_frequency import RUN,SUPPORT,pad_targets,match_loss


def main(output):
    torch=torch_setup()
    from scripts.train_paired_modes import load_data
    from scripts.train_observed_geometry import batch_inputs
    from routeset.observed_probability import ProbabilisticGeometryRouteHead
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.segment_clearance import segment_clearance_loss
    from routeset.paired_modes import workspace_floor_loss
    from routeset.observed_training_audit import tensor_state_digest
    out=RUN/output;out.mkdir(exist_ok=False)
    assert sha(SUPPORT/'support.npz')==read(SUPPORT/'receipt.json')['support_sha256']
    with np.load(SUPPORT/'support.npz') as z:support={k:z[k] for k in z.files}
    data,geo,configs,centers,halves,_,_=load_data();index={str(k):j for j,k in enumerate(data['scene_ids'])}
    train=np.flatnonzero(support['splits']=='TRAIN');mapped=np.array([index[str(support['ids'][j])] for j in train])
    assert len(train)==1152 and all(data['splits'][j]=='TRAIN' for j in mapped)
    checkpoint=RUN/'safety_mean/last.pt';source_sha=sha(checkpoint);saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    base=ProbabilisticGeometryRouteHead(**saved['config']['head_options']).cuda();base.load_state_dict(saved['model']);base.train()
    initial=tensor_state_digest(base.state_dict());rng=np.random.default_rng(0);lrng=np.random.default_rng(6100);records=[]
    for batch in range(4):
        local=rng.integers(len(train),size=32);si=train[local];ids=mapped[local]
        inp=batch_inputs(data,geo,ids,'cuda')
        tags=[support['modes'][s,support['mask'][s]] for s in si]
        paths,events,valid=pad_targets([support['paths'][s,support['mask'][s]] for s in si],[support['events'][s,support['mask'][s]] for s in si])
        tx=torch.tensor(paths,device='cuda');te=torch.tensor(events,device='cuda');mask=torch.tensor(valid,device='cuda')
        cs=torch.tensor(centers[ids],dtype=torch.float32,device='cuda');hs=torch.tensor(halves[ids],dtype=torch.float32,device='cuda')
        floors=torch.tensor([configs[j]['post_base_z']+.02 for j in ids],dtype=torch.float32,device='cuda')
        assignment_state=copy.deepcopy(lrng.bit_generator.state);vectors={};updates={};values={};reference=None;loss_rng_after=None
        def evaluate(model):
            loss_rng=np.random.default_rng();loss_rng.bit_generator.state=copy.deepcopy(assignment_state)
            xyz,event,details=model(**inp);pred=torch.cat([xyz[:,:,1:],event[:,:,1:,None]*.2],-1);truth=torch.cat([tx[:,:,1:],te[:,:,1:,None]*.2],-1)
            regression=torch.stack([match_loss(pred[i:i+1],truth[i:i+1,:len(tags[i])],np.array([tags[i]]),loss_rng) for i in range(len(pred))]).mean()
            ground=.02*positive_endpoint_attention_loss(details['attention'],inp['world_xyz'],inp['valid_mask'],tx[:,:,-1],mask,.025)
            clear=160*(segment_clearance_loss(xyz,cs,hs)+workspace_floor_loss(xyz,floors))
            return dict(regression=regression,grounding=ground,clearance=clear),xyz,event,details,copy.deepcopy(loss_rng.bit_generator.state)
        for mode in ('straight_through_peak','hard_peak'):
            torch.manual_seed(0);model=copy.deepcopy(base);model.geometry.anchor_mode=mode
            assert tensor_state_digest(model.state_dict())==initial
            optimizer=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.0001)
            components,xyz,event,details,loss_rng_after=evaluate(model)
            before_paths=xyz.detach().cpu().numpy();before_events=event.detach().cpu().numpy();before_anchor=details['attention'].argmax(-1).detach().cpu().numpy()
            if reference is None:reference=(before_paths,before_events,before_anchor)
            else:
                for a,b in zip(reference,(before_paths,before_events,before_anchor)):np.testing.assert_array_equal(a,b)
            total=sum(components.values());before_loss={k:float(v) for k,v in components.items()};total.backward()
            vectors[mode]=torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().reshape(-1) for p in model.parameters()])
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            updates[mode]=torch.cat([(p-b).detach().reshape(-1) for p,b in zip(model.parameters(),base.parameters())])
            with torch.no_grad():after,px,pe,da,_=evaluate(model)
            changed=int(np.sum(da['attention'].argmax(-1).cpu().numpy()!=before_anchor))
            values[mode]=dict(before=before_loss,after={k:float(v) for k,v in after.items()},changed_anchor_requests=changed,
                gradient_norm=float(torch.linalg.vector_norm(vectors[mode])),update_norm=float(torch.linalg.vector_norm(updates[mode])),
                maximum_path_change_m=float((px.cpu()-torch.tensor(before_paths)).abs().max()))
            name=f'batch{batch}_{mode}'
            np.savez_compressed(out/(name+'.npz'),before_paths=before_paths,before_events=before_events,after_paths=px.cpu().numpy(),after_events=pe.cpu().numpy(),ids=[str(data['scene_ids'][j]) for j in ids])
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=1,parent_sha256=source_sha,anchor_mode=mode,
                cpu_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),sampler=rng.bit_generator.state,assignment_rng_before=assignment_state,assignment_rng_after=loss_rng_after),out/(name+'.pt'))
            del model,optimizer,components,xyz,event,details,total,after,px,pe,da
        lrng.bit_generator.state=loss_rng_after
        result=dict(batch=batch,ids=[str(data['scene_ids'][j]) for j in ids],modes=values,initial_forward_exact=True)
        for key,vecs in [('gradient',vectors),('update',updates)]:
            a,b=(vecs[k] for k in ('straight_through_peak','hard_peak'))
            result[key+'_difference_fraction']=float(torch.linalg.vector_norm(a-b)/(torch.linalg.vector_norm(a)+1e-30))
            result[key+'_cosine']=float(torch.dot(a,b)/(torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)+1e-30))
        records.append(result);print({k:v for k,v in result.items() if k!='ids'},flush=True)
    assert tensor_state_digest(base.state_dict())==initial and sha(checkpoint)==source_sha
    write(out/'RESULTS.json',dict(rows=records,parent_sha256=source_sha,parent_unchanged=True,data_fingerprint=geo['fingerprint'],
        support_sha256=sha(SUPPORT/'support.npz'),artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.pt','.npz')},
        scope='Eight isolated scratch step1 updates, each reset to the same original parent. All four batches retained, same reference RNG, TRAIN only. No generalization or novelty claim.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
