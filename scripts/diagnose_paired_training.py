"""Four fixed TRAIN batches: diagnose correspondence without optimizer updates."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write,sha,torch_setup
from scripts.train_paired_modes import load_data,paired_population,sample_batch
from scripts.paired_modes_data import RUN


def main(output,semantic=False):
    torch=torch_setup()
    from routeset.observed_probability import ProbabilisticGeometryRouteHead,balanced_assignment_loss
    from routeset.paired_modes import class_weights,relation_cost,partial_pair_loss,within_scene_loss,workspace_floor_loss
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from routeset.segment_clearance import segment_clearance_loss
    from routeset.train_v2 import positive_assignment_loss
    from scripts.train_observed_geometry import batch_inputs
    from scripts.observed_layout_variation import crossing_signature
    out=Path(output);out.mkdir(parents=True,exist_ok=False)
    data,geo,configs,cs,hs,_,_=load_data();groups,old=paired_population(data)
    ids_train=np.flatnonzero(np.isin(data['splits'],['TRAIN','FUTURE_GENERATOR_TRAIN'])&data['path_mask'].any(1))
    weights=np.zeros(data['path_mask'].shape,dtype=np.float32);modes=[[] for _ in configs]
    ww,mm=class_weights(data['paths'][ids_train],data['path_mask'][ids_train],[configs[i] for i in ids_train],crossing_signature)
    weights[ids_train]=ww
    for i,m in zip(ids_train,mm):modes[i]=m
    cs=torch.as_tensor(cs,dtype=torch.float32,device='cuda');hs=torch.as_tensor(hs,dtype=torch.float32,device='cuda')
    floors=torch.tensor([c['post_base_z']+.02 for c in configs],device='cuda')
    rng=np.random.default_rng(0);batches=[sample_batch(rng,groups,old) for _ in range(4)]
    coefficients=read(RUN/'probe_seed0/coefficients.json');results={};hashes={}
    for arm in (('R1','R2','R3') if semantic else ('R1','R2')):
        for seed in range(3):
            key='%s_seed%d'%(arm,seed);p=RUN/key/'last.pt';hashes[key]=sha(p)
            state=torch.load(p,map_location='cpu',weights_only=False)
            model=ProbabilisticGeometryRouteHead(**state['config']['head_options']).cuda()
            model.load_state_dict(state['model']);model.train();named=[(n,p) for n,p in model.named_parameters() if p.requires_grad];params=[p for _,p in named]
            rows=[]
            for batch,(ids,pairs) in enumerate(batches):
                torch.manual_seed(0);loss_rng=np.random.default_rng(9000)
                inp=batch_inputs(data,geo,ids,'cuda');xyz,events,details=model(**inp)
                tx=torch.as_tensor(data['paths'][ids],device='cuda');te=torch.as_tensor(data['events'][ids],device='cuda')
                pred=torch.cat([xyz[:,:,1:],events[:,:,1:,None]*.2],-1);target=torch.cat([tx[:,:,1:],te[:,:,1:,None]*.2],-1)
                balanced,_=balanced_assignment_loss(pred[:16],target[:16],torch.as_tensor(weights[ids[:16]],device='cuda'))
                oldloss=positive_assignment_loss(pred[16:],target[16:],data['path_mask'][ids[16:]],'saturation',loss_rng)
                ground=positive_endpoint_attention_loss(details['attention'],inp['world_xyz'],inp['valid_mask'],tx[:,:,-1],torch.as_tensor(data['path_mask'][ids],device='cuda'),.025)
                clear=segment_clearance_loss(xyz,cs[ids],hs[ids])+workspace_floor_loss(xyz,floors[ids])
                cfgs=[configs[i] for i in ids[:16]];types=[modes[i] for i in ids[:16]]
                rel=within_scene_loss(xyz[:16],cfgs,types);pair,matched=partial_pair_loss(xyz[:16],cfgs,types,pairs,require_present=arm=='R3')
                base=.5*(balanced+oldloss)+.02*ground+160*clear+coefficients['relation']*rel
                terms={'R1_objective':base,'clearance':160*clear,'relation':coefficients['relation']*rel,'partial_pair':coefficients['pair']*pair}
                if semantic:
                    endpoint=(xyz[:,:,-1,None]-tx[:,None,:,-1]).square().sum(-1)
                    endpoint=endpoint.masked_fill(~torch.as_tensor(data['path_mask'][ids,None],device='cuda'),float('inf')).amin(-1).mean()
                    terms.update(endpoint=endpoint,grounding=.02*ground)
                grads={k:torch.autograd.grad(v,params,retain_graph=True,allow_unused=True) for k,v in terms.items()}
                norms={k:float(torch.sqrt(sum(g.square().sum() for g in gs if g is not None))) for k,gs in grads.items()}
                cos={k:float(sum((g*h).sum() for g,h in zip(grads['partial_pair'],gs) if g is not None and h is not None))/(norms['partial_pair']*norms[k]+1e-30) for k,gs in grads.items() if k!='partial_pair'}
                both=0;total=0;violations=[]
                for a,b in pairs:
                    common=sorted(set(types[a])&set(types[b]))
                    if not common:continue
                    ca=relation_cost(xyz[a],cfgs[a],common).detach().amin(0)
                    cb=relation_cost(xyz[b],cfgs[b],common).detach().amin(0)
                    both+=int(((ca<=1e-12)&(cb<=1e-12)).sum());total+=len(common)
                    violations.extend(torch.maximum(ca,cb).cpu().tolist())
                rows.append(dict(batch=batch,gradient_norms=norms,pair_cosines=cos,shared_classes=total,
                    shared_classes_represented_both_sides=both,mean_max_relation_cost=float(np.mean(violations)),matched=matched))
                if semantic:
                    partitions={label:[i for i,(n,_) in enumerate(named) if (n.startswith('geometry.'))==is_geometry]
                                for label,is_geometry in [('geometry',True),('route_head',False)]}
                    info={}
                    for label,indices in partitions.items():
                        ns={k:float(torch.sqrt(sum((g[i].square().sum() for i in indices if g[i] is not None),xyz.new_zeros(())))) for k,g in grads.items()}
                        info[label]=dict(norms=ns,pair_cosines={k:float(sum((grads['partial_pair'][i]*g[i]).sum() for i in indices if grads['partial_pair'][i] is not None and g[i] is not None))/(ns['partial_pair']*ns[k]+1e-30) for k,g in grads.items() if k!='partial_pair'})
                    rows[-1]['parameter_partitions']=info
            results[key]=rows
            del model,state,grads,terms,xyz,events,details
    write(out/'RESULTS.json',dict(results=results,checkpoint_sha256=hashes,coefficients=coefficients,
        scope='Four fixed TRAIN batches, final checkpoints, no optimizer updates or DEV inputs. Gradient conflicts are local evidence, not causal proof of generalization failure.',
        batch_ids=[[str(data['scene_ids'][i]) for i in ids] for ids,_ in batches]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--semantic',action='store_true');a=p.parse_args();main(a.output,a.semantic)
