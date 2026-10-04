"""TRAIN-only bounded probe and strict historical-A pairing checks."""
import hashlib
import numpy as np
from scripts.run_observed_probability import ROOT,RUN,OLD,PARENT,read,write,sha,lines,verify_role


def training_boxes(data,train_ids,pcfg):
    import torch
    from scripts.evaluate_observed_two_row import validate_labels
    labels={r['id']:r for folder in (OLD,verify_role('FUTURE_GENERATOR_TRAIN'))
            for r in lines(folder/'supervision.jsonl') if r['split'] in ('TRAIN','FUTURE_GENERATOR_TRAIN')}
    centers=np.zeros((len(data['scene_ids']),4,3),np.float32);halves=np.zeros_like(centers);hashes={}
    for i in train_ids:
        label=labels[str(data['scene_ids'][i])]
        assert label['split']==str(data['splits'][i]) and label['parent_id']==str(data['parent_ids'][i])
        f=label['verification_only'];hashes[f]=sha(f)
        with np.load(f) as z:truth={k:z[k] for k in ('obstacle_centers','obstacle_halfsizes')}
        cfg=read(label['route_config']);assert sha(label['route_config'])==label['route_config_sha256']
        validate_labels(truth,label['semantic_targets'],cfg,label['route_types'])
        centers[i]=truth['obstacle_centers'];halves[i]=truth['obstacle_halfsizes']
    assert len(set(data['parent_ids'][train_ids]))==191
    return torch.tensor(centers,device='cuda'),torch.tensor(halves,device='cuda'),hashes


def audit_pair(seed,initial,geometry,train_ids,data,cfg):
    import torch
    folder=RUN/('M8_ordinary_seed%d_expanded'%seed)
    summary=read(folder/'summary.json');init=read(folder/'initialization.json')
    state=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
    assert summary['initial_sha256']==initial==state['initial_sha256']==init['model_sha256']
    assert sha(folder/'last.pt')==summary['last_checkpoint_sha256']
    assert state['parent_sha256']==sha(PARENT/'last.pt')
    assert state['dataset_fingerprint']==geometry['fingerprint']
    assert state['config']==cfg and state['step']==3000 and state['arm']=='ordinary'
    assert init['train_parent_ids']==sorted(set(map(str,data['parent_ids'][train_ids])))
    opt=state['optimizer']['param_groups'];assert len(opt)==1
    assert opt[0]['lr']==cfg['lr'] and opt[0]['weight_decay']==.0001
    return dict(historical_checkpoint=str(folder/'last.pt'),checkpoint_sha256=sha(folder/'last.pt'),
                historical_summary_sha256=sha(folder/'summary.json'),initial_sha256=initial,
                historical_draw_sha256=summary['draw_sha256'],dataset_fingerprint=geometry['fingerprint'],
                config=cfg,optimizer_group=opt[0],same_trainable_parameter_scope=True,reused_A=True)


def probe_loss(model,data,geometry,train_ids,centers,halves,rng,output):
    import torch
    from routeset.segment_clearance import segment_clearance_loss,path_segment_clearances
    from routeset.train_v2 import positive_assignment_loss
    from routeset.observed_geometry import positive_endpoint_attention_loss
    from scripts.train_observed_geometry import batch_inputs
    from routeset.geometry import segment_aabb_intersection
    params=[p for p in model.parameters() if p.requires_grad];rows=[];ratios=[]
    for n in range(4):
        ids=rng.choice(train_ids,size=32);inputs=batch_inputs(data,geometry,ids,'cuda')
        xyz,events,details=model(**inputs)
        target_xyz=torch.tensor(data['paths'][ids],device='cuda');target_events=torch.tensor(data['events'][ids],device='cuda')
        pred=torch.cat([xyz[:,:,1:],events[:,:,1:,None]*.2],-1)
        target=torch.cat([target_xyz[:,:,1:],target_events[:,:,1:,None]*.2],-1)
        base=positive_assignment_loss(pred,target,data['path_mask'][ids],'saturation',rng)
        base=base+.02*positive_endpoint_attention_loss(details['attention'],inputs['world_xyz'],inputs['valid_mask'],target_xyz[:,:,-1],torch.tensor(data['path_mask'][ids],device='cuda'),.025)
        loss=segment_clearance_loss(xyz,centers[ids],halves[ids])
        gradients=[]
        for value in (base,loss):
            g=torch.autograd.grad(value,params,retain_graph=True,allow_unused=True)
            gradients.append(float(torch.sqrt(sum(x.square().sum() for x in g if x is not None))))
        assert all(np.isfinite(gradients)) and gradients[1]>0
        ratios.append(gradients[0]/gradients[1])
        points=xyz.detach().cpu().numpy();c=centers[ids].cpu().numpy();h=halves[ids].cpu().numpy()
        shape=points[:,:,:-1,None].shape[:-2]+(4,3)
        checker=segment_aabb_intersection(np.broadcast_to(points[:,:,:-1,None],shape),np.broadcast_to(points[:,:,1:,None],shape),c[:,None,None]-h[:,None,None]-.02,c[:,None,None]+h[:,None,None]+.02).any(-1)
        exact=path_segment_clearances(xyz,centers[ids],halves[ids]).detach().cpu().numpy()<=.02
        vertex=np.max(np.abs(points[:,:,:,None]-c[:,None,None])-h[:,None,None],axis=-1).min(-1)<=.02
        missed=checker & ~vertex[:,:,:-1] & ~vertex[:,:,1:]
        assert np.array_equal(checker,exact)
        rows.append(dict(batch=n,ids=data['scene_ids'][ids].tolist(),original_loss=float(base),clearance_loss=float(loss),parameter_gradient_norms=gradients,gradient_ratio=ratios[-1],segments=int(checker.size),colliding_segments=int(checker.sum()),vertex_only_missed_segments=int(missed.sum()),analytic_missed_segments=int((checker&~exact).sum())))
    coefficient=float('%.2g'%np.median(ratios))
    write(output/'TRAIN_PROBE.json',dict(rows=rows,lambda_frozen_recommendation=coefficient,rule='median original/clearance parameter-gradient norms rounded2 significant digits',optimizer_updates=0,DEV_used=False,physical_geometry_scope='only four checker AABBs on TRAIN'))
    print('TRAIN_ONLY_LAMBDA',coefficient,flush=True)
