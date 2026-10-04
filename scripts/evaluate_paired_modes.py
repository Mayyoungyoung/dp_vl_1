"""Seal actual observation-only predictions, then evaluate high-level validity.

No geometric repair, oracle selection, or reference filtering of predictions.
Paired statistics cluster all targets and variants by original layout family.
"""
import argparse
import hashlib
import json
import random
import time
import numpy as np
from scripts.run_observed_probability import ROOT,SOURCE,OLD,read,write,sha,lines,torch_setup,verify_role
from scripts.paired_modes_data import DATA,RUN
from routeset.geometric_modes import portal_word,encode_word

FROZEN_Q=ROOT/'runs/observed_probability_v1/M8_ordinary_seed0_expanded_scores/deployment_seed0/planner.pt'
FROZEN_Q_SHA='574e89ceb1358af474036575a7183edb67c1ee2427a90bb7254135fb6493d96a'


def dataset(split):
    if split=='paired_dev':return DATA/'export','DEV_MODEL'
    if split=='old_dev':return OLD,'DEV_MODEL'
    if split=='dev32':return ROOT/'data/geometric_modes_v1/DEV_MODEL','DEV_MODEL'
    if split in ('SCORE_TRAIN','DEV_SCORE','CALIBRATION'):return verify_role(split),split
    raise ValueError('Unregistered split')


def check_candidates(paths,events,label,current,truth,config):
    if 'post_heights' not in config:
        from scripts.evaluate_observed_two_row import scene_metrics
        metrics,candidates=scene_metrics(paths,events,current,truth,label['semantic_targets'],label['route_types'],config)
    else:
        from scripts.evaluate_observed_obstacles import scene_metrics
        from scripts.observed_layout_variation import geometry,crossing_signature
        cs,hs=geometry(config)
        for a,b in ((cs,truth['obstacle_centers']),(hs,truth['obstacle_halfsizes'])):
            np.testing.assert_allclose(a,b,rtol=0,atol=1e-6)
        np.testing.assert_allclose(config['goal_xyz'],label['semantic_targets']['centers'],rtol=0,atol=1e-6)
        assert config['tip_clearance_m']==.02 and label['semantic_targets']['tolerance']==.03
        metrics,candidates=scene_metrics(paths,events,current,truth,label['semantic_targets'],label['route_types'],
                             clearance=.02,type_classifier=lambda path:crossing_signature(path,config))
    # Prospective common task-workspace constraint. Historical artifacts stay intact.
    floor=config['post_base_z']+.02
    metrics['original_post_only_metrics']=dict(metrics)
    for path,c in zip(paths,candidates):
        c['post_only_tip_valid']=c['TipValid'];c['post_segments_clear']=c['tip_segments_clear']
        c['minimum_z_m']=float(np.min(path[:,2])) if np.isfinite(path).all() else None
        c['workspace_floor_correct']=bool(c['minimum_z_m'] is not None and c['minimum_z_m']>=floor)
        c['tip_segments_clear']=c['tip_segments_clear'] and c['workspace_floor_correct']
        c['TipValid']=c['TipValid'] and c['workspace_floor_correct']
        c['classified_tip_valid']=c['TipValid'] and c['declared_passage_type'] is not None
    valid=sum(c['TipValid'] for c in candidates);known=sum(c['classified_tip_valid'] for c in candidates)
    types={tuple(c['declared_passage_type']) for c in candidates if c['classified_tip_valid']}
    references={tuple(m) for m in label['route_types'] if m is not None}
    metrics.update(TipValidAtK=valid/len(candidates),AnyTipValidAtK=float(valid>0),UniqueClassifiedTipValidAtK=len(types),
        UnknownTypeTipValidCount=valid-known,DuplicateClassifiedTipValidCount=known-len(types),
        KnownReferenceTypeCoverageAtK=len(types&references)/len(references) if references else None,
        TipClearAtK=float(np.mean([c['tip_segments_clear'] for c in candidates])),workspace_minimum_z_m=floor,
        validity_scope='Goal3cm, start5mm, reach events, continuous post2cm clearance AND registered minimum workspace height; no full-arm execution certificate')
    return metrics,candidates


def references(label):
    from routeset.observed_route_head import resample_event_segments
    config=read(label['route_config'])
    with np.load(label['observation']) as a:current={k:a[k] for k in ('gripper_pose','gripper_open')}
    with np.load(label['verification_only']) as a:truth={k:a[k] for k in ('obstacle_centers','obstacle_halfsizes')}
    paths=[];events=[]
    for file in label['routes']:
        with np.load(file) as a:p,e=resample_event_segments(a['gripper_pose'],a['gripper_open'],24)
        paths.append(p);events.append(e)
    candidates=check_candidates(np.array(paths),np.array(events),label,current,truth,config)[1] if paths else []
    words=[encode_word(portal_word(p,config)) if c['TipValid'] else None for p,c in zip(paths,candidates)]
    return dict(config=config,paths=np.array(paths),words=words,current=current,truth=truth,label=label,
                reference_valid=[c['TipValid'] for c in candidates],reference_candidates=candidates)


def inputs_for(row,label,cache,torch,hashes):
    from scripts.train_observed_geometry import read_geometry
    key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
    file=cache/(key+'.npz');hashes[str(file)]=sha(file)
    with np.load(file) as a:
        assert all(str(a[k].item())==row[k] for k in ('id','parent_id','split'))
        assert str(a['image_sha256'].item())==sha(row['image'])
        feat=np.r_[a['mean_hidden'].reshape(-1),a['last_hidden'].reshape(-1)].astype('float32')
    geo=read_geometry(row['image'],label['observation'],2)
    with np.load(label['observation']) as a:current=np.r_[a['gripper_pose'],np.asarray(a['gripper_open']).reshape(1)].astype('float32')
    for f in (row['image'],label['observation']):hashes[str(f)]=sha(f)
    return dict(features=torch.tensor(feat[None],device='cuda'),current=torch.tensor(current[None],device='cuda'),
                **{k:torch.tensor(v[None],device='cuda') for k,v in geo.items()})


def infer(arm,seed,split,checkpoint=None):
    torch=torch_setup()
    from routeset.observed_probability import load_scored_planner,route_observation_features
    folder,role=dataset(split);cache=folder/'qwen_cache';cc=read(cache/'cache_config.json')
    assert cc['manifest_sha256']==sha(folder/'observations.jsonl') and cc['model_trainable_parameter_count']==0
    oldcc=read(OLD/'qwen_cache/cache_config.json')
    for k in ('revision','processor','max_pixels','dtype','input_contract'):assert cc[k]==oldcc[k]
    rows=[r for r in lines(folder/'observations.jsonl') if r['split']==role]
    labels={r['id']:r for r in lines(folder/'supervision.jsonl') if r['split']==role}
    assert sha(FROZEN_Q)==FROZEN_Q_SHA
    planner=load_scored_planner(FROZEN_Q,'cuda')
    checkpoint=checkpoint or (ROOT/'runs/segment_clearance_v1'/('B_seed%d'%seed)/'last.pt' if arm=='B' else RUN/('%s_seed%d'%(arm,seed))/'last.pt')
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert state['step']==3000
    planner.generator.load_state_dict(state['model'],strict=True);planner.eval();planner.requires_grad_(False)
    out=RUN/'evaluation'/split/('%s_seed%d'%(arm,seed));out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0);random.seed(0);np.random.seed(0)
    torch.save(dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()),out/'rng_before.pt')
    values={k:[] for k in ('paths','events','q','nodes','context','ids','parents','labels')};hashes={};tic=time.monotonic()
    for row in rows:
        inp=inputs_for(row,labels[row['id']],cache,torch,hashes)
        with torch.inference_mode():
            xyz,event,_=planner.generator(**inp)
            geo=planner.generator.geometry(**inp,return_point_features=True)
            context=planner.generator.head.feature_encoder(inp['features'])+planner.generator.head.state_encoder(inp['current'])+geo['context']
            nodes,ctx=route_observation_features(xyz,event,inp['current'],inp['world_xyz'],inp['rgb'],inp['valid_mask'],geo['point_features'],context,geo['anchor_xyz'])
            logits=planner.scorer((nodes-planner.nodes_mean)/planner.nodes_std,(ctx-planner.context_mean)/planner.context_std)
            q=(logits/planner.temperature).sigmoid()
        pred={k:v[0].cpu().numpy() for k,v in dict(paths=xyz,events=event,q=q,nodes=nodes,context=ctx).items()}
        np.savez_compressed(out/(row['id']+'.npz'),paths=pred['paths'],events=pred['events'],q=pred['q'])
        for k,v in pred.items():values[k].append(v)
        values['ids'].append(row['id']);values['parents'].append(row['parent_id'])
    # Every actual candidate has now been saved; oracle checking starts here.
    refs={};scenes=[]
    for i,row in enumerate(rows):
        ref=references(labels[row['id']]);refs[row['id']]=ref
        metrics,candidates=check_candidates(values['paths'][i],values['events'][i],ref['label'],ref['current'],ref['truth'],ref['config'])
        values['labels'].append([c['TipValid'] for c in candidates])
        scenes.append(dict(id=row['id'],metrics=metrics,candidates=candidates,prediction_sha256=sha(out/(row['id']+'.npz'))))
    arrays={k:np.asarray(v) for k,v in values.items()};np.savez_compressed(out/'pool.npz',**arrays)
    write(out/'per_scene.json',scenes)
    write(out/'receipt.json',dict(generator_checkpoint_sha256=sha(checkpoint),frozen_q_sha256=FROZEN_Q_SHA,
        arm=arm,seed=seed,split=split,manifest_sha256=sha(folder/'observations.jsonl'),input_sha256=hashes,
        pool_sha256=sha(out/'pool.npz'),requests=len(rows),complete_path_states=len(rows)*8,new_qwen_requests=0,
        elapsed_seconds=time.monotonic()-tic,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        probability_scope='Frozen historical q transfer only; not recalibrated to the new generator'))
    if role=='DEV_MODEL':analyze(out,split,arrays,scenes,refs)


def analyze(out,split,arrays=None,scenes=None,refs=None):
    from scripts.analyze_geometric_modes import evaluate_pool
    if arrays is None:
        with np.load(out/'pool.npz') as a:arrays={k:a[k] for k in a.files}
        scenes=read(out/'per_scene.json');folder,role=dataset(split)
        labels={r['id']:r for r in lines(folder/'supervision.jsonl') if r['split']==role}
        refs={str(i):references(labels[str(i)]) for i in arrays['ids']}
    parents=arrays['parents']
    if split=='paired_dev':parents=np.array([p.rsplit('_',1)[0] for p in parents])
    summary,rows=evaluate_pool(arrays['paths'],arrays['labels'].astype(bool),arrays['ids'],parents,[s['candidates'] for s in scenes],refs,arrays['q'])
    if split=='paired_dev':
        summary['paired']=pair_metrics(rows,refs,arrays)
        requested=read(DATA/'registration.json')['policy']['dev_families']*9
        summary['requested_requests']=requested;summary['missing_requests']=requested-len(rows)
        summary['all_requested_valid_rate']=summary['CandidateValidRate']*len(rows)/requested
        summary['all_requested_modes']=summary['K']['8']['GeometricModeCount']*len(rows)/requested
    write(out/'RESULTS.json',summary);write(out/'rows.json',rows)
    write(out/'REFERENCE_GEOMETRY.json',{k:dict(paths=v['paths'].tolist(),truth={a:b.tolist() for a,b in v['truth'].items()},config=v['config']) for k,v in refs.items()})
    print(json.dumps(dict(out=str(out),valid=summary['CandidateValidRate'],modes=summary['K']['8']['GeometricModeCount'],paired=summary.get('paired',{}).get('metrics'))),flush=True)


def pair_metrics(rows,refs,arrays):
    grouped={};pathmap=dict(zip(arrays['ids'],arrays['paths']));eventmap=dict(zip(arrays['ids'],arrays['events']))
    for row in rows:
        prefix,target=row['id'].rsplit('_target',1);family,variant=prefix.rsplit('_',1)
        grouped.setdefault((family,target),{})[variant]=row
    details=[]
    for (family,target),variants in sorted(grouped.items()):
        if set(variants)!={'open','closed','shifted'}:continue
        pred={};witness={}
        for v,row in variants.items():
            pred[v]={tuple(c['declared_passage_type']) for c in row['candidates'] if c['TipValid'] and c['declared_passage_type'] is not None}
            witness[v]={tuple(c['declared_passage_type']) for c in refs[row['id']]['reference_candidates'] if c['TipValid'] and c['declared_passage_type'] is not None}
        common=witness['open']&witness['closed'];shared_hit=common&pred['open']&pred['closed']
        config=refs[variants['closed']['id']]['config']
        from scripts.observed_layout_variation import gap_certificates
        closed=[c for c in gap_certificates(config) if c['closed_for_this_low_relation']]
        assert len(closed)==1
        changed=closed[0]['row'];opened={m for m in witness['open'] if m[changed]=='gap1'}
        closedref=refs[variants['closed']['id']]
        _,copied=check_candidates(pathmap[variants['open']['id']],eventmap[variants['open']['id']],
            closedref['label'],closedref['current'],closedref['truth'],closedref['config'])
        static_valid=float(np.mean([c['TipValid'] for c in copied]))
        adapted_valid=variants['closed']['K']['8']['ValidCount']/8
        low_attempt=0
        for path in pathmap[variants['closed']['id']]:
            x=config['row_x'][changed];ys=config['post_y'][changed]
            for a,b in zip(path[:-1],path[1:]):
                if a[0]<=x<b[0]:
                    p=a+(x-a[0])/(b[0]-a[0])*(b-a)
                    lo,hi=closed[0]['low_height_band_m']
                    low_attempt+=int(ys[0]<=p[1]<=ys[1] and lo<=p[2]<=hi);break
        details.append(dict(family=family,target=target,shared_witness_count=len(common),shared_hit=len(shared_hit),
            shared_recall=len(shared_hit)/len(common) if common else None,
            conditional_retention=len(shared_hit)/len(common&pred['open']) if common&pred['open'] else None,
            opened_witness_count=len(opened),opened_hit=len(opened&pred['open']),
            opened_recall=len(opened&pred['open'])/len(opened) if opened else None,
            closed_low_center_attempt_rate=low_attempt/8,
            static_open_paths_in_closed_valid=static_valid,closed_adaptation_gain=adapted_valid-static_valid,
            shifted_shared_recall=len(witness['open']&witness['shifted']&pred['open']&pred['shifted'])/len(witness['open']&witness['shifted']) if witness['open']&witness['shifted'] else None))
    metrics={k:float(np.mean([r[k] for r in details if r[k] is not None])) if any(r[k] is not None for r in details) else None
        for k in ('shared_recall','conditional_retention','opened_recall','closed_low_center_attempt_rate','shifted_shared_recall','closed_adaptation_gain','static_open_paths_in_closed_valid')}
    return dict(metrics=metrics,complete_family_targets=len(details),details=details,
        copied_path_checker_calls=8*len(details),additional_model_generations=0,
        scope='Positive witnessed relation recall; opened modes require certified low-gap closure. Center-attempt diagnostic counts invalid candidates too; not exhaustive infeasible modes.')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--split',choices=['paired_dev','old_dev','dev32','SCORE_TRAIN','DEV_SCORE','CALIBRATION'],required=True)
    p.add_argument('--checkpoint');a=p.parse_args();infer(a.arm,a.seed,a.split,a.checkpoint)
