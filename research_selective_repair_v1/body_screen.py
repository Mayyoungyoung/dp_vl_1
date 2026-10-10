"""Seal one body correction per slot,then run unchanged full q and select four."""
import argparse,time,itertools
from pathlib import Path
import numpy as np
import torch
from research_selective_repair_v1.core import *
from research_selective_repair_v1.body_options import options
from research_selective_repair_v1.body_forecast import load,predict,WORDS
from research_selective_repair_v1.body_allocation import allocate
from research_selective_repair_v1.constraints import boxes
from research_selective_repair_v1.constraint_operator import configuration
from routeset.observed_probability import load_scored_planner,route_observation_features
from scripts.research_v3_audit import mode

def screen(name,kind,checkpoint=None,protected=True,word_safe=False,forecast='categorical',all_goals=False,role='DEV_MODEL',continuous=False,family_start=0,families=None,dataset_scope='formal'):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    assert role in ('TRAIN','DEV_MODEL')
    if continuous:assert protected and word_safe and forecast in ('event','crossing','binary') and kind in ('actual','planned','success')
    source=RUN/('constraints_global_interventions_TRAIN_body_v1' if role=='TRAIN' else 'constraints_global_interventions_v1');data=ROOT/'data/selective_repair_interventions_v1'
    receipt=read(source/'SEAL.json')
    assert receipt['current_observation_only'] and sha(source/'sealed_predictions.npz')==receipt['prediction_sha256']
    with np.load(source/'sealed_predictions.npz') as z:
        mask=np.ones(len(z['ids']),bool) if all_goals else np.array([str(i).endswith('_target0') for i in z['ids']]);d={k:z[k][mask] for k in ('ids','paths','events','completed','modes')}
    if families is not None:
        byfamily=np.array(['_'.join(str(i).split('_')[:3]) for i in d['ids']]);chosen=sorted(set(byfamily))[family_start:family_start+families]
        selected=np.isin(byfamily,chosen);d={k:v[selected] for k,v in d.items()}
    with np.load(RUN/'interventions_calibrated_targets_v1/samples.npz') as z:
        legal=(z['splits']==role)&(np.ones(len(z['ids']),bool) if all_goals else np.array([str(i).endswith('_target0') for i in z['ids']]));ctx={str(i):c for i,c in zip(z['ids'][legal],z['context'][legal])}
    assert set(map(str,d['ids'])).issubset(ctx),'Every source request must match the explicitly registered role/current observation'
    with np.load(source/'pool.npz') as z:base_q={str(i):q for i,q in zip(z['ids'],z['q'])}
    static=kind in ('identity','lift','preserved');model=ck=None
    if not static:
        assert checkpoint is not None
        if forecast=='prefix':
            from research_selective_repair_v1.body_prefix_forecast import load as forecast_load,predict as forecast_predict
            from research_selective_repair_v1.body_prefix_probabilities import compose,computation_counts
        elif forecast=='event':
            from research_selective_repair_v1.body_event_forecast import load as forecast_load,predict as forecast_predict,compose
        elif forecast=='crossing':
            from research_selective_repair_v1.body_crossing_measure import load as forecast_load,predict as forecast_predict,compose
        elif forecast=='binary':
            from research_selective_repair_v1.body_binary_forecast import load as forecast_load,predict as forecast_predict
        else:forecast_load,forecast_predict=load,predict
        model,ck=forecast_load(checkpoint)
        if all_goals:
            dataset=ck['settings']['dataset_sha256']
            if dataset_scope=='half':
                assert forecast in ('event','crossing','binary')
                folder='body_halfgoal_events_v2' if forecast in ('event','crossing') else 'body_halfgoal_feedback_v1'
                manifest=read(RUN/folder/'MANIFEST.json');assert manifest['rows']==1152 and manifest['no_DEV_feedback']
            else:manifest=read(RUN/('body_events_data_v1' if forecast in ('event','crossing') else 'body_prefix_data_v1' if forecast=='prefix' else 'body_feedback_data_v3')/'MANIFEST.json')
            assert dataset==manifest['samples_sha256'],'Allgoal screen requires exact frozen TRAIN supervision'
    boxck=torch.load(RUN/'constraints_seed0_v1/last.pt',map_location='cpu',weights_only=False)
    t=lambda x:torch.tensor(x,device='cuda',dtype=torch.float32)
    paths=[];choices=[];probabilities=[];queries=[];forecast_queries=[];torch.cuda.synchronize();tic=time.monotonic()
    for i,ident in enumerate(d['ids']):
        candidates=options(d['paths'][i],d['completed'][i]);prob=np.zeros((8,3,17),np.float32);prob[:,:,0]=1
        planned=None
        cfg=None
        if kind=='planned' or word_safe or forecast in ('prefix','event','crossing','binary'):
            c,h=boxes(t(d['completed'][i:i+1]),boxck['settings']);cfg=configuration(c[0].cpu().numpy(),h[0].cpu().numpy(),boxck['settings'])
            words=[mode(p,cfg) for p in candidates.reshape(24,24,3)];planned=np.array([WORDS.index(w)+1 if w in WORDS else 0 for w in words]).reshape(8,3)
        detail={}
        if continuous:
            from research_selective_repair_v1.body_predictive_repair import repair as predictive_repair
            repaired,choice,prob,detail=predictive_repair(model,ck,d['paths'][i],d['completed'][i],ctx[str(ident)],cfg,forecast=forecast,kind=kind)
            paths.append(repaired);choices.append(choice);probabilities.append(prob);queries.append(33);forecast_queries.append(detail)
            continue
        if not static:
            prediction=forecast_predict(model,ck,candidates.reshape(24,24,3),np.broadcast_to(d['completed'][i],(24,4,3)),np.broadcast_to(ctx[str(ident)],(24,128)))
            if forecast=='prefix':
                prob,detail=compose(prediction,candidates.reshape(24,24,3),cfg);prob=prob.reshape(8,3,17)
                detail.update(computation_counts(ck['settings'],24))
            elif forecast in ('event','crossing'):
                prob,detail=compose(prediction,candidates.reshape(24,24,3),cfg);prob=prob.reshape(8,3,17)
            elif forecast=='binary':
                success=prediction.reshape(8,3);prob=np.zeros((8,3,17),np.float32);prob[:,:,0]=1-success
                for slot in range(8):
                    for option in range(3):prob[slot,option,int(planned[slot,option])]+=success[slot,option]
            else:prob=prediction.reshape(8,3,17)
        allowed=(planned==planned[:,:1]) if word_safe else None
        choice,count=allocate(prob,kind,protected,planned,allowed);paths.append(candidates[np.arange(8),choice]);choices.append(choice);probabilities.append(prob);queries.append(count);forecast_queries.append(detail)
    torch.cuda.synchronize();seconds=time.monotonic()-tic;paths=np.asarray(paths)
    np.savez_compressed(out/'sealed_predictions.npz',ids=d['ids'],paths=paths,events=d['events'],modes=d['modes'],drafts=d['paths'],completed=d['completed'],choices=choices,probabilities=probabilities)
    seal=sha(out/'sealed_predictions.npz');write(out/'SEAL.json',dict(prediction_sha256=seal,source_pool_sha256=sha(source/'pool.npz'),checkpoint_sha256=sha(checkpoint) if checkpoint else None,role=role,current_observation_only=True,kind=kind,forecast=forecast,all_goals=all_goals,dataset_scope=dataset_scope,continuous=continuous,protected=protected,word_safe=word_safe,predicted_signature_calls_per_request=40 if continuous else 24 if word_safe or kind=='planned' or forecast in ('prefix','event','crossing','binary') else 0,internal_alternatives_per_request=24,critic_route_forwards_per_request=264 if continuous else 0 if static else 24,final_candidates_per_request=8,allocation_objective_queries=queries,forecast_query_counts=forecast_queries,cached_correction_seconds=seconds,latency_scope='Correction only; common64step box geometry and original observation encoders/scorer excluded. Full online latency still required.'))
    # Independent geometry labels only after all predictions and choices are sealed.
    observations={r['id']:r for r in lines(data/'export/observations.jsonl') if r['split']==role}
    labels={r['id']:r for r in lines(data/'export/supervision.jsonl') if r['split']==role}
    scorer=load_scored_planner(Q,'cuda').requires_grad_(False);center,_=center_model();rows=[];qs=[];valids=[];selected=[];hashes={}
    for i,ident in enumerate(d['ids']):
        ident=str(ident);inp=inputs_for(observations[ident],labels[ident],data/'export/qwen_cache',torch,hashes);runner=SceneRunner(center,scorer,inp)
        with torch.no_grad():
            p=t(paths[i:i+1]);ev=t(d['events'][i:i+1]);nodes,cx=route_observation_features(p,ev,inp['current'],inp['world_xyz'],inp['rgb'],inp['valid_mask'],runner.qgeo['point_features'],runner.qcontext,runner.qgeo['anchor_xyz'])
            q=(scorer.scorer((nodes-scorer.nodes_mean)/scorer.nodes_std,(cx-scorer.context_mean)/scorer.context_std)/scorer.temperature).sigmoid().cpu().numpy()
        ref=references(labels[ident]);a=assess(paths[i:i+1],d['events'][i:i+1],q,ref);b=assess(d['paths'][i:i+1],d['events'][i:i+1],base_q[ident][None],ref)
        old=wordset(d['paths'][i],b['valid'][0],ref['config']);new=wordset(paths[i],a['valid'][0],ref['config'])
        rows.append(dict(id=ident,family=ident.split('_')[2],utility=a['utility'][0].tolist(),base_utility=b['utility'][0].tolist(),added=len(new-old),lost=len(old-new),repaired=int((~b['valid'][0]&a['valid'][0]).sum()),damaged=int((b['valid'][0]&~a['valid'][0]).sum()),base_valid=int(b['valid'][0].sum()),raw_mode_changed=int((a['raw_words'][0]!=b['raw_words'][0]).sum()),selected_indices=a['selected'][0].tolist(),valid=a['valid'][0].tolist()))
        qs.append(q[0]);valids.append(a['valid'][0]);selected.append(a['selected'][0])
    assert sha(out/'sealed_predictions.npz')==seal
    np.savez_compressed(out/'pool.npz',ids=d['ids'],paths=paths,events=d['events'],drafts=d['paths'],modes=d['modes'],q=qs,valid=valids,selected=selected)
    write(out/'ROWS.json',rows);write(out/'rows.json',rows)
    report=dict(kind=kind,forecast=forecast,role=role,continuous=continuous,dataset_scope=dataset_scope,protected=protected,word_safe=word_safe,mean_utility=np.mean([r['utility'] for r in rows],0).tolist(),requests=len(rows),damaged=sum(r['damaged'] for r in rows),raw_mode_changed=sum(r['raw_mode_changed'] for r in rows),correction_seconds=seconds,head_scope='Initial C0 allocation head screen; own matched TRAIN heads required before formal acceptance',learned_task_scope='all3goals,registered role variants;8family pilot' if all_goals and dataset_scope=='half' else 'all3goals,registered role variants' if all_goals else 'target0 pilot only',actual_body_execution='Not run by this geometry screen',locked_access=False)
    write(out/'SUMMARY.json',report);print(report,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--kind',choices=['identity','lift','preserved','success','planned','actual','coordinate'],required=True);p.add_argument('--checkpoint',type=Path);p.add_argument('--unprotected',dest='protected',action='store_false');p.add_argument('--word-safe',action='store_true');p.add_argument('--forecast',choices=['categorical','prefix','event','crossing','binary'],default='categorical');p.add_argument('--all-goals',action='store_true');p.add_argument('--role',choices=['TRAIN','DEV_MODEL'],default='DEV_MODEL');p.add_argument('--continuous',action='store_true');p.add_argument('--family-start',type=int,default=0);p.add_argument('--families',type=int);p.add_argument('--dataset-scope',choices=['formal','half'],default='formal');screen(**vars(p.parse_args()))
