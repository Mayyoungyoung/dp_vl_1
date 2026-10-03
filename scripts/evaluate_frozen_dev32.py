"""One frozen v1 transfer evaluation on preregistered new DEV_MODEL32 only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import ROOT, RUN, EXT, SOURCE, read, write, sha, lines, contained

DATA=ROOT/'data/geometric_modes_v1/DEV_MODEL'
OUT=ROOT/'runs/geometric_modes_v1/dev32_pools'
POLICY=SOURCE/'configs/geometric_modes_v1.json'

def export():
    policy=read(POLICY)
    registration=read(EXT/'registration.json')
    plans=registration['parent_plan'][256:288]
    assert len(plans)==32 and all(p['index']==256+i and p['parent_id']=='two_row_reach_%d'%(400256+i) and p['role']=='DEV_MODEL' for i,p in enumerate(plans))
    # Freeze deployed bytes before ANY new observation/reference payload read.
    models={}
    for arm,key in [('ordinary','ordinary'),('balanced_probability','balanced')]:
        file=RUN/('M8_%s_seed0_expanded_scores'%arm)/'deployment_seed0/planner.pt'
        assert sha(file)==policy[key+'_planner_sha256']
        models[arm]=dict(path=str(file),sha256=sha(file))
    DATA.mkdir(parents=True,exist_ok=False)
    write(DATA/'frozen_models.json',models)
    observations=[];labels=[];inventory=[];hashes={str(POLICY):sha(POLICY),str(EXT/'registration.json'):sha(EXT/'registration.json')}
    for plan in plans:
        parent=plan['parent_id'];index=plan['index'];folder=EXT/'parents/DEV_MODEL'/parent
        closure=read(EXT/'closures'/('%03d.json'%index))
        assert closure['parent_id']==parent and closure['index']==index and closure['role']=='DEV_MODEL'
        if closure.get('model_eligible'):
            assert closure['actual_geometry_1mm_sha256']==plan['registered_geometry_1mm_sha256']
        inventory.append(dict(parent_id=parent,closure=closure))
        if not (folder/'observations.jsonl').exists():
            inventory[-1]['missing_observations']=3;continue
        artifacts=read(folder/'artifact_hashes.json')
        assert sha(folder/'artifact_hashes.json')==closure['mechanical_files_sha256']['artifact_hashes.json']
        hashes[str(folder/'artifact_hashes.json')]=sha(folder/'artifact_hashes.json')
        def checked(name):
            path=(folder/name).resolve()
            if not contained(path,folder) or name not in artifacts or sha(path)!=artifacts[name]:raise ValueError('Source hash/containment failed: '+name)
            hashes[str(path)]=artifacts[name];return str(path)
        rows=lines(checked('observations.jsonl'));attempts=lines(checked('attempts.jsonl'))
        original={r['id']:r for r in lines(checked('supervision.jsonl'))}
        inventory[-1]['actual_observations']=len(rows)
        inventory[-1]['reference_attempts']=len(attempts)
        inventory[-1]['successful_reference_attempts']=sum(bool(r['success']) for r in attempts)
        for row in rows:
            assert row['parent_id']==parent and row['split']=='DEV_MODEL'
            positive=sorted([r for r in attempts if r['input_id']==row['id'] and r['success']],key=lambda r:r['attempt'])
            routes=[checked(parent+'/'+r['trajectory']['file']) for r in positive]
            if row['id'] in original:
                assert [str((folder/f).resolve()) for f in original[row['id']]['routes']]==routes
            cfg=checked('route_configs/'+parent+'.json')
            labels.append(dict(id=row['id'],parent_id=parent,split='DEV_MODEL',task='rlbench_derived_two_row_reach',
                observation=checked(parent+'/observation.npz'),routes=routes,route_types=[r['actual_route_type'] for r in positive],
                verification_only=checked(parent+'/verification_only.npz'),route_config=cfg,route_config_sha256=sha(cfg),
                semantic_targets=dict(centers=plan['config']['goal_xyz'],target_index=int(row['id'].rsplit('target',1)[1]),tolerance=.03)))
            observations.append(dict(row,image=checked(row['image'])))
    for name,rows in [('observations',observations),('supervision',labels)]:
        (DATA/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write(DATA/'manifest.json',dict(role='DEV_MODEL',requested_parents=32,requested_inputs=96,actual_inputs=len(observations),
        parents=inventory,source_files_sha256=hashes,models=models,policy_sha256=sha(POLICY),TEST_LOCKED_access=False,
        output_files_sha256={n:sha(DATA/n) for n in ('observations.jsonl','supervision.jsonl')}))
    print(json.dumps(dict(exported=len(observations),requested=96)),flush=True)

def infer():
    import torch
    from routeset.observed_probability import load_scored_planner
    from scripts.train_observed_geometry import read_geometry
    from scripts.evaluate_observed_two_row import scene_metrics
    import os
    import random
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    manifest=read(DATA/'manifest.json')
    for f,h in manifest['source_files_sha256'].items():assert sha(f)==h
    for f,h in manifest['output_files_sha256'].items():assert sha(DATA/f)==h
    cache=DATA/'qwen_cache'; cc=read(cache/'cache_config.json')
    assert cc['manifest_sha256']==sha(DATA/'observations.jsonl') and cc['model_trainable_parameter_count']==0
    assert cc['revision']=='89644892e4d85e24eaac8bacfd4f463576704203'
    old_cc=read(ROOT/'data/observation_two_row_composite108_v1/qwen_cache/cache_config.json')
    for key in ('revision','processor','max_pixels','dtype','input_contract'):
        assert cc[key]==old_cc[key], 'Frozen feature protocol mismatch: '+key
    rows=lines(DATA/'observations.jsonl');labels={r['id']:r for r in lines(DATA/'supervision.jsonl')}
    for arm,model in manifest['models'].items():
        assert sha(model['path'])==model['sha256']
        planner=load_scored_planner(model['path'],'cuda');planner.requires_grad_(False)
        folder=OUT/arm;folder.mkdir(parents=True,exist_ok=False)
        torch.save(dict(seed=0,cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()),folder/'rng_before.pt')
        values={k:[] for k in ('paths','events','q','pi','ids','parents','labels')};summaries=[];inputs_sha={}
        # First generate and seal every actual path without invoking checker.
        for row in rows:
            label=labels[row['id']];key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
            file=cache/(key+'.npz');inputs_sha[str(file)]=sha(file)
            with np.load(file) as a:
                assert str(a['image_sha256'].item())==sha(row['image'])
                for k in ('id','parent_id','split'):assert str(a[k].item())==row[k]
                feat=np.concatenate([a['mean_hidden'].reshape(-1),a['last_hidden'].reshape(-1)]).astype('float32')
            geo=read_geometry(row['image'],label['observation'],2)
            with np.load(label['observation']) as a:current=np.r_[a['gripper_pose'],np.asarray(a['gripper_open']).reshape(1)].astype('float32')
            inputs=dict(features=torch.tensor(feat[None],device='cuda'),current=torch.tensor(current[None],device='cuda'),
                **{k:torch.tensor(v[None],device='cuda') for k,v in geo.items()})
            with torch.inference_mode():out=planner(**inputs,return_k=4)
            pred={k:out[k][0].cpu().numpy() for k in ('paths','events','q','pi')}
            np.savez_compressed(folder/(row['id']+'.npz'),**pred)
            for k in pred:values[k].append(pred[k])
            values['ids'].append(row['id']);values['parents'].append(row['parent_id'])
        for n,row in enumerate(rows):
            label=labels[row['id']]
            with np.load(label['observation']) as a:current={k:a[k] for k in ('gripper_pose','gripper_open')}
            with np.load(label['verification_only']) as a:truth={k:a[k] for k in ('obstacle_centers','obstacle_halfsizes')}
            metrics,candidates=scene_metrics(values['paths'][n],values['events'][n],current,truth,label['semantic_targets'],label['route_types'],read(label['route_config']))
            values['labels'].append([c['TipValid'] for c in candidates])
            summaries.append(dict(id=row['id'],metrics=metrics,candidates=candidates,prediction_sha256=sha(folder/(row['id']+'.npz'))))
        np.savez_compressed(folder/'pool.npz',**{k:np.array(v) for k,v in values.items()})
        write(folder/'per_scene.json',summaries)
        write(folder/'receipt.json',dict(model=model,inputs_sha256=inputs_sha,manifest_sha256=sha(DATA/'manifest.json'),
            pool_sha256=sha(folder/'pool.npz'),requests=len(rows),complete_path_states=len(rows)*8,training_steps=0,new_qwen_requests=0,
            generator_seed=0,scorer_seed=0,inference_seed=0,rng_before_sha256=sha(folder/'rng_before.pt'),M=8,H=24,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),configuration_tuning=False))
        del planner;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['export','infer']);a=p.parse_args()
    (export if a.action=='export' else infer)()
