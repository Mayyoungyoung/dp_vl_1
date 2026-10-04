"""Fixed Ordinary q transfer evaluation; unchanged generator forward and metric rules."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import ROOT,RUN,OLD,read,write,sha,lines
POLICY_ORDINARY_HASH='574e89ceb1358af474036575a7183edb67c1ee2427a90bb7254135fb6493d96a'
def infer(split, seed, arm):
    DATA=OLD if split=="old_dev" else ROOT/"data/geometric_modes_v1/DEV_MODEL"
    OUT=ROOT/"runs/segment_clearance_v1/evaluation_v2"/split
    checkpoint=(RUN/("M8_ordinary_seed%d_expanded"%seed) if arm=="A" else ROOT/"runs/segment_clearance_v1"/("B_seed%d"%seed))/"last.pt"
    import torch
    from routeset.observed_probability import load_scored_planner
    from scripts.train_observed_geometry import read_geometry
    from scripts.evaluate_observed_two_row import scene_metrics
    import os
    import random
    assert os.environ['CUDA_VISIBLE_DEVICES']=='1'
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.35)
    torch.manual_seed(0);np.random.seed(0);random.seed(0)
    manifest_file=DATA/('export_manifest.json' if split=='old_dev' else 'manifest.json')
    manifest=read(manifest_file)
    # Read only existing role-scoped exports, never raw collector roots.
    cache=DATA/'qwen_cache'; cc=read(cache/'cache_config.json')
    assert cc['manifest_sha256']==sha(DATA/'observations.jsonl') and cc['model_trainable_parameter_count']==0
    assert cc['revision']=='89644892e4d85e24eaac8bacfd4f463576704203'
    old_cc=read(ROOT/'data/observation_two_row_composite108_v1/qwen_cache/cache_config.json')
    for key in ('revision','processor','max_pixels','dtype','input_contract'):
        assert cc[key]==old_cc[key], 'Frozen feature protocol mismatch: '+key
    rows=[r for r in lines(DATA/'observations.jsonl') if r['split']=='DEV_MODEL'];labels={r['id']:r for r in lines(DATA/'supervision.jsonl')}
    for model in [dict(path=str(RUN/'M8_ordinary_seed0_expanded_scores/deployment_seed0/planner.pt'),sha256=POLICY_ORDINARY_HASH)]:
        assert sha(model['path'])==model['sha256']
        planner=load_scored_planner(model['path'],'cuda');planner.requires_grad_(False)
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert state['step']==3000 and state['seed']==seed
        planner.generator.load_state_dict(state['model'],strict=True);planner.eval()
        folder=OUT/('%s_seed%d'%(arm,seed));folder.mkdir(parents=True,exist_ok=False)
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
        write(folder/'receipt.json',dict(model=model,inputs_sha256=inputs_sha,manifest_sha256=sha(manifest_file),
            pool_sha256=sha(folder/'pool.npz'),requests=len(rows),complete_path_states=len(rows)*8,training_steps=0,new_qwen_requests=0,
            generator_checkpoint_sha256=sha(checkpoint),generator_seed=seed,scorer_seed=0,inference_seed=0,rng_before_sha256=sha(folder/'rng_before.pt'),M=8,H=24,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),configuration_tuning=False))
        from scripts.analyze_geometric_modes import load_references,evaluate_pool,SOURCES
        refs=load_references(DATA,values['ids'])
        summary,mode_rows=evaluate_pool(np.array(values['paths']),np.array(values['labels'],bool),values['ids'],values['parents'],[r['candidates'] for r in summaries],refs,np.array(values['q']),np.array(values['pi']))
        write(folder/'RESULTS.json',summary);write(folder/'rows.json',mode_rows)
        write(folder/'reference_source_sha256.json',SOURCES)
        if seed==0 and arm=='B':
            geometry_export={k:dict(paths=v['paths'].tolist(),truth={x:y.tolist() for x,y in v['truth'].items()}) for k,v in refs.items()}
            write(folder/'REFERENCE_GEOMETRY.json',geometry_export)
        del planner;torch.cuda.empty_cache()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--arm',choices=['A','B'],required=True);p.add_argument('--split',choices=['old_dev','dev32'],required=True)
    a=p.parse_args();infer(a.split,a.seed,a.arm)
