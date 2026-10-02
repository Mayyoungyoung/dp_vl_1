"""TRAIN-only audit of reference resampling and learned obstacle-route failures.

Uses the existing 2cm tip-box protocol unchanged. No new training, thresholds,
route repair or success filtering. Unknown passage types remain valid if all
the original TipValid gates pass.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# Legacy collection helpers import their sibling modules by basename.
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.collect_obstacle_reach import tip_polyline_clear
from scripts.evaluate_observed_obstacles import scene_metrics,PROTOCOL,sha256
from scripts.export_observation_roles import selected_rows


CLEARANCE=.02
TRAIN_PARENTS={'obstacle_reach_%06d'%number for number in range(272000,272032)}
GATES=('finite_xyz','finite_event_values','semantic_goal_correct','starts_at_current_state',
       'tip_segments_clear','event_state_sequence_correct','TipValid')


def train_path(data,value,parent):
    path=(data/value).resolve()
    if parent not in TRAIN_PARENTS or path.parent.name!=parent:
        raise ValueError('raw file does not belong to the explicitly selected TRAIN parent')
    return path


def first_collision(path,geometry):
    path=np.asarray(path,dtype=np.float64)
    centers,halfsizes=np.asarray(geometry['obstacle_centers']),np.asarray(geometry['obstacle_halfsizes'])
    if not np.isfinite(path).all() or tip_polyline_clear(path,centers,halfsizes,CLEARANCE):return None
    distances=np.linalg.norm(np.diff(path,axis=0),axis=-1);total=float(distances.sum())
    for segment in range(len(path)-1):
        for obstacle in range(len(centers)):
            if not tip_polyline_clear(path[segment:segment+2],centers[obstacle:obstacle+1],halfsizes[obstacle:obstacle+1],CLEARANCE):
                return dict(segment_index=segment,segment_count=len(path)-1,obstacle_index=obstacle,
                    start_xyz=path[segment].tolist(),end_xyz=path[segment+1].tolist(),
                    normalized_vertex_progress=segment/(len(path)-1),
                    normalized_arc_before_segment=float(distances[:segment].sum()/total) if total else 0.)
    raise AssertionError('segment localization disagrees with fixed whole-polyline collision helper')


def one_route(path,opened,current,geometry,label):
    _,rows=scene_metrics(np.asarray(path)[None],np.asarray(opened)[None],current,geometry,
                        label['semantic_targets'],label.get('route_types',[]),clearance=CLEARANCE)
    row=rows[0];row['first_collision']=first_collision(path,geometry)
    assert bool(row['first_collision'] is None)==bool(row['tip_segments_clear'] or not row['finite_xyz'])
    return row


def summarize(rows):
    locations=[row['first_collision'] for row in rows if row['first_collision'] is not None]
    return dict(candidates=len(rows),independent_gate_failures={gate:sum(not row[gate] for row in rows) for gate in GATES},
        tip_valid_candidates=sum(row['TipValid'] for row in rows),
        tip_valid_unknown_type_candidates=sum(row['TipValid'] and row['declared_passage_type'] is None for row in rows),
        first_collision_segment_histogram=dict(Counter(str(row['segment_index']) for row in locations)),
        first_collision_arc_quartile_histogram=dict(Counter(str(min(3,int(row['normalized_arc_before_segment']*4))) for row in locations)),
        first_collision_arc_before_segment_mean=float(np.mean([row['normalized_arc_before_segment'] for row in locations])) if locations else None)


def read_train(data):
    manifest=json.loads((data/'manifest.json').read_text())
    if manifest['acceptance']['tip_polyline_clearance_m']!=CLEARANCE:raise ValueError('original 2cm margin required')
    observations=selected_rows(data/'observations.jsonl',TRAIN_PARENTS)
    labels=selected_rows(data/'supervision.jsonl',TRAIN_PARENTS)
    if len(observations)!=96 or len(labels)!=96:raise ValueError('all original32 TRAIN parents/96 observations required')
    if {row['parent_id'] for row in observations}!=TRAIN_PARENTS:raise ValueError('TRAIN parent coverage changed')
    if any(set(row)!={'id','parent_id','split','image','instruction'} or row['split']!='TRAIN' for row in observations):
        raise ValueError('TRAIN-only strict input whitelist required')
    labels={row['id']:row for row in labels}
    if sum(bool(row['routes']) for row in labels.values())!=94:raise ValueError('all94 original supervised TRAIN inputs required')
    current,geometry,hashes={},{},{}
    for row in observations:
        label=labels[row['id']];parent=row['parent_id']
        if label['split']!='TRAIN' or label['parent_id']!=parent or label['semantic_targets']['tolerance']!=.03:
            raise ValueError('TRAIN metadata identity or semantic tolerance changed')
        for key in ('observation','verification_only'):
            path=train_path(data,label[key],parent);hashes[str(path)]=sha256(path)
            with np.load(path,allow_pickle=False) as archive:
                if key=='observation':current[row['id']]={name:archive[name] for name in ('gripper_pose','gripper_open')}
                else:geometry[row['id']]={name:archive[name] for name in ('obstacle_centers','obstacle_halfsizes')}
    return observations,labels,current,geometry,hashes


def prepare_forward(data,observations,labels,current,config,hashes):
    from scripts.train_observed_geometry import read_geometry,POINT_FIELDS
    from routeset.observed_route_head import QWEN_REVISION,CACHE_KEYS
    import hashlib
    cache=Path(config['cache_dir']);cache_config=json.loads((cache/'cache_config.json').read_text())
    if cache_config['manifest_sha256']!=sha256(data/'observations.jsonl') or cache_config['revision']!=QWEN_REVISION or cache_config['model_trainable_parameter_count']!=0:
        raise ValueError('unchanged genuine frozen Qwen cache required')
    features,states,points,parent_index,point_indices=[],[],[],{},[]
    parent_sources={}
    for row in observations:
        key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
        path=cache/(key+'.npz');hashes[str(path)]=sha256(path)
        image=train_path(data,row['image'],row['parent_id']);hashes[str(image)]=sha256(image)
        with np.load(path,allow_pickle=False) as archive:
            if set(archive.files)!=CACHE_KEYS:raise ValueError('cache whitelist violated')
            if any(str(archive[name].item())!=row[name] for name in ('id','parent_id','split')):raise ValueError('TRAIN cache identity mismatch')
            if str(archive['image_sha256'].item())!=hashes[str(image)]:raise ValueError('TRAIN image changed')
            features.append(np.r_[archive['mean_hidden'],archive['last_hidden']].astype(np.float32))
        state=current[row['id']]
        states.append(np.r_[state['gripper_pose'],np.asarray(state['gripper_open']).reshape(1)].astype(np.float32))
        observation=train_path(data,labels[row['id']]['observation'],row['parent_id'])
        sources=(str(image),str(observation))
        if row['parent_id'] in parent_sources and parent_sources[row['parent_id']]!=sources:
            raise ValueError('same-parent current geometry changed across language instructions')
        parent_sources[row['parent_id']]=sources
        if row['parent_id'] not in parent_index:
            parent_index[row['parent_id']]=len(points)
            points.append(read_geometry(image,observation,config['pixel_stride']))
        point_indices.append(parent_index[row['parent_id']])
    return dict(features=np.stack(features),current=np.stack(states),point_indices=np.asarray(point_indices),
                points={name:np.stack([point[name] for point in points]) for name in POINT_FIELDS})


def predict_last(run,observations,forward,config):
    import torch
    from routeset.observed_geometry import ObservedGeometryRouteHead
    checkpoint=torch.load(run/'last.pt',map_location='cpu',weights_only=False)
    if checkpoint['step']!=1000 or checkpoint['config']['dataset_fingerprint']!=config['dataset_fingerprint']:
        raise ValueError('last1000 source mismatch')
    model=ObservedGeometryRouteHead(config['feature_dim'],config['horizon'],config['candidates'],config['width'],config['depth'],
        config['point_width'],config['endpoint_residual_bound'],anchor_mode=config.get('anchor_mode','soft')).eval()
    model.load_state_dict(checkpoint['model'],strict=True)
    paths,events=[],[]
    with torch.no_grad():
        for start in range(0,len(observations),8):
            ids=np.arange(start,min(start+8,len(observations)))
            inputs={name:torch.as_tensor(forward[name][ids]) for name in ('features','current')}
            inputs.update({name:torch.as_tensor(value[forward['point_indices'][ids]]) for name,value in forward['points'].items()})
            xyz,opened,_=model(**inputs);paths.append(xyz.numpy());events.append(opened.numpy())
    return np.concatenate(paths),np.concatenate(events)


def self_test():
    geometry=dict(obstacle_centers=np.array([[0.,0.,0.]]),obstacle_halfsizes=np.array([[.1,.1,.1]]))
    path=np.array([[-.4,.3,0.],[-.4,0.,0.],[.4,0.,0.]])
    collision=first_collision(path,geometry)
    assert collision['segment_index']==1 and collision['obstacle_index']==0
    assert np.isclose(collision['normalized_arc_before_segment'],.3/1.1)
    assert first_collision(path+np.array([0.,0.,1.]),geometry) is None
    boundary=np.array([[-.3,.12,0.],[.3,.12,0.]])
    assert first_collision(boundary,geometry) is not None
    print(json.dumps(dict(status='passed',checks=['first segment matches fixed helper','arc coordinate','clear polyline','2cm boundary unchanged'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--data',type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--seeds',nargs='+',type=int,default=[0])
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if not all((args.root,args.data,args.output)):parser.error('root/data/output required')
    if args.output.exists():raise FileExistsError('new diagnostic output required')
    if len(set(args.seeds))!=len(args.seeds) or not set(args.seeds)<={0,1,2}:raise ValueError('available unique seeds0/1/2 only')
    import torch
    from routeset.observed_route_head import resample_event_segments
    from scripts.reevaluate_anchor_fixed_step import source_run
    torch.set_num_threads(1);started=time.perf_counter()
    observations,labels,current,geometry,hashes=read_train(args.data)
    references=[];reference_ids=set()
    for row in observations:
        identifier=row['id'];label=labels[identifier]
        for filename in label['routes']:
            path=train_path(args.data,filename,row['parent_id']);hashes[str(path)]=sha256(path)
            with np.load(path,allow_pickle=False) as archive:
                poses,opened=archive['gripper_pose'],archive['gripper_open']
                saved24=archive['xyz_24']
            sampled,events=resample_event_segments(poses,opened,24)
            raw=one_route(poses[:,:3],opened,current[identifier],geometry[identifier],label)
            h24=one_route(sampled,events,current[identifier],geometry[identifier],label)
            reference_ids.add(identifier)
            references.append(dict(scene_id=identifier,route_file=str(path),raw_steps=len(poses),raw=raw,training_h24=h24,
                saved_collector_h24_clear=bool(tip_polyline_clear(saved24,geometry[identifier]['obstacle_centers'],geometry[identifier]['obstacle_halfsizes'],CLEARANCE)),
                saved_collector_vs_training_h24_max_abs_m=float(np.abs(saved24-sampled).max())))
    if len(reference_ids)!=94 or len(references)!=181:raise ValueError('all original94 TRAIN inputs/181 references required')
    args.output.mkdir(parents=True)
    results={};forward=None;first_config=None
    expected_ids={row['id'] for row in observations}
    for seed in args.seeds:
        for method in ('soft','peak'):
            run=source_run(args.root,'obstacle32',method,seed)
            config=json.loads((run/'config.json').read_text())
            if Path(config['observations']).resolve()!=(args.data/'observations.jsonl').resolve():
                raise ValueError('source checkpoint does not use the requested old32 TRAIN dataset')
            if config['seed']!=seed or config.get('anchor_mode','soft')!=('soft' if method=='soft' else 'straight_through_peak'):
                raise ValueError('source arm mismatch')
            if config['pooling']!='both' or config['horizon']!=24 or config['candidates']!=4:raise ValueError('expected shared observed head configuration')
            if first_config is None:
                first_config=config;forward=prepare_forward(args.data,observations,labels,current,config,hashes)
            elif config['dataset_fingerprint']!=first_config['dataset_fingerprint']:raise ValueError('same TRAIN data required across source arms')
            training_hashes=json.loads((run/'source_hashes.json').read_text())
            for name,expected in training_hashes.items():
                if name in hashes and hashes[name]!=expected:raise ValueError('selected TRAIN source changed after original training')
            for stage in ('best','last1000'):
                if stage=='best':
                    source=run/'train/predictions.npz'
                    with np.load(source,allow_pickle=False) as archive:
                        ids=list(map(str,archive['scene_ids']));paths,events=archive['paths'],archive['gripper_open']
                    provenance=dict(prediction_source=str(source),prediction_sha256=sha256(source),checkpoint_sha256=sha256(run/'best.pt'),
                        best_step=json.loads((run/'summary.json').read_text())['best_step'])
                else:
                    ids=[row['id'] for row in observations];paths,events=predict_last(run,observations,forward,config)
                    out=args.output/(method+'_seed'+str(seed)+'_last1000_train_predictions.npz')
                    np.savez_compressed(out,scene_ids=np.asarray(ids),parent_ids=np.asarray([labels[i]['parent_id'] for i in ids]),paths=paths,gripper_open=events)
                    provenance=dict(prediction_source=str(out),prediction_sha256=sha256(out),checkpoint_sha256=sha256(run/'last.pt'),checkpoint_step=1000)
                if set(ids)!=expected_ids or len(ids)!=96 or paths.shape!=(96,4,24,3):raise ValueError('all96 TRAIN inputs/all4 candidates required')
                candidates=[]
                for identifier,xyz,opened in zip(ids,paths,events):
                    for candidate,(path,event) in enumerate(zip(xyz,opened)):
                        result=one_route(path,event,current[identifier],geometry[identifier],labels[identifier])
                        result.update(scene_id=identifier,candidate=candidate,has_reference=identifier in reference_ids)
                        candidates.append(result)
                key=method+'_seed'+str(seed)+'_'+stage
                results[key]=dict(all96=summarize(candidates),reference94=summarize([row for row in candidates if row['has_reference']]),
                    no_reference2=summarize([row for row in candidates if not row['has_reference']]),per_candidate=candidates,
                    provenance=dict(**provenance,source_run=str(run),source_config_sha256=sha256(run/'config.json'),source_commit=config['code_commit']))
                print(json.dumps(dict(arm=key,all96=results[key]['all96'])),flush=True)
    report=dict(evaluation_protocol=PROTOCOL,diagnostic='TRAIN reference raw/H24 and best/last prediction gate audit',clearance_m=CLEARANCE,
        train_parent_count=32,train_observation_count=96,supervised_train_inputs=94,no_reference_train_inputs=2,references=181,
        reference_raw=summarize([row['raw'] for row in references]),reference_h24=summarize([row['training_h24'] for row in references]),
        raw_valid_to_h24_invalid=sum(row['raw']['TipValid'] and not row['training_h24']['TipValid'] for row in references),
        saved_collector_h24_clear_count=sum(row['saved_collector_h24_clear'] for row in references),
        max_collector_vs_training_h24_abs_m=max(row['saved_collector_vs_training_h24_max_abs_m'] for row in references),
        per_reference=references,prediction_results=results,raw_source_hashes=hashes,
        source_commit=os.environ.get('CODE_COMMIT'),script_sha256=sha256(__file__),
        fixed_evaluator_sha256=sha256(Path(__file__).with_name('evaluate_observed_obstacles.py')),
        fixed_geometry_helper_sha256=sha256(Path(__file__).with_name('collect_obstacle_reach.py')),
        observation_manifest_sha256=sha256(args.data/'observations.jsonl'),supervision_manifest_sha256=sha256(args.data/'supervision.jsonl'),
        total_cpu_seconds=time.perf_counter()-started,raw_dev_or_locked_files_opened=False,
        scope='added physical-box tip segments only; full arm/environment/IK/execution remain unverified; unknown modes are not invalid')
    (args.output/'diagnostic.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ('reference_raw','reference_h24','raw_valid_to_h24_invalid','max_collector_vs_training_h24_abs_m','total_cpu_seconds')}),flush=True)


if __name__=='__main__':main()
