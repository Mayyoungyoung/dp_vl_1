"""Train-only Qwen-feature nearest-reference diagnostic; no RGB-D input.

This is an ordinary small-data retrieval control, not a method contribution or
an equal-information comparison to the RGB-D model. DEV targets never select
the training observation or change its translated reference candidates.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def normalize(features):
    values=np.asarray(features,dtype=np.float64)
    if values.ndim!=2 or not np.isfinite(values).all():raise ValueError('finite feature matrix required')
    norms=np.linalg.norm(values,axis=1,keepdims=True)
    if (norms<=1e-12).any():raise ValueError('zero-norm Qwen feature cannot be retrieved')
    return values/norms


class NearestReferenceBank:
    """The prediction API accepts only cached Qwen features and current state."""
    def __init__(self,features,paths,events,mask,scene_ids):
        self.features=normalize(features)
        self.paths=np.asarray(paths,dtype=np.float32).copy()
        self.events=np.asarray(events,dtype=np.float32).copy()
        self.mask=np.asarray(mask,dtype=bool).copy()
        self.ids=np.asarray(scene_ids,dtype=str).copy()
        n=len(self.features)
        if not n or self.paths.ndim!=4 or self.paths.shape[0]!=n or self.paths.shape[-1]!=3:
            raise ValueError('nonempty TRAIN reference bank [N,R,H,3] required')
        if self.events.shape!=self.paths.shape[:-1] or self.mask.shape!=self.paths.shape[:2]:
            raise ValueError('matching reference paths, open states and mask required')
        if len(self.ids)!=n or len(set(self.ids))!=n or not self.mask.any(1).all():
            raise ValueError('unique TRAIN observation IDs and positive references required')
        if not np.isfinite(self.paths[self.mask]).all() or not np.isfinite(self.events[self.mask]).all():
            raise ValueError('finite positive TRAIN references required')

    def predict(self,features,current,k=4):
        features=normalize(features);current=np.asarray(current,dtype=np.float32)
        if current.shape!=(len(features),8) or not np.isfinite(current).all() or not isinstance(k,int) or k<1:
            raise ValueError('finite current [N,8] and positive integer K required')
        if features.shape[1]!=self.features.shape[1]:raise ValueError('Qwen feature dimension mismatch')
        similarities=features@self.features.T
        paths=[];events=[];details=[]
        for query,scores in enumerate(similarities):
            nearest=int(np.lexsort((self.ids,-scores))[0])
            references=np.flatnonzero(self.mask[nearest])
            chosen=references[np.arange(k)%len(references)]
            original=self.paths[nearest,chosen]
            prediction=original-original[:,:1]+current[query,None,None,:3]
            prediction[:,0]=current[query,:3]
            opened=self.events[nearest,chosen].copy();opened[:,0]=np.clip(current[query,7],0,1)
            paths.append(prediction);events.append(opened)
            details.append(dict(nearest_train_id=str(self.ids[nearest]),nearest_bank_index=nearest,
                feature_cosine_similarity=float(scores[nearest]),positive_references_in_selected_observation=len(references),
                original_reference_indices=chosen.tolist(),complete_proposals=k,discarded_proposals=0,
                duplicated_reference_slots=int(k-len(set(chosen))),
                selection='highest cosine over all positive-reference TRAIN observations; exact ties use ascending observation ID'))
        return np.stack(paths),np.stack(events),details


def fit_train_bank(data):
    # Read reference labels only after selecting TRAIN metadata indices.
    train_ids=np.flatnonzero(data['splits']=='TRAIN')
    train_ids=train_ids[data['path_mask'][train_ids].any(1)]
    return NearestReferenceBank(data['features'][train_ids],data['paths'][train_ids],data['events'][train_ids],
        data['path_mask'][train_ids],data['scene_ids'][train_ids]),train_ids


def run(args):
    import torch
    from routeset.observed_route_head import load_observed_dataset
    from routeset.observed_multitask import check_multitask_model_gate,aggregate_task_parent_reference
    from scripts.train_observed_routes import observation_metrics
    torch.set_num_threads(1)
    started=time.perf_counter();output=Path(args.output)
    if output.exists():raise FileExistsError('use a fresh read-only evaluation output')
    gate=check_multitask_model_gate(args.observations,args.supervision)
    if gate is None:raise ValueError('this registered multitask control requires its current snapshot gate')
    data=load_observed_dataset(args.observations,args.supervision,args.cache_dir,args.horizon,args.pooling)
    tic=time.perf_counter();bank,train_ids=fit_train_bank(data);bank_seconds=time.perf_counter()-tic
    ids=np.flatnonzero(data['splits']=='DEV_MODEL')
    if not len(ids) or set(data['parent_ids'][train_ids])&set(data['parent_ids'][ids]):
        raise ValueError('parent-disjoint positive TRAIN and complete DEV_MODEL inputs required')
    predictions=[];events=[];retrieval=[];times=[]
    for idx in ids:
        tic=time.perf_counter()
        paths,opened,details=bank.predict(data['features'][idx:idx+1],data['current'][idx:idx+1],args.candidates)
        times.append(time.perf_counter()-tic)
        predictions.append(paths[0]);events.append(opened[0]);retrieval.append(details[0])
    predictions,events=np.stack(predictions),np.stack(events)
    # Only after every prediction exists do DEV references/tasks enter metrics.
    metrics,rows=observation_metrics(predictions,events,data,ids)
    aggregate_task_parent_reference(metrics,rows,[data['tasks'][idx] for idx in ids])
    for row,detail,idx in zip(rows,retrieval,ids):
        nearest=train_ids[detail['nearest_bank_index']]
        row.update(retrieval=detail,retrieved_train_task=data['tasks'][nearest],
            task_metadata_use='post-prediction report only; no task or DEV label used for lookup')
    output.mkdir(parents=True)
    np.savez_compressed(output/'predictions.npz',paths=predictions,gripper_open=events,
        scene_ids=data['scene_ids'][ids],parent_ids=data['parent_ids'][ids],
        nearest_train_ids=np.asarray([row['nearest_train_id'] for row in retrieval]),
        cosine_similarity=np.asarray([row['feature_cosine_similarity'] for row in retrieval]))
    write_json(output/'per_scene.json',rows)
    write_json(output/'metrics.json',metrics)
    config=dict(vars(args),code_commit=os.environ.get('CODE_COMMIT'),source_script_sha256=sha(__file__),
        dataset_fingerprint=data['fingerprint'],cache_config=data['cache_config'],model_use_gate=gate,
        bank_train_observation_ids=data['scene_ids'][train_ids].tolist(),
        bank_train_parents=len(set(data['parent_ids'][train_ids])),bank_positive_reference_slots=int(data['path_mask'][train_ids].sum()),
        bank_reference_count_note='references are reused across language paraphrases; these slots are not independent collected trajectories',
        input_fields=['frozen genuine Qwen RGB+language hidden states','current pose/open'],
        excluded_inputs=['task ID','DEV reference/semantic targets','RGB-D geometry'],
        training='no parameter optimization; positive TRAIN demonstration bank only',
        lower_information_scope='memory/retrieval diagnostic, not same-information superiority evidence against RGB-D models')
    write_json(output/'config.json',config);write_json(output/'source_hashes.json',data['source_hashes'])
    summary=dict(metrics=metrics,bank_fit_seconds=bank_seconds,elapsed_s=time.perf_counter()-started,
        cached_qwen_retrieval_cpu_batch1_ms_median=float(np.median(times)*1000),
        cached_qwen_retrieval_cpu_batch1_ms_p95=float(np.percentile(times,95)*1000),
        latency_scope='CPU feature normalization, exhaustive TRAIN cosine lookup, deterministic tie resolution and K translated complete paths; excludes Qwen extraction and input/bank disk loading',
        generation_budget=dict(final_candidates=args.candidates,complete_path_states_per_request=args.candidates,
            total_complete_path_states=len(ids)*args.candidates,extra_drafts=0,candidate_filtering=False),
        prediction_sha256=sha(output/'predictions.npz'),source_script_sha256=sha(__file__),
        evidence_scope='ordinary lower-information nearest-reference baseline; reference reconstruction only, not semantic/geometry/robot success')
    write_json(output/'summary.json',summary)
    write_json(output/'status.json',dict(status='completed',exit_code=0,examples=len(ids)))
    print(json.dumps(summary),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('observations','supervision','cache-dir','output'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--horizon',type=int,default=24);parser.add_argument('--candidates',type=int,default=4,choices=(1,2,4,8))
    parser.add_argument('--pooling',default='both',choices=('both','mean','last'))
    run(parser.parse_args())
