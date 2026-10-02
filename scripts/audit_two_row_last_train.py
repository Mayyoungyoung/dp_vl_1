"""Fixed-last TRAIN fit diagnostic; no optimization, reselection or repair."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def select_train(data,train_parents=16):
    if train_parents not in (16,32,64):raise ValueError('Only registered TRAIN prefix sizes accepted')
    expected={'two_row_reach_%d_target%d'%(parent,target) for parent in range(283200,283200+train_parents) for target in range(3)}
    ids=np.flatnonzero(data['splits']=='TRAIN')
    actual=list(map(str,data['scene_ids'][ids]))
    if not actual or len(actual)!=len(set(actual)) or not set(actual)<=expected:
        raise ValueError('Unique observed conditions from the registered TRAIN prefix required')
    return ids


def summarize_residual(prediction,matched):
    distance=np.linalg.norm(np.asarray(prediction)-np.asarray(matched),axis=-1)
    return dict(mean_vertex_distance_m=float(distance.mean()),max_vertex_distance_m=float(distance.max()),
        endpoint_distance_m=float(distance[-1]),per_vertex_distance_m=distance.tolist())


def run(run_path,data_path,output):
    import torch
    from routeset.common import sha256,write_json
    from routeset.observed_geometry import ObservedGeometryRouteHead
    from routeset.observed_route_head import load_observed_dataset
    from scripts.train_observed_geometry import load_geometry
    from scripts.train_observed_two_row import evaluate
    from scripts.evaluate_observed_two_row_online import head_options
    from scripts.export_two_row_observations import verify_export
    from scripts.diagnose_observed_anchor_gradients import assigned_targets
    from routeset.train_v2 import positive_assignment_loss
    torch.set_num_threads(1);started=time.perf_counter()
    run_path,data_path,output=map(Path,(run_path,data_path,output))
    if output.exists():raise FileExistsError('New diagnostic output required')
    manifest,gate=verify_export(data_path)
    config=json.loads((run_path/'config.json').read_text());summary=json.loads((run_path/'summary.json').read_text())
    if Path(config['observations']).resolve()!=(data_path/'observations.jsonl').resolve():raise ValueError('Original data required')
    hashes={str(run_path/name):sha256(run_path/name) for name in ('config.json','summary.json','last.pt','train/predictions.npz')}
    if hashes[str(run_path/'last.pt')]!=summary['last_checkpoint_sha256'] or summary['last_step']!=1500:
        raise ValueError('Original last1500 checkpoint required')
    saved=torch.load(run_path/'last.pt',map_location='cpu',weights_only=False)
    if saved['step']!=1500 or saved['config']['dataset_fingerprint']!=config['dataset_fingerprint']:
        raise ValueError('Checkpoint identity changed')
    model=ObservedGeometryRouteHead(**head_options(config)).eval();model.load_state_dict(saved['model'],strict=True)
    versions=tuple(p._version for p in model.parameters())
    data=load_observed_dataset(config['observations'],config['supervision'],config['cache_dir'],config['horizon'],config['pooling'])
    geometry=load_geometry(data,config['observations'],config['supervision'],config['pixel_stride'])
    if geometry['fingerprint']!=config['dataset_fingerprint']:raise ValueError('Dataset fingerprint changed')
    ids=select_train(data,manifest['selection']['requested_parents']['TRAIN'])
    observed_train={r['id'] for r in map(json.loads,(data_path/'observations.jsonl').read_text().splitlines()) if r['split']=='TRAIN'}
    if set(map(str,data['scene_ids'][ids]))!=observed_train:raise ValueError('Loader omitted an actual TRAIN observation')
    positive_rows=np.flatnonzero(data['path_mask'][ids].any(1));positive_ids=ids[positive_rows]
    if not len(positive_rows):raise ValueError('Original trained model requires known positive TRAIN references')
    # Existing loader opens the original TRAIN/DEV development export. Only the
    # registered TRAIN rows receive new forwards; labels are never forward inputs.
    metrics=evaluate(model,data,geometry,ids,'cpu',output/'last_train',
        evaluation_sources=(config['observations'],config['supervision']),selection_metric='tip_unique_valid')
    stages={}
    label_rows={r['id']:r for r in map(json.loads,(data_path/'supervision.jsonl').read_text().splitlines())}
    for stage,path in [('best_train',run_path/'train/predictions.npz'),('last_train',output/'last_train/predictions.npz')]:
        with np.load(path,allow_pickle=False) as a:
            if list(a['scene_ids'])!=list(data['scene_ids'][ids]):raise ValueError('TRAIN prediction row order changed')
            xyz,opened=a['paths'],a['gripper_open']
        pred=torch.tensor(np.concatenate((xyz[positive_rows,:,1:],opened[positive_rows,:,1:,None]*config['event_scale']),axis=-1))
        targets=torch.tensor(np.concatenate((data['paths'][positive_ids,:,1:],data['events'][positive_ids,:,1:,None]*config['event_scale']),axis=-1))
        matched,indices=assigned_targets(pred,targets,data['path_mask'][positive_ids])
        loss=positive_assignment_loss(pred,targets,data['path_mask'][positive_ids],'saturation',np.random.default_rng(0))
        if not torch.allclose((pred-matched).square().mean(),loss,rtol=1e-5,atol=1e-7):raise ValueError('Original loss reconstruction failed')
        records=[]
        for row,idx in enumerate(positive_ids):
            identifier=str(data['scene_ids'][idx]);references=label_rows[identifier]['routes']
            for k in range(4):
                ref=indices[row][k]
                records.append(dict(id=identifier,candidate=k,matched_reference=references[ref],
                    matched_reference_sha256=sha256(references[ref]),
                    **summarize_residual(xyz[positive_rows[row],k],data['paths'][idx,ref])))
        stages[stage]=dict(original_saturation_loss=float(loss),prediction_sha256=sha256(path),per_candidate=records,
            matched_positive_conditions=len(positive_ids),unreferenced_conditions=len(ids)-len(positive_ids))
    if versions!=tuple(p._version for p in model.parameters()):raise ValueError('Model updated')
    if any(sha256(path)!=value for path,value in hashes.items()):raise ValueError('Original run changed')
    _,final_gate=verify_export(data_path)
    report=dict(protocol='two_row_last1500_registered_train_fit_v2',checkpoint_sha256=hashes[str(run_path/'last.pt')],
        registered_train_parents=manifest['selection']['requested_parents']['TRAIN'],
        requested_train_inputs=3*manifest['selection']['requested_parents']['TRAIN'],
        unavailable_train_inputs=3*manifest['selection']['requested_parents']['TRAIN']-len(ids),
        actual_unreferenced_train_inputs=len(ids)-len(positive_ids),
        fixed_last_step=1500,metrics=metrics,stages=stages,new_forward_requests=len(ids),new_complete_path_states=4*len(ids),
        cached_qwen=True,new_qwen_encodings=0,new_dev_predictions=0,optimizer_updates=0,
        source_hashes=hashes,script_sha256=sha256(__file__),initial_gate=gate,final_gate=final_gate,
        export_manifest_sha256=sha256(data_path/'export_manifest.json'),elapsed_seconds=time.perf_counter()-started,
        scope='TRAIN fit diagnosis of fixed best and fixed last; incomplete known positives, no collision repair or model selection; no robot validity claim.')
    write_json(output/'report.json',report)
    write_json(output/'artifact_index.json',{p.relative_to(output).as_posix():dict(sha256=sha256(p),bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file()})
    print(json.dumps({k:metrics[k] for k in ('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','semantic_goal_accuracy','TipClearAtK')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','data','output'):parser.add_argument('--'+name,required=True)
    a=parser.parse_args();run(a.run,a.data,a.output)
