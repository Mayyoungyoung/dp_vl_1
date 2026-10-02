"""CPU transfer of twelve preselected checkpoints; no optimizer or reselection."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from routeset.common import sha256,write_json
from routeset.legacy_multitask_transfer import (validate_checkpoint_plan,verify_export,
    aggregate,validate_cache_compatibility,audit_model_source)
from routeset.observed_geometry import ObservedGeometryRouteHead
from routeset.observed_route_head import load_observed_dataset
from scripts.train_observed_geometry import load_geometry,evaluate


def inspect_checkpoint(row,project_root=None):
    run=Path(row['run'])
    for name,expected in [('summary.json',row['summary_sha256']),('config.json',row['config_sha256']),
                          (row['checkpoint'],row['checkpoint_sha256'])]:
        if sha256(run/name)!=expected:raise ValueError('Frozen checkpoint/config/summary changed: '+str(run/name))
    config=json.loads((run/'config.json').read_text())
    if (config['code_commit']!=row['training_commit'] or config['seed']!=row['seed']
            or config.get('endpoint_mode')!='free_offset' or config.get('refinement_mode','none')!='none'
            or config['candidates']!=4 or config['horizon']!=24 or config['steps']!=1500
            or config.get('metric_aggregation')!='task_parent'):
        raise ValueError('Original registered architecture/seed/protocol required')
    expected_weight=0. if row['arm']=='ordinary' else .02
    if config['grounding_weight']!=expected_weight:raise ValueError('Incorrect paired arm')
    saved=torch.load(run/row['checkpoint'],map_location='cpu',weights_only=False)
    if saved['config']!=config or saved['step']!=row['step']:
        raise ValueError('Checkpoint state disagrees with frozen metadata')
    project_root=Path(project_root) if project_root is not None else run.parents[2]
    source=audit_model_source(project_root,Path(__file__).resolve().parents[1],config)
    return config,saved,source


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registration',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Fresh transfer output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('','-1'):raise ValueError('CPU-only fixed transfer evaluation')
    torch.set_num_threads(1);registration=json.loads(a.registration.read_text());plan=validate_checkpoint_plan(registration)
    export,gate=verify_export(a.data,registration)
    configs=[];source_audits=[]
    for row in plan:
        config,saved,source=inspect_checkpoint(row);configs.append(config);source_audits.append(source);del saved
    common=('feature_dim','horizon','candidates','width','depth','point_width','endpoint_residual_bound','pooling','pixel_stride','anchor_mode','endpoint_mode')
    if any(any(c[k]!=configs[0][k] for k in common) for c in configs):raise ValueError('Paired model information/architecture differ')
    cache=json.loads((a.data/'qwen_cache/cache_config.json').read_text())
    cache_compatibility=validate_cache_compatibility(cache,configs,sha256(a.data/'observations.jsonl'))
    runtime=dict(torch_threads=torch.get_num_threads(),cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
                 omp_threads=os.environ.get('OMP_NUM_THREADS'),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    started=time.perf_counter();config=configs[0]
    data=load_observed_dataset(a.data/'observations.jsonl',a.data/'supervision.jsonl',a.data/'qwen_cache',config['horizon'],config['pooling'])
    if data['features'].shape[1]!=cache_compatibility['output_dimension']:raise ValueError('Real Qwen output dimension differs')
    geometry=load_geometry(data,a.data/'observations.jsonl',a.data/'supervision.jsonl',config['pixel_stride'])
    ids=np.flatnonzero(data['splits']=='DEV_MODEL')
    if len(ids)!=len(data['scene_ids']) or len(ids)!=export['observations']:raise ValueError('All exported DEV observations required')
    preprocessing=time.perf_counter()-started
    selected=set(data['parent_ids'])
    for config in configs:
        old=[json.loads(line) for line in Path(config['observations']).read_text().splitlines() if line.strip()]
        if selected&{r['parent_id'] for r in old}:raise ValueError('Transfer parent appeared in source training/selection')
    entries=[];paired_rows=[]
    for row in plan:
        config,saved,source=inspect_checkpoint(row)
        model=ObservedGeometryRouteHead(config['feature_dim'],config['horizon'],config['candidates'],config['width'],config['depth'],
            config['point_width'],config['endpoint_residual_bound'],geometry_pooling=config['geometry_pooling'],anchor_mode=config['anchor_mode'],endpoint_mode=config['endpoint_mode']).eval()
        model.load_state_dict(saved['model'],strict=True)
        out=a.output/('seed'+str(row['seed']))/row['arm']/row['checkpoint_kind'];tic=time.perf_counter()
        metrics=evaluate(model,data,geometry,ids,'cpu',out,metric_aggregation='task_parent')
        seconds=time.perf_counter()-tic
        # The historical evaluator computes selection_score for reporting only.
        # There is no checkpoint search/update in this entry point.
        metrics.update(transfer_selection='original best or predeclared fixed1500; no new selection',additional_training_steps=0)
        write_json(out/'metrics.json',metrics)
        provenance=dict(row,evaluation_protocol='observation_eval_v2',registration_sha256=sha256(a.registration),
            export_manifest_sha256=sha256(a.data/'export_manifest.json'),cache_config_sha256=sha256(a.data/'qwen_cache/cache_config.json'),
            evaluation_dataset_fingerprint=geometry['fingerprint'],training_dataset_fingerprint=config['dataset_fingerprint'],
            prediction_sha256=sha256(out/'predictions.npz'),cpu_seconds=seconds,shared_preprocess_seconds=preprocessing,
            source_checkpoint_unchanged=sha256(Path(row['run'])/row['checkpoint'])==row['checkpoint_sha256'],
            runtime=runtime,source_audit=source,cache_compatibility=cache_compatibility,
            evaluator_sha256=sha256(__file__),evaluation_commit=os.environ.get('CODE_COMMIT'),additional_training_steps=0,
            timing_scope='Frozen real-Qwen cache + CPU head; actual cache encoding cost separately recorded')
        if not provenance['source_checkpoint_unchanged']:raise ValueError('Source model modified during read-only evaluation')
        write_json(out/'provenance.json',provenance);entries.append(dict(row,metrics=metrics,provenance=provenance))
        print(json.dumps(dict(arm=row['arm'],seed=row['seed'],checkpoint=row['checkpoint_kind'],ADE=metrics['candidate_matched_ADE_m'],seconds=seconds)),flush=True)
        del saved,model
    verify_export(a.data,registration)
    for kind in ('best','last'):
        for seed in (0,1,2):
            base=a.output/('seed'+str(seed))
            old=json.loads((base/'ordinary'/kind/'per_scene.json').read_text())
            new=json.loads((base/'event_supported'/kind/'per_scene.json').read_text())
            if [r['scene_id'] for r in old]!=[r['scene_id'] for r in new]:raise ValueError('Paired scene identities differ')
            for left,right in zip(old,new):
                delta={key:None if left[key] is None or right[key] is None else right[key]-left[key]
                    for key in ('candidate_matched_ADE_m','candidate_endpoint_error_m','event_state_accuracy','event_sequence_accuracy')}
                paired_rows.append(dict(seed=seed,checkpoint_kind=kind,scene_id=left['scene_id'],parent_id=left['parent_id'],ordinary=left,event_supported=right,aux_minus_ordinary=delta))
    write_json(a.output/'summary.json',dict(protocol='multitask_legacy12_frozen_transfer_v1',status='completed',entries=entries,
        aggregate=aggregate(entries),paired_scenes=paired_rows,mechanical_gate=gate,collection_denominators={key:export[key] for key in
            ('requested_parents','requested_attempts','actual_attempt_records','parents_with_observation','parents_with_zero_reference','successful_reference_routes','observations')},
        additional_training_steps=0,checkpoint_reselection=False,locked_access=False,
        scope='Same six-task unseen-parent transfer; reference/event metrics only, semantic/collision/execution/success null.'))


if __name__=='__main__':main()
