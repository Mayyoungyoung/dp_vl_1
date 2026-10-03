"""Ordinary frozen-Qwen/RGB-D saturation baseline for the fixed two-row export.

The existing optimizer/model/resume loop is reused under an explicit scoped
evaluation adapter. Its standalone defaults and historical behavior are not
edited. Two-row labels are loaded only after all prediction calls complete.
"""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from routeset.common import sha256,write_json
from scripts import train_observed_geometry as base
from scripts.export_two_row_observations import verify_export,selection_guard
from scripts.evaluate_observed_two_row import scene_metrics,PROTOCOL
from scripts.train_observed_routes import observation_metrics,paired_language_indices

DRIVER_PROTOCOL='ordinary_two_row_frozen_qwen_saturation_v1'
TIP_FIELDS=('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','UnknownTypeTipValidCount',
    'DuplicateClassifiedTipValidCount','KnownReferenceTypeCoverageAtK','TipClearAtK','StartCorrectAtK','EventSequenceCorrectAtK')


def reused_language_control(predictions,events,data,geometry,ids):
    switched=paired_language_indices(data,ids);eligible=np.flatnonzero(switched>=0)
    if not len(eligible):return None,None
    positions={int(idx):i for i,idx in enumerate(ids)}
    source_positions=[]
    for row in eligible:
        original,other=int(ids[row]),int(switched[row])
        if (other not in positions or geometry['index'][original]!=geometry['index'][other]
                or not np.array_equal(data['current'][original],data['current'][other])
                or data['image_hashes'][original]!=data['image_hashes'][other]):
            raise ValueError('Language reuse requires identical current observation and geometry')
        source_positions.append(positions[other])
    swapped,opened=predictions[source_positions],events[source_positions]
    wrong,_=observation_metrics(swapped,opened,data,ids[eligible])
    own,_=observation_metrics(swapped,opened,data,switched[eligible])
    metrics=dict(examples=len(eligible),additional_forward_requests=0,additional_complete_path_states=0,
        same_image_other_target_language_original_goal_accuracy=wrong['semantic_goal_accuracy'],
        same_image_other_target_language_new_goal_accuracy=own['semantic_goal_accuracy'],
        mean_endpoint_response_m=float(np.linalg.norm(swapped[:,:,-1]-predictions[eligible,:,-1],axis=-1).mean()),
        explanation='Reindex already-generated alternate-target predictions with identical image/current/geometry; no new candidates or forward calls.')
    arrays=dict(paths=swapped,gripper_open=opened,original_scene_ids=data['scene_ids'][ids[eligible]],
        language_source_ids=data['scene_ids'][switched[eligible]])
    return metrics,arrays


def add_two_row_metrics(result,rows,predictions,events,data,ids,evaluation_sources):
    observations,supervision=map(Path,evaluation_sources)
    if observations.parent!=supervision.parent:raise ValueError('One closed export required')
    manifest,_=verify_export(observations.parent)
    selected={str(data['scene_ids'][idx]) for idx in ids}
    labels={r['id']:r for r in [json.loads(x) for x in supervision.read_text().splitlines()] if r['id'] in selected}
    if set(labels)!=selected:raise ValueError('Complete evaluation labels required')
    summaries=[];hashes={}
    def checked(path):
        path=Path(path);value=sha256(path)
        if manifest['source_files_sha256'].get(str(path))!=value:raise ValueError('Evaluation label source changed')
        hashes[str(path)]=value;return path
    for position,idx in enumerate(ids):
        label=labels[str(data['scene_ids'][idx])]
        if label['parent_id']!=str(data['parent_ids'][idx]) or label['split']!=str(data['splits'][idx]):raise ValueError('Evaluation identity changed')
        with np.load(checked(label['verification_only']),allow_pickle=False) as archive:
            geometry={key:archive[key] for key in ('obstacle_centers','obstacle_halfsizes')}
        config=json.loads(checked(label['route_config']).read_text())
        if sha256(label['route_config'])!=label['route_config_sha256']:raise ValueError('Registered route config hash changed')
        summary,candidates=scene_metrics(predictions[position],events[position],
            dict(gripper_pose=data['current'][idx,:7],gripper_open=data['current'][idx,7]),geometry,
            label['semantic_targets'],label['route_types'],config)
        if summary['semantic_goal_accuracy']!=rows[position]['semantic_goal_accuracy']:raise ValueError('Semantic protocol differs')
        summaries.append(summary);rows[position].update(tip_evaluation=summary,tip_candidates=candidates)
    for key in TIP_FIELDS:
        values=[r[key] for r in summaries if r[key] is not None];result[key]=float(np.mean(values)) if values else None
    result.update(tip_evaluation_protocol=PROTOCOL,tip_evaluation_examples=len(ids),tip_geometry_label_source_sha256=hashes,
        geometry_validation='Original two-row tip-only2cm/goal3cm checks; no arm/IK/execution certificate',
        evaluation_geometry_use='after all predictions only; never a model input, loss, repair or extra candidate')


@torch.no_grad()
def evaluate(model,data,geometry,ids,device,output=None,batch_size=8,selection_metric='reference_ADE',
             evaluation_sources=None,metric_aggregation='instruction'):
    if metric_aggregation!='instruction':raise ValueError('Predeclared instruction mean; three recorded targets per complete parent')
    if evaluation_sources is None:raise ValueError('Explicit two-row exported labels required')
    model.eval();paths=[];opened=[];anchors=[]
    for start in range(0,len(ids),batch_size):
        xyz,event,details=model(**base.batch_inputs(data,geometry,ids[start:start+batch_size],device))
        paths.append(xyz.cpu().numpy());opened.append(event.cpu().numpy());anchors.append(details['anchor_xyz'].cpu().numpy())
    paths,opened,anchors=map(np.concatenate,(paths,opened,anchors))
    metrics,rows=observation_metrics(paths,opened,data,ids)
    control,control_arrays=reused_language_control(paths,opened,data,geometry,ids)
    metrics['paired_language_control']=control
    add_two_row_metrics(metrics,rows,paths,opened,data,ids,evaluation_sources)
    metrics.update(selection_score=base.checkpoint_selection_score(metrics,selection_metric),
        checkpoint_selection_protocol='dev_two_row_tip_unique_valid_v1' if selection_metric=='tip_unique_valid' else 'reference_ADE_v1',
        checkpoint_selection_criterion='UniqueClassifiedTipValidAtK + .05 * TipValidAtK' if selection_metric=='tip_unique_valid' else 'negative candidate_matched_ADE_m',
        generation_budget=dict(final_candidates=int(paths.shape[1]),draft_complete_paths=0,updates=0,
            total_complete_path_states=int(paths.shape[1]),candidate_filtering=False,evaluated_requests=len(ids),
            evaluated_complete_path_states=int(len(ids)*paths.shape[1]),language_control_additional_requests=0))
    if output is not None:
        output=Path(output);output.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output/'predictions.npz',paths=paths,gripper_open=opened,learned_surface_anchor=anchors,
            scene_ids=data['scene_ids'][ids],parent_ids=data['parent_ids'][ids])
        if control_arrays is not None:np.savez_compressed(output/'paired_language_predictions.npz',**control_arrays)
        write_json(output/'metrics.json',metrics);write_json(output/'per_scene.json',rows)
    return metrics


@contextmanager
def evaluation_adapter(sources):
    original=base.evaluate
    def adapted(*args,**kwargs):
        if kwargs.get('evaluation_sources') is None:kwargs['evaluation_sources']=sources
        return evaluate(*args,**kwargs)
    base.evaluate=adapted
    try:yield
    finally:base.evaluate=original


def validate_adapter_resume(current,saved):
    for key in ('two_row_driver_protocol','two_row_driver_sha256','two_row_export_sha256','two_row_quality_audit_sha256',
                'two_row_selection_sha256','two_row_tip_protocol'):
        if current.get(key)!=saved.get(key):raise ValueError('Two-row resume adapter mismatch: '+key)


def prediction_artifact_index(output):
    """Bind every saved evaluation pool, including TRAIN, to its actual bytes."""
    output=Path(output);index={}
    for split in ('train','dev_model','last_dev_model'):
        folder=output/split
        metrics=json.loads((folder/'metrics.json').read_text())
        files=['predictions.npz','per_scene.json','metrics.json']
        if metrics.get('paired_language_control') is not None:files.append('paired_language_predictions.npz')
        for name in files:
            path=folder/name
            index[split+'/'+name]=dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size)
    return index


def train(args):
    data=Path(args.data);selection=json.loads(Path(args.selection).read_text());selection_guard(selection)
    manifest,gate=verify_export(data)
    if manifest['selection']!=selection:raise ValueError('Registered export selection changed')
    quality=json.loads(Path(args.quality_audit).read_text())
    if (quality.get('protocol')!='two_row_train_endpoint_capacity_and_reference_quality_v1'
            or not quality['capacity_gate_passed'] or quality['dev_raw_arrays_opened'] or quality['locked_raw_opened']):
        raise ValueError('Predeclared TRAIN-only surface endpoint capacity gate must pass')
    train_labels=[r for r in [json.loads(x) for x in (data/'supervision.jsonl').read_text().splitlines()] if r['split']=='TRAIN']
    if (quality.get('source_export_manifest_sha256')!=sha256(data/'export_manifest.json')
            or quality.get('train_input_ids')!=sorted(r['id'] for r in train_labels)
            or quality.get('train_reference_counts')!={r['id']:len(r['routes']) for r in train_labels}):
        raise ValueError('Capacity audit must cover every registered TRAIN input and positive reference')
    for path,value in quality['source_files_sha256'].items():
        if manifest['source_files_sha256'].get(path)!=value or sha256(path)!=value:raise ValueError('TRAIN quality receipt is from different data')
    cfg=selection['ordinary_training']
    expected=dict(steps=1500,batch_size=32,candidates=4,horizon=24,seed=0,eval_every=250,lr=.0003,
        width=128,depth=2,point_width=64,pooling='both',pixel_stride=2,event_scale=.2,
        anchor_mode='straight_through_peak',endpoint_mode='surface_anchor',grounding_weight=.02,grounding_sigma=.025,
        endpoint_residual_bound=.05,refinement_mode='none',sampling_mode='uniform',objective='saturation',
        grounding_target='endpoint',sample_stream_audit=True,selection='UniqueClassifiedTipValidAtK + .05 * TipValidAtK',
        tip_evaluation_protocol=PROTOCOL,save_best_and_last=True)
    if cfg!=expected:raise ValueError('Predeclared ordinary baseline changed')
    options={k:v for k,v in cfg.items() if k not in ('selection','tip_evaluation_protocol','save_best_and_last')}
    options.update(observations=str(data/'observations.jsonl'),supervision=str(data/'supervision.jsonl'),cache_dir=str(data/'qwen_cache'),
        output=args.output,checkpoint_selection='tip_unique_valid',geometry_pooling='spatial',metric_aggregation='instruction',
        multitask_snapshot_manifest=None,refinement_sigma=None,refinement_prefix_fraction=None,refinement_bound=None,
        resume=args.resume,stop_after=args.stop_after,device=args.device,threads=1,
        two_row_driver_protocol=DRIVER_PROTOCOL,two_row_driver_sha256=sha256(__file__),
        two_row_export_sha256=sha256(data/'export_manifest.json'),two_row_quality_audit_sha256=sha256(args.quality_audit),
        two_row_selection_sha256=sha256(args.selection),two_row_tip_protocol=PROTOCOL,
        model_use_gate_source='scripts.export_two_row_observations.verify_export',
        posttrain_latency_diagnostic_requests=24,posttrain_latency_diagnostic_complete_path_states=96)
    if args.resume:
        saved=torch.load(Path(args.output)/'last.pt',map_location='cpu',weights_only=False)
        validate_adapter_resume(options,saved['config']);del saved
    with evaluation_adapter((options['observations'],options['supervision'])):
        base.train(SimpleNamespace(**options))
    _,final_gate=verify_export(data)
    summary=Path(args.output)/'summary.json'
    if summary.exists():
        result=json.loads(summary.read_text());history=json.loads((Path(args.output)/'history.json').read_text())
        requests=sum(r['dev_model']['examples'] for r in history)+sum(result[key]['examples'] for key in ('metrics','last_metrics','train_metrics'))
        write_json(Path(args.output)/'two_row_driver_receipt.json',dict(protocol=DRIVER_PROTOCOL,
            source_sha256=sha256(__file__),export_sha256=options['two_row_export_sha256'],quality_audit_sha256=options['two_row_quality_audit_sha256'],
            final_live_gate=final_gate,main_evaluation_condition_requests=requests,main_evaluation_complete_path_states=requests*4,
            language_control_additional_requests=0,posttrain_latency_diagnostic_requests=24,
            posttrain_latency_diagnostic_complete_path_states=96,optimizer_resume_loop='unchanged train_observed_geometry.train',
            actual_training_summary_sha256=sha256(summary),prediction_artifacts=prediction_artifact_index(args.output),
            training_candidate_path_states=result['trajectory_exposures']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data','selection','quality-audit','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    train(p.parse_args())
