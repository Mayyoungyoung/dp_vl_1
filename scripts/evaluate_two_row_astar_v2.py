"""Fixed TRAIN-fitted observation A* v2 control for a closed registered export.

No neural model, planning threshold change, ground-truth inference input,
replacement candidate, or checkpoint selection. Four submitted slots include
localization/search failures and duplicates. Labels are used after each pool
has been saved and hashed; fitting sees only TRAIN positive-reference fields.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np
from PIL import Image

from scripts.export_two_row_observations import verify_export,digest,write_json,INPUT_KEYS,CURRENT_KEYS
from scripts.evaluate_observed_two_row import scene_metrics,PROTOCOL as TIP_PROTOCOL

PROTOCOL='closed_two_row_observation_astar_v2_control_v1'
FROZEN_SOURCES={
    'observation_multiroute_astar_v2.py':'197366fb638a102cc4ed04f84df407e3f805e841523c4dc5a3a3e29c521839cc',
    'observation_multiroute_astar.py':'cf487a91b5825c50ace087f5ebd29c1a7ca2b91607ccc8ff98af0e6a21991c93',
    'observation_prototype_grounding.py':'2f3046ea2ebeb9c5f3b27921a8fa7f19655a91cbc7621e36321c27a3dbaaf0c2'}
FROZEN_CRLF_SOURCES={
    'observation_multiroute_astar_v2.py':'d0166218effb7f576318052ab79cefaa9b8a418b22f0b7d907a54ac5fde82edc',
    'observation_multiroute_astar.py':'95d71e2f708e1e2c534b297f81f555980e685081ebf6d7cdb228d682dc37cd39',
    'observation_prototype_grounding.py':'fdeafdbc22f2e0351477e11ea597400d3037d5571369c150a1c70879a1f0a565'}
METRICS=('TipValidAtK','AnyTipValidAtK','UniqueClassifiedTipValidAtK','UnknownTypeTipValidCount',
    'DuplicateClassifiedTipValidCount','KnownReferenceTypeCoverageAtK','semantic_goal_accuracy','AnySemanticGoalAtK',
    'TipClearAtK','StartCorrectAtK','EventSequenceCorrectAtK','endpoint_error_m','mean_path_length_m')


def verify_frozen_source(path):
    """Accept only the two audited exact-byte forms, then check LF content too."""
    path=Path(path);name=path.name
    if name not in FROZEN_SOURCES:raise ValueError('Unregistered planner source: '+name)
    raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
    canonical=hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()
    if actual not in (FROZEN_SOURCES[name],FROZEN_CRLF_SOURCES[name]) or canonical!=FROZEN_SOURCES[name]:
        raise ValueError('Existing planner source changed beyond the two exact LF/CRLF forms: '+name)
    return dict(actual_sha256=actual,canonical_lf_sha256=canonical,
        exact_byte_form='LF' if actual==FROZEN_SOURCES[name] else 'CRLF',
        accepted_exact_sha256=[FROZEN_SOURCES[name],FROZEN_CRLF_SOURCES[name]],
        policy='Only registered uniform LF/CRLF forms; no whitespace, mixed-newline or code normalization')


def dependencies():
    # Verify bytes before importing the reused planner. Lazy imports keep
    # schema/budget unit tests independent of SciPy.
    for name in FROZEN_SOURCES:verify_frozen_source(Path(__file__).with_name(name))
    from scripts import observation_multiroute_astar_v2 as planner
    if (planner.CONFIG['candidates'],planner.CONFIG['horizon'],planner.CONFIG['maximum_expanded_nodes_per_candidate'],
            planner.CONFIG['search_deadline_seconds'])!=(4,24,20000,2.):raise ValueError('Frozen K4 search limits changed')
    return planner


def checked(path,manifest,hashes):
    path=Path(path);value=digest(path)
    if manifest['source_files_sha256'].get(str(path))!=value:raise ValueError('Closed source changed: '+str(path))
    hashes[str(path)]=value;return path


def label_for(data,row):
    # TRAIN callers return before any subsequent DEV label rows are needed.
    # DEV calls are made only after this request's output pool has been sealed.
    for line in (Path(data)/'supervision.jsonl').read_text().splitlines():
        label=json.loads(line)
        if label['id']==row['id']:
            if label['parent_id']!=row['parent_id'] or label['split']!=row['split']:raise ValueError('Label identity changed')
            return label
    raise ValueError('Registered input lacks its preserved supervision row')


def fit_train(data,observations,manifest,planner):
    started=time.perf_counter();train=[r for r in observations if r['split']=='TRAIN'];labels={};hashes={}
    if not train:raise ValueError('TRAIN observation population required')
    for row in train:
        label=label_for(data,row)
        # Strip semantic targets, types, guide/config fields before both fitters.
        pointer=str(Path(row['image']).with_name('observation.npz'))
        if label['observation']!=pointer:raise ValueError('Current observation pointer changed')
        labels[row['id']]=dict(observation=pointer,routes=list(label['routes']))
        for path in [row['image'],pointer]+label['routes']:checked(path,manifest,hashes)
    model,parents=planner.prototype.fit_prototypes(Path(data),train,labels)
    prior=planner.v1.fit_workspace(Path(data),train,labels)
    for fitted in (model['training_source_sha256'],prior['source_sha256']):
        if any(hashes.get(path)!=value for path,value in fitted.items()):raise ValueError('Fitter read outside exact TRAIN source set')
    return model,prior,dict(elapsed_seconds=time.perf_counter()-started,parents=sorted(parents),
        observed_train_inputs=len(train),positive_reference_inputs=sum(bool(r['routes']) for r in labels.values()),
        positive_reference_occurrences=sum(len(r['routes']) for r in labels.values()),source_files_sha256=hashes,
        semantic_targets_or_route_types_passed_to_fitters=False)


def load_current(row,manifest,hashes,planner):
    if set(row)!=INPUT_KEYS or row['split']!='DEV_MODEL':raise ValueError('Strict DEV observation input required')
    start=time.perf_counter();image=checked(row['image'],manifest,hashes)
    pointer=checked(image.with_name('observation.npz'),manifest,hashes)
    rgb=np.asarray(Image.open(image).convert('RGB'))
    with np.load(pointer,allow_pickle=False) as archive:
        if set(archive.files)!=CURRENT_KEYS:raise ValueError('Observation contains undeclared fields')
        current={key:archive[key].copy() for key in archive.files}
    io_seconds=time.perf_counter()-start;start=time.perf_counter()
    color,xyz,valid=planner.prototype.observed_grid(rgb,current['depth'],current['camera_intrinsics'],current['camera_extrinsics'])
    state=np.r_[current['gripper_pose'],np.asarray(current['gripper_open']).reshape(1)]
    return (color,xyz,valid,current['depth'],current['camera_intrinsics'],current['camera_extrinsics'],state),current,dict(
        observation_io_hash_seconds=io_seconds,backprojection_seconds=time.perf_counter()-start)


def validate_pool(paths,raw,record):
    if (np.asarray(paths).shape!=(4,24,3) or len(raw)!=4 or record.get('submitted_candidate_budget')!=4
            or len(record.get('attempts',[]))!=4 or [r['slot'] for r in record['attempts']]!=list(range(4))):
        raise ValueError('Exactly four bounded submitted slots required; excess candidates cannot be hidden')
    count=0
    for slot,path in enumerate(raw):
        emitted=record['attempts'][slot].get('complete_raw_paths_emitted')
        if path is None:
            if emitted!=0 or not np.isnan(paths[slot]).all():raise ValueError('Failed slot must remain NaN')
        else:
            if emitted!=1 or np.asarray(path).ndim!=2 or np.asarray(path).shape[1]!=3:raise ValueError('One raw complete path per successful slot required')
            count+=1
    if record.get('raw_complete_paths')!=count or record.get('failed_slots')!=4-count:raise ValueError('Candidate accounting differs')


def one_request(data,row,manifest,model,prior,planner,output):
    start=time.perf_counter();hashes={};output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write_json(output/'status.json',dict(status='running',id=row['id'],submitted_slot_limit=4,
        scope='Four slots reserved for this request; actual stage/outputs determine completed search count.'))
    inputs,current,timing=load_current(row,manifest,hashes,planner)
    began=time.perf_counter()
    paths,raw,record=planner.plan_observation(*inputs,row['instruction'],model,prior)
    validate_pool(paths,raw,record)
    timing['localization_grid_search_and_original_proxy_seconds']=time.perf_counter()-began
    opened=np.full((4,24),np.asarray(current['gripper_open']).item())
    began=time.perf_counter()
    prediction=output/'predictions.npz'
    np.savez_compressed(prediction,paths=paths[None],gripper_open=opened[None],scene_ids=[row['id']],parent_ids=[row['parent_id']])
    raw_file=output/'raw_paths.npz'
    np.savez_compressed(raw_file,**{'candidate_%d'%i:np.empty((0,3)) if path is None else path for i,path in enumerate(raw)})
    prediction_hash=digest(prediction);raw_hash=digest(raw_file)
    seal=dict(id=row['id'],prediction_sha256=prediction_hash,raw_paths_sha256=raw_hash,
        submitted_candidate_budget=4,raw_complete_paths=record['raw_complete_paths'],evaluation_labels_opened=False)
    write_json(output/'generation_seal.json',seal)
    timing['output_sealing_seconds']=time.perf_counter()-began
    timing['observation_to_sealed_pool_seconds']=time.perf_counter()-start
    began=time.perf_counter()
    # No evaluation labels are passed to the planner, even for failed slots.
    label=label_for(data,row)
    with np.load(checked(label['verification_only'],manifest,hashes),allow_pickle=False) as archive:
        geometry={key:archive[key].copy() for key in ('obstacle_centers','obstacle_halfsizes')}
    route_config=json.loads(checked(label['route_config'],manifest,hashes).read_text())
    metric,candidates=scene_metrics(paths,opened,current,geometry,label['semantic_targets'],label['route_types'],route_config)
    timing['label_io_and_two_row_check_seconds']=time.perf_counter()-began
    timing['observation_to_checked_pool_seconds']=time.perf_counter()-start
    result=dict(id=row['id'],parent_id=row['parent_id'],split='DEV_MODEL',metrics=metric,candidates=candidates,
        reference_count=len(label['routes']),known_reference_count=sum(t is not None for t in label['route_types']),
        generation=record,timing=timing,prediction_sha256=prediction_hash,raw_paths_sha256=raw_hash,
        source_files_sha256=hashes,scored_after_sealed_pool=True,no_output_filtering_or_repair=True)
    write_json(output/'result.json',result)
    write_json(output/'status.json',dict(status='completed',id=row['id'],submitted_candidate_budget=4,
        raw_complete_paths=record['raw_complete_paths'],failed_slots=record['failed_slots']))
    return result,paths,opened


def aggregate(records):
    if not records:raise ValueError('No observed DEV inputs')
    if len({r['id'] for r in records})!=len(records) or len(records)>36:raise ValueError('Fixed DEV input population changed')
    metrics={key:float(np.mean([r['metrics'][key] for r in records if r['metrics'][key] is not None]))
        if any(r['metrics'][key] is not None for r in records) else None for key in METRICS}
    latencies=np.array([r['timing']['observation_to_checked_pool_seconds'] for r in records])
    metrics.update(examples=len(records),parents=len({r['parent_id'] for r in records}),requested_dev_parents=12,
        requested_dev_inputs=36,unobserved_requested_dev_inputs=36-len(records),
        candidates=4,submitted_candidate_budget=4*len(records),requested_candidate_slots=144,
        unattempted_missing_input_slots=4*(36-len(records)),
        raw_complete_paths=sum(r['generation']['raw_complete_paths'] for r in records),
        failed_candidate_slots=sum(r['generation']['failed_slots'] for r in records),
        exact_grid_duplicate_slots=sum(a.get('exact_grid_path_duplicate',False) for r in records for a in r['generation']['attempts']),
        unsupported_instruction_conditions=sum(r['generation']['localization']['status']=='unsupported_exact_instruction' for r in records),
        reference_evaluation_examples=sum(r['reference_count']>0 for r in records),
        examples_without_reference=sum(r['reference_count']==0 for r in records),
        known_reference_coverage_evaluation_examples=sum(r['metrics']['known_reference_types']>0 for r in records),
        semantic_evaluation_examples=len(records),evaluation_protocol=TIP_PROTOCOL,
        request_latency_seconds=dict(first=float(latencies[0]),median=float(np.median(latencies)),p95=float(np.quantile(latencies,.95)),total=float(latencies.sum()),
            scope='Per-request image/current IO+hashes, backprojection, localization, grid, four bounded searches, original proxy checks, output sealing, label IO and unchanged two-row checking; no Qwen or robot execution'),
        reference_ADE_m=None,SelectedTipValidAtK=None,full_robot_validity=None,
        missing_input_policy='No fabricated input/prediction; missing requested conditions remain separate from failed generated slots.')
    return metrics


def run(data,output):
    data,output=Path(data),Path(output)
    if output.exists():raise FileExistsError('Fresh traditional control output required')
    started=time.perf_counter();manifest,gate=verify_export(data);planner=dependencies()
    observations=[json.loads(line) for line in (data/'observations.jsonl').read_text().splitlines()]
    if any(set(row)!=INPUT_KEYS or row['split'] not in ('TRAIN','DEV_MODEL') for row in observations):raise ValueError('TRAIN/DEV-only input whitelist required')
    if len({row['id'] for row in observations})!=len(observations):raise ValueError('Duplicate input IDs')
    dev=sorted((row for row in observations if row['split']=='DEV_MODEL'),key=lambda row:row['id'])
    if not dev or len(dev)>36:raise ValueError('All actually recorded registered DEV inputs required')
    model,prior,fit=fit_train(data,observations,manifest,planner)
    if set(fit['parents'])&{r['parent_id'] for r in dev}:raise ValueError('TRAIN/DEV parent overlap')
    output.mkdir(parents=True)
    write_json(output/'fitted_train_model.json',dict(prototype=model,workspace=prior,fit=fit))
    planner_source_audit={name:verify_frozen_source(Path(__file__).with_name(name)) for name in FROZEN_SOURCES}
    source={str(Path(__file__).with_name(name)):receipt['actual_sha256'] for name,receipt in planner_source_audit.items()}
    source.update({str(Path(__file__).with_name(name)):digest(Path(__file__).with_name(name)) for name in (
        'evaluate_observed_two_row.py','evaluate_observed_obstacles.py','export_two_row_observations.py',
        'collect_observed_two_row_pilot.py','collect_obstacle_reach.py')})
    source.update(fit['source_files_sha256'])
    source.update({str(data/'export_manifest.json'):digest(data/'export_manifest.json'),str(Path(__file__)):digest(__file__)})
    records=[];paths=[];events=[]
    for row in dev:
        request_output=output/'requests'/row['id']
        try:
            result,xyz,opened=one_request(data,row,manifest,model,prior,planner,request_output)
        except Exception:
            # Unexpected implementation/resource faults abort without invented
            # completed searches. Existing emitted pools and traceback remain.
            request_output.mkdir(parents=True,exist_ok=True)
            write_json(request_output/'failure.json',dict(id=row['id'],status='failed',traceback=traceback.format_exc(),
                automatic_retries=0,completed_prior_requests=len(records),later_requests_unattempted=len(dev)-len(records)-1))
            raise
        records.append(result);paths.append(xyz);events.append(opened);source.update(result['source_files_sha256'])
        write_json(output/'progress.json',dict(completed_inputs=len(records),expected_actual_inputs=len(dev),last_id=row['id']))
        print(json.dumps(dict(id=row['id'],failed_slots=result['generation']['failed_slots'],request_seconds=result['timing']['observation_to_checked_pool_seconds'])),flush=True)
    prediction=output/'predictions.npz'
    np.savez_compressed(prediction,paths=np.stack(paths),gripper_open=np.stack(events),scene_ids=[r['id'] for r in dev],parent_ids=[r['parent_id'] for r in dev])
    metric=aggregate(records)
    _,final_gate=verify_export(data)
    if any(digest(path)!=value for path,value in source.items()):raise ValueError('Source changed during fixed control')
    report=dict(protocol=PROTOCOL,baseline='TRAIN prototype plus original observed grid A* v2',metrics=metric,
        source_export_manifest_sha256=digest(data/'export_manifest.json'),
        requested_total_inputs=manifest['selection']['requested_inputs'],
        actual_total_inputs=manifest['actual_inputs'],
        requested_total_parents=sum(manifest['selection']['requested_parents'].values()),
        registered_train_parents=manifest['selection']['requested_parents']['TRAIN'],
        planner_config=planner.CONFIG,prototype_config=planner.prototype.CONFIG,fit=fit,
        frozen_planner_source_integrity=planner_source_audit,
        fitting_seconds=fit['elapsed_seconds'],elapsed_seconds=time.perf_counter()-started,
        cpu_threads={key:os.environ.get(key) for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        prediction_sha256=digest(prediction),source_files_sha256=source,initial_gate=gate,final_gate=final_gate,
        per_scene=records,search_status_counts=dict(Counter(a['status'] for r in records for a in r['generation']['attempts'])),
        candidate_budget_interpretation='Four raw searches at most and four submitted H24 slots per actual request; raw/resampled representations and all proxy checks recorded, no extra proposal or post-check replacement.',
        comparison='Same registered TRAIN positive pool and K4; closed exact-language color control, not same backbone/information processing or fixed-time superiority.',
        limitations='Finite visible-point/depth free-space proxy with original contact allowances; unseen instructions fail. Tip checks after output do not certify arm, IK, execution, hidden geometry or open-vocabulary ability.')
    write_json(output/'report.json',report)
    write_json(output/'artifact_index.json',{p.relative_to(output).as_posix():dict(sha256=digest(p),bytes=p.stat().st_size) for p in output.rglob('*') if p.is_file()})
    print(json.dumps(metric));return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.data,args.output)
