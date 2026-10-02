"""Compare sealed native pools and historical Python edge pools, without search.

Paths, all submitted slots and deterministic search counters are compared by
observation ID/slot. Differences are retained, never filtered or retried. No
collection data, geometry labels, compiler, model or planner is imported.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from scripts import analyze_observed_spatial_penalty_astar as saved

PROTOCOL = 'native_astar_sealed_pool_differential_v1'
NATIVE_PROTOCOL = 'observed_two_row_native_paired_runner_v1'
HISTORICAL_SHA = {
    'train_preflight':'1571670e2523976e04aabe7976e64388cfded8b7c124533238799123392e719b',
    'dev':'6c108dff3c01dbf54053f80ea9eba563e57e654de6959466e360b00e2218e06e'}
FIT_SHA = '384bc02b2d0237dfa7a1331257502a6858db1b5df62e62d463010ae8921d6b1b'
SEARCH_KEYS = ('status','expanded_nodes','neighbor_edges_examined','free_edges_examined',
    'diagonal_supercover_voxels_examined','virtual_start_edges','virtual_goal_edges',
    'virtual_goal_edges_examined','permission_start_edge_checks','permission_target_edge_checks',
    'successful_route_grid_edges','successful_route_virtual_edges','successful_route_edges_in_start_permission',
    'successful_route_edges_in_target_permission','selected_start_attachment','selected_goal_attachment',
    'maximum_expanded_nodes','search_deadline_seconds','complete_raw_paths_emitted')


def array_difference(left, right):
    left,right=np.asarray(left),np.asarray(right)
    same_shape=left.shape==right.shape
    exact=bool(same_shape and np.array_equal(left,right,equal_nan=True))
    finite_same=bool(same_shape and np.array_equal(np.isfinite(left),np.isfinite(right)))
    maximum=None
    if same_shape:
        both=np.isfinite(left)&np.isfinite(right)
        if both.any():maximum=float(np.max(np.abs(left[both]-right[both])))
    return dict(left_shape=list(left.shape),right_shape=list(right.shape),equal_values=exact,
        equal_dtype_and_bytes=bool(same_shape and left.dtype==right.dtype and left.tobytes()==right.tobytes()),
        finite_masks_equal=finite_same,max_absolute_error_on_joint_finite_entries=maximum)


def compare_slot(old, new):
    result={name:array_difference(old[name],new[name]) for name in ('raw','h24','opened')}
    differences={key:dict(python=old['attempt'].get(key),native=new['attempt'].get(key)) for key in SEARCH_KEYS
        if old['attempt'].get(key)!=new['attempt'].get(key)}
    result.update(deterministic_search_equal=not differences,deterministic_search_differences=differences,
        python_search_status=old['attempt']['status'],native_search_status=new['attempt']['status'],
        python_search_timed_out=old['attempt']['status']=='search_time_budget_exhausted',
        native_search_timed_out=new['attempt']['status']=='search_time_budget_exhausted',
        saved_candidate_decisions_equal=old['candidate']==new['candidate'])
    return result


def pair_slots(old, new):
    left,right=({(r['id'],r['slot']):r for r in rows} for rows in (old,new))
    saved.require(len(left)==len(old) and len(right)==len(new) and left.keys()==right.keys(),
        'Exactly one saved slot for every paired ID/slot required')
    return [dict(id=identifier,slot=slot,**compare_slot(left[(identifier,slot)],right[(identifier,slot)]))
            for identifier,slot in sorted(left)]


def read_pool(meta,binary,records,ids,arm,native,stage,report,verified=None):
    meta,binary=Path(meta),Path(binary)
    pool_name=arm+'_predictions.npz' if native or stage=='train_preflight' else 'predictions.npz'
    if verified is not None:saved.require(pool_name in verified,'Combined pool is not sealed')
    with np.load(binary/pool_name,allow_pickle=False) as data:
        pool_ids=data['scene_ids'].astype(str).tolist();parents=data['parent_ids'].astype(str).tolist()
        paths=data['paths'].copy();events=data['gripper_open'].copy()
    saved.require(sorted(pool_ids)==ids and len(pool_ids)==len(set(pool_ids)) and
        paths.shape==(len(ids),4,24,3) and events.shape==(len(ids),4,24),'Full saved pool identity/shape required')
    by_id={r['id']:r for r in records}
    saved.require(len(by_id)==len(records)==len(ids) and sorted(by_id)==ids,'Complete request population required')
    cases,slots,outcomes,statuses=[],[],Counter(),Counter()
    role='TRAIN' if stage=='train_preflight' else 'DEV_MODEL'
    for identifier in ids:
        row=by_id[identifier];i=pool_ids.index(identifier);parent=identifier.rsplit('_target',1)[0]
        relative=Path(arm)/identifier if native or stage=='train_preflight' else Path('requests')/identifier
        q,b=meta/relative,binary/relative
        saved.require(row['split']==role and row['parent_id']==parents[i]==parent,'Parent/role differs')
        saved.require(saved.read(q/'result.json')==row and saved.read(q/'status.json')['status']=='completed',
            'Saved request/result differs')
        seal=saved.read(q/'generation_seal.json')
        saved.require(seal['id']==identifier and seal['evaluation_labels_opened'] is False and
            seal['submitted_candidate_budget']==4,'Post-seal label boundary differs')
        for field,name in [('prediction_sha256','predictions.npz'),('raw_paths_sha256','raw_paths.npz')]:
            saved.require(row[field]==seal[field]==saved.sha(b/name),'Sealed request SHA differs')
        if verified is not None:
            for name in ('result.json','status.json','generation_seal.json','predictions.npz','raw_paths.npz'):
                saved.require((relative/name).as_posix() in verified,'Unsealed request artifact')
        with np.load(b/'predictions.npz',allow_pickle=False) as request:
            saved.require(request['paths'].shape==(1,4,24,3) and request['gripper_open'].shape==(1,4,24) and
                request['scene_ids'].tolist()==[identifier] and request['parent_ids'].tolist()==[parent],
                'Per-request shape/identity differs')
            saved.require(np.array_equal(paths[i],request['paths'][0],equal_nan=True) and
                np.array_equal(events[i],request['gripper_open'][0],equal_nan=True),'Combined/request arrays differ')
        with np.load(b/'raw_paths.npz',allow_pickle=False) as raw:
            saved.require(set(raw.files)=={'candidate_%d'%k for k in range(4)},'All raw slots required')
            raw_paths=[raw['candidate_%d'%k].copy() for k in range(4)]
        generation=row['generation'];attempts=generation['attempts']
        detail=saved.summarize_candidates(row['candidates'],paths[i],events[i],raw_paths,attempts)
        saved.require(generation['submitted_candidate_budget']==4 and generation['raw_complete_paths']==detail['raw_complete']
            and generation['failed_slots']==detail['failed_generation'],'All K4 failure budget required')
        outcomes.update(detail['outcomes']);statuses.update(detail['search_statuses'])
        for key,value in [('TipValidAtK',detail['valid']/4),('AnyTipValidAtK',float(detail['valid']>0)),
            ('UniqueClassifiedTipValidAtK',detail['known_unique']),('DuplicateClassifiedTipValidCount',detail['known_duplicate']),
            ('UnknownTypeTipValidCount',detail['unknown_valid'])]:
            saved.equal_number(row['metrics'][key],value,'Saved checker count differs: '+key)
        if native:
            wrapper=saved.read(q/'native_request_receipt.json')
            saved.require((relative/'native_request_receipt.json').as_posix() in verified and
                wrapper['id']==identifier and wrapper['arm']==arm and wrapper['split']==role and
                wrapper['library_sha256']==report['library_sha256'],'Native request/binary receipt differs')
            seconds=wrapper['native_context_through_checked_pool_seconds']
        else:seconds=row['timing']['observation_to_checked_pool_seconds']
        for k,attempt in enumerate(attempts):
            if native and attempt['status'] not in ('localization_failure','grid_construction_rejected'):
                saved.require(attempt.get('native',{}).get('library_sha256')==report['library_sha256'] and
                    attempt['native']['test_only'] is False and attempt['maximum_expanded_nodes']==20000 and
                    attempt['search_deadline_seconds']==2.,'Wrong native backend or budget')
            slots.append(dict(id=identifier,slot=k,raw=raw_paths[k],h24=paths[i,k],opened=events[i,k],
                attempt=attempt,candidate=row['candidates'][k]))
        field_seconds=sum(a.get('spatial_penalty',{}).get('preprocessing_seconds',0.) for a in attempts)
        cases.append(dict(id=identifier,parent_id=parent,metrics=row['metrics'],**detail,
            request_seconds=seconds,field_seconds=field_seconds))
    result=dict(cases=cases,totals={key:sum(c[key] for c in cases) for key in
        ('valid','known_unique','known_duplicate','unknown_valid','raw_complete','failed_generation','finite_invalid')},
        outcomes=dict(outcomes),search_statuses=dict(statuses),field_seconds=sum(c['field_seconds'] for c in cases),
        request_seconds=sum(c['request_seconds'] for c in cases))
    return result,slots


def analyze(run,binary,historical,historical_binary):
    run,binary,historical,historical_binary=map(Path,(run,binary,historical,historical_binary))
    report=saved.read(run/'report.json');stage=report['stage']
    saved.require(stage in HISTORICAL_SHA and report['status']=='completed' and report['protocol']==NATIVE_PROTOCOL,
        'Only completed native paired stage supported')
    start,stop,role=(283200,283204,'TRAIN') if stage=='train_preflight' else (283264,283276,'DEV_MODEL')
    ids=sorted('two_row_reach_%d_target%d'%(p,t) for p in range(start,stop) for t in range(3))
    saved.require(sorted(report['input_ids'])==ids and len(report['input_ids'])==len(ids) and
        report['requested_candidate_slots']==len(ids)*8 and report['completed_requests']==len(ids)*2 and
        report['failed_requests']==report['unattempted_requests']==0 and
        set(report['per_request'])==set(report['results'])=={'edge','spatial'},'Full fixed paired stage required')
    saved.require(report['shared_fit_sha256']==FIT_SHA and report['actual_train_parents']==31,
        'Different TRAIN fit')
    verified=saved.verify_artifacts(report,run,binary)
    old=saved.read(historical/'report.json')
    saved.require(saved.sha(historical/'report.json')==HISTORICAL_SHA[stage], 'Different historical Python baseline')
    saved.require(old['planner_config']==report['planner_config'] and old['prototype_config']==report['prototype_config'],
        'Algorithm parameters changed')
    native_results={};native_slots={}
    for arm in ('edge','spatial'):
        native_results[arm],native_slots[arm]=read_pool(run,binary,report['per_request'][arm],ids,arm,True,stage,report,verified)
        for key in saved.METRICS:
            values=[r['metrics'][key] for r in native_results[arm]['cases'] if r['metrics'][key] is not None]
            saved.equal_number(report['results'][arm][key],np.mean(values) if values else None,'Native aggregate '+key)
    if stage=='train_preflight':
        old_verified=saved.verify_artifacts(old,historical,historical_binary)
        old_rows=old['per_request']['edge']
    else:
        old_verified=None;old_rows=old['per_scene']
        saved.require(saved.sha(historical_binary/'predictions.npz')==old['prediction_sha256'],'Historical combined pool hash differs')
    old_results,old_slots=read_pool(historical,historical_binary,old_rows,ids,'edge',False,stage,old,old_verified)
    differential=pair_slots(old_slots,native_slots['edge'])
    paired=saved.paired_cases(native_results['edge'],native_results['spatial'])
    summary=dict(slots=len(differential),old_complete_paths=old_results['totals']['raw_complete'],
        exact_raw_slots=sum(d['raw']['equal_values'] for d in differential),
        exact_h24_slots=sum(d['h24']['equal_values'] for d in differential),
        exact_event_slots=sum(d['opened']['equal_values'] for d in differential),
        exact_deterministic_search_slots=sum(d['deterministic_search_equal'] for d in differential),
        exact_saved_checker_slots=sum(d['saved_candidate_decisions_equal'] for d in differential),
        historical_python_timeout_slots=sum(d['python_search_timed_out'] for d in differential))
    return dict(protocol=PROTOCOL,stage=stage,role=role,source_report_sha256=saved.sha(run/'report.json'),
        historical_python_report_sha256=saved.sha(historical/'report.json'),source_commit=report['source_commit'],
        source_files_sha256=report['source_files_sha256'],verified_native_artifacts=verified,
        native_arms=native_results,native_all_pairs=paired,historical_python_edge=old_results,
        differential_summary=summary,all_python_native_edge_slots=differential,
        source_analysis_sha256=saved.sha(__file__),source_helper_sha256=saved.sha(saved.__file__),
        new_search_calls=0,new_model_calls=0,raw_collection_labels_opened=False,paths_filtered_or_retried=False,
        limitation='Exact equality is checked only on these saved cases, not a proof for all inputs. Unequal paths and counters remain visible; elapsed time is excluded from deterministic search equality.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','binary-root','historical','historical-binary','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    saved.require(not args.output.exists(),'Fresh derived output required')
    result=analyze(args.run,args.binary_root,args.historical,args.historical_binary)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['differential_summary']))
