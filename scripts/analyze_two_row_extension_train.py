"""Read-only extension288 fixed TRAIN prefix all-slot audit; no new inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from scripts import collect_two_row_extension as extension
from scripts import analyze_two_row_formal_train as historical

physical=historical.physical
pilot=historical.pilot
batch=historical.batch
diagnostic=historical.diagnostic
TrainReader=historical.TrainReader
indexed_slots=historical.indexed_slots
condition_summary=historical.condition_summary
trace_diagnostics=historical.trace_diagnostics
plot_target=historical.plot_target
PROTOCOL='two_row_extension288_fixed_train_prefix_all_slots_v1'
PREFIXES=(32,64,128,256)


def prefix_indices(train_parents):
    if isinstance(train_parents,bool) or not isinstance(train_parents,int) or train_parents not in PREFIXES:
        raise ValueError('only fixed TRAIN prefixes 32/64/128/256 are allowed')
    return list(range(train_parents))


def mechanical_snapshot(corpus,train_parents):
    """Only registration, manifest, closed mechanical receipts/ledger hashes."""
    indices=prefix_indices(train_parents)
    value,manifest=extension.verify_corpus(corpus)
    rows=extension.checked_closures(corpus,value)
    selected={r['index']:r for r in rows if r['index'] in indices}
    return dict(protocol=PROTOCOL,source_registration_sha256=manifest['registration_sha256'],
        selected_train_indices=indices,requested_parents=train_parents,requested_slots=27*train_parents,
        closed_train_parents=len(selected),missing_closure_indices=[i for i in indices if i not in selected],
        unavailable_initial_parent_ids=[r['parent_id'] for r in selected.values() if not r['initial_observation_saved']],
        attempted_lower=sum(r['attempted_lower'] for r in selected.values()),
        attempted_upper=sum(r['attempted_upper'] for r in selected.values()),
        closed_parent_unattempted_lower=sum(r['unattempted_lower'] for r in selected.values()),
        closed_parent_unattempted_upper=sum(r['unattempted_upper'] for r in selected.values()),
        unfinished_parent_requested_slots=27*(train_parents-len(selected)),
        all_raw_opened=False,new_dev_raw_sealed=True,model_training_authorized=False,
        interpretation='Live mechanical snapshot only; unfinished slots are not classified as final failures or unattempted.')


def closed_selection(corpus,train_parents):
    """Entire registered prefix must close before the first TRAIN raw read."""
    indices=prefix_indices(train_parents)
    value,manifest=extension.verify_corpus(corpus)
    rows=extension.checked_closures(corpus,value)
    mapping={r['index']:r for r in rows}
    missing=[i for i in indices if i not in mapping]
    if missing:
        raise ValueError('entire fixed TRAIN prefix must be closed before raw analysis: '+str(missing))
    plans=[value['parent_plan'][i] for i in indices]
    if any(p['role']!='TRAIN' or p['index']!=i for i,p in zip(indices,plans)):
        raise ValueError('selection is not the registered TRAIN prefix')
    return plans,[mapping[i] for i in indices],manifest,extension.live_layout_gate(corpus)


def analyze(corpus, output, train_parents, plots=True):
    started=time.perf_counter();corpus=Path(corpus);output=Path(output)
    if output.exists():raise FileExistsError('analysis requires a fresh output directory')
    plans,closures,manifest,gate=closed_selection(corpus,train_parents)
    requested_slots=27*train_parents
    output.mkdir(parents=True);parents=[];conditions=[];details=[];source_hashes={};status_counts=Counter()
    for plan,closure in zip(plans,closures):
        parent=plan['parent_id'];raw_root=corpus/'parents'/'TRAIN'/parent
        if raw_root.resolve()!=corpus.resolve()/'parents'/'TRAIN'/parent:
            raise ValueError('selected TRAIN root is redirected outside its registered location')
        reader=TrainReader(raw_root)
        index=reader.json('artifact_hashes.json',{})
        for name,digest in index.items():reader.checked(name,digest)
        records,partial_attempt=reader.rows('attempts.jsonl');ledger,partial_ledger=reader.rows('slot_ledger.jsonl')
        observations,_=reader.rows('observations.jsonl');supervision,_=reader.rows('supervision.jsonl')
        layouts,_=reader.rows('layout_audits.jsonl');summary=reader.json('summary.json',{})
        if any(r.get('parent_id')!=parent or r.get('split')!='TRAIN' for r in observations+supervision):
            raise ValueError('raw observation/supervision identity or split changed')
        expected_ids={parent+'_target%d'%i for i in range(3)}
        if any(set(r)!={'id','parent_id','split','image','instruction'} or r['id'] not in expected_ids for r in observations):
            raise ValueError('observation whitelist or target id changed')
        if len({r['id'] for r in observations})!=len(observations):raise ValueError('duplicate observation id')
        same_image=bool(observations) and len({r['image'] for r in observations})==1
        if observations and (not same_image or len({r['instruction'] for r in observations})!=len(observations)):
            raise ValueError('same image / different target language contract changed')
        for row in observations:reader.checked(row['image'])
        goals=np.asarray(plan['config']['goal_xyz'])
        verification=parent+'/verification_only.npz'
        if reader.path(verification).exists():
            with np.load(reader.checked(verification),allow_pickle=False) as arrays:goals=arrays['target_centers'].copy()
            if not np.allclose(goals,plan['config']['goal_xyz'],atol=1e-6,rtol=0):raise ValueError('actual TRAIN goals differ from registration')
        slots=indexed_slots(parent,records,ledger,partial_ledger)
        if sum(s['issued'] for s in slots)!=closure['attempted_lower'] or sum(s['ledger_completed'] for s in slots)!=closure['completed_slots']:
            raise ValueError('slot reconstruction differs from mechanical closure')
        paths={}
        for slot in slots:
            detail,xyz=trace_diagnostics(reader,plan,slot,goals);details.append(detail);status_counts[slot['status']]+=1
            if xyz is not None:paths[(slot['target'],slot['attempt'])]=xyz
        parent_dir=output/parent;parent_dir.mkdir()
        for target in range(3):
            selected=[s for s in slots if s['target']==target]
            condition=dict(parent_id=parent,id=parent+'_target%d'%target,target=target,**condition_summary(selected))
            conditions.append(condition)
            if plots:plot_target(plan,target,selected,{a:p for (t,a),p in paths.items() if t==target},parent_dir/('target%d_all9.png'%target))
        if observations:
            initial=reader.checked(observations[0]['image']);(parent_dir/'front.png').write_bytes(initial.read_bytes())
        if any(not k.startswith('route:') for k in summary.get('planning_api_entry_counts',{})):
            raise ValueError('non-route initialization/planning guard entry found')
        parents.append(dict(parent_id=parent,index=plan['index'],closure=closure,layout_audits=layouts,
            collection_summary=summary,complete_outcome_records=len(records),attempt_jsonl_partial=partial_attempt,
            ledger_jsonl_partial=partial_ledger,artifact_index_entries_verified=len(index),
            observed_target_ids=[r['id'] for r in observations],missing_observation_ids=sorted(expected_ids-{r['id'] for r in observations}),
            same_image_three_distinct_instructions=bool(same_image and len(observations)==3),
            model_use_blocked=parent in gate['blocked_parent_ids']))
        reader.verify_unchanged();source_hashes[parent]=reader.hashes
    accepted=[r for r in details if r['status']=='accepted']
    lengths=[r['recomputed_fields']['length_m'] for r in accepted]
    failed=[r for r in details if r['status']=='failed']
    # Diagnostic predicates overlap: do not imply mutually exclusive root causes.
    failure_predicates=dict(
        recorded_robot_collision=sum(bool(r.get('collision_pair')) for r in failed),
        failed_planning_stage=sum((r.get('first_failed_stage') or {}).get('stage')=='planning' for r in failed),
        failed_simulation_stage=sum((r.get('first_failed_stage') or {}).get('stage')=='simulation' for r in failed),
        strict_restore_not_passed=sum(not r.get('strict_restore_passed',False) for r in failed),
        raw_tip_check_failed=sum(r.get('recomputed_fields',{}).get('tip_polyline_clear') is False for r in failed),
        h24_tip_check_failed=sum(r.get('recomputed_fields',{}).get('tip_polyline_24_clear') is False for r in failed),
        raw_h24_type_changed=sum('recomputed_fields' in r and r['recomputed_fields']['actual_route_type']!=r['recomputed_fields']['h24_route_type'] for r in failed),
        diagnostic_tip_endpoint_h24_passed_despite_robot_failure=sum(r.get('diagnostic_tip_endpoint_h24_passed',False) for r in failed))
    result=dict(protocol=PROTOCOL,source_corpus=str(corpus),source_registration_sha256=manifest['registration_sha256'],
        source_collector_sha256=manifest['source_sha256'],analysis_source_sha256=batch.digest(__file__),
        analysis_dependencies_sha256={Path(m.__file__).name:batch.digest(m.__file__) for m in (historical,diagnostic,extension)},
        requested_parents=train_parents,requested_conditions=3*train_parents,requested_slots=requested_slots,slot_status_counts=dict(status_counts),
        attempted_lower=sum(c['attempted_lower'] for c in closures),attempted_upper=sum(c['attempted_upper'] for c in closures),
        unattempted_lower=sum(c['unattempted_lower'] for c in closures),unattempted_upper=sum(c['unattempted_upper'] for c in closures),
        accepted_reference_count=len(accepted),accepted_per_requested_slot=len(accepted)/requested_slots,
        accepted_unknown_count=sum(c['unknown_valid_references'] for c in conditions),
        known_duplicate_reference_count=sum(c['known_duplicate_references'] for c in conditions),
        conditions_with_more_than_K4_known_types=sum(c['has_more_than_K4_known_types'] for c in conditions),
        distinct_known_type_histogram=dict(Counter(c['distinct_known_types'] for c in conditions)),
        failure_errors=dict(Counter(r.get('error') for r in details if r['status']=='failed')),
        failure_predicates_overlapping=failure_predicates,
        strict_route_restore_passes=sum(r.get('strict_restore_passed',False) for r in details),
        explicit_get_path_calls=sum(r.get('planning_calls',0) for r in details),
        route_planning_seconds=sum(r.get('planning_seconds',0) for r in details),
        route_simulation_seconds=sum(r.get('simulation_seconds',0) for r in details),
        worker_elapsed_seconds=sum(c['worker_elapsed_seconds'] or 0 for c in closures),
        unknown_worker_elapsed_parents=sum(c['worker_elapsed_unknown'] for c in closures),
        accepted_length_m=dict(minimum=min(lengths),mean=float(np.mean(lengths)),maximum=max(lengths)) if lengths else None,
        mechanical_layout_gate=gate,mechanical_gate_is_current_snapshot=True,
        parents=parents,conditions=conditions,source_files_sha256=source_hashes,
        selected_train_indices=list(range(train_parents)),missing_observation_count=sum(len(p['missing_observation_ids']) for p in parents),
        new_dev_raw_sealed=True,reference_types_are_incomplete=True,
        plots_written=3*train_parents if plots else 0,analysis_elapsed_seconds=time.perf_counter()-started,
        dev_or_higher_raw_opened=False,all_solution_count=None,
        interpretation='Narrow-ID TRAIN collection quality only. Known reference support is incomplete; unknown stays positive. Tip/H24 rechecks are not a continuous whole-robot certificate. No label filtering, method comparison or final-test claim.')
    batch.write(output/'analysis.json',result);batch.write(output/'all_requested_slots.json',details)
    batch.write(output/'artifact_hashes.json',{p.relative_to(output).as_posix():batch.digest(p) for p in output.rglob('*') if p.is_file()})
    return result



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True)
    parser.add_argument('--train-parents',type=int,choices=PREFIXES,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--mechanical-only',action='store_true')
    args=parser.parse_args()
    if args.mechanical_only:
        if args.output.exists():raise FileExistsError('mechanical snapshot output must be fresh')
        value=mechanical_snapshot(args.corpus,args.train_parents)
        args.output.parent.mkdir(parents=True,exist_ok=True);batch.write(args.output,value)
        print(json.dumps({k:value[k] for k in ('requested_parents','closed_train_parents','unfinished_parent_requested_slots')}))
    else:
        value=analyze(args.corpus,args.output,args.train_parents)
        print(json.dumps({k:value[k] for k in ('requested_slots','slot_status_counts','accepted_reference_count','analysis_elapsed_seconds')}))


if __name__=='__main__':main()
