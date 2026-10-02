"""Read-only, fixed first-16 TRAIN audit; every requested slot is retained."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from scripts import collect_two_row_formal as formal
from scripts import analyze_observed_two_row_pilot as diagnostic

physical = formal.physical
pilot = physical.pilot
batch = physical.batch
PROTOCOL = 'two_row_formal_first16_train_all432_slots_v1'
INDICES = tuple(range(16))


def closed_selection(corpus):
    """Finish the complete metadata gate before opening any selected raw file."""
    registration, manifest = formal.verify_corpus(corpus)
    selected = [registration['parent_plan'][index] for index in INDICES]
    plans = {p['parent_id']: p for p in registration['parent_plan']}
    closures = []
    for plan in selected:
        if plan['role'] != 'TRAIN' or plan['index'] not in INDICES:
            raise ValueError('only the fixed first sixteen TRAIN parents are allowed')
        path = corpus/'closures'/('%03d.json' % plan['index'])
        if not path.exists():
            raise ValueError('entire fixed TRAIN prefix must be closed before raw analysis')
        row = json.loads(path.read_text())
        formal.validate_closure(corpus, row, plans)
        closures.append(row)
    # This function reads mechanical metadata only, including later roles.
    gate = formal.live_layout_gate(corpus)
    return selected, closures, manifest, gate


class TrainReader:
    """Restrict even index/trace symlinks to this one registered TRAIN parent."""
    def __init__(self, root):
        self.root = root.resolve()
        self.hashes = {}

    def path(self, name):
        path = (self.root/name).resolve()
        try:
            path.relative_to(self.root)
        except ValueError:
            raise ValueError('TRAIN artifact escaped its registered parent directory')
        return path

    def checked(self, name, expected=None):
        path = self.path(name)
        actual = batch.digest(path)
        if expected is not None and actual != expected:
            raise ValueError('TRAIN artifact SHA mismatch: '+str(name))
        self.hashes[path.relative_to(self.root).as_posix()] = actual
        return path

    def json(self, name, default=None):
        if not self.path(name).exists():
            return default
        return json.loads(self.checked(name).read_text())

    def rows(self, name):
        if not self.path(name).exists():
            return [], False
        text = self.checked(name).read_text()
        partial = bool(text and not text.endswith('\n'))
        lines = text.splitlines()
        if partial:
            lines = lines[:-1]
        return [json.loads(line) for line in lines], partial

    def verify_unchanged(self):
        if any(batch.digest(self.path(name)) != value for name, value in self.hashes.items()):
            raise ValueError('closed TRAIN evidence changed during analysis')


def indexed_slots(parent, records, ledger, partial_ledger=False):
    """An interrupted slot is distinct from an unissued slot or a failed route."""
    def key(row):
        if row.get('parent_id') != parent:
            raise ValueError('attempt/ledger parent differs from selection')
        names = [parent+'_target%d' % i for i in range(3)]
        if row.get('input_id') not in names or not isinstance(row.get('attempt'), int) or not 0 <= row['attempt'] < 9:
            raise ValueError('attempt/ledger target or proposal outside registration')
        return names.index(row['input_id']), row['attempt']
    issued, completed, indexed = set(), set(), {}
    for row in ledger:
        k = key(row)
        if row['event'] == 'started' and k not in issued:
            issued.add(k)
        elif row['event'] == 'completed' and k in issued and k not in completed:
            completed.add(k)
        else:
            raise ValueError('duplicate or disordered mechanical slot')
    for row in records:
        k = key(row)
        if k in indexed or k not in issued:
            raise ValueError('duplicate or unissued attempt record')
        indexed[k] = row
    if not completed.issubset(indexed):
        raise ValueError('completed ledger slot lacks its prior complete attempt record')
    result = []
    unknown_assigned = False
    for target in range(3):
        for attempt in range(9):
            k = target, attempt
            record = indexed.get(k)
            status = ('accepted' if record['success'] else 'failed') if record else ('interrupted' if k in issued else 'unattempted')
            if partial_ledger and status == 'unattempted' and not unknown_assigned:
                status = 'unattempted_or_unrecorded'
                unknown_assigned = True
            result.append(dict(parent_id=parent, input_id=parent+'_target%d' % target, target=target,
                attempt=attempt, status=status, issued=k in issued, ledger_completed=k in completed,
                record=record))
    return result


def condition_summary(slots):
    accepted = [s['record'] for s in slots if s['status'] == 'accepted']
    known = [tuple(r['actual_route_type']) for r in accepted if r['actual_route_type'] is not None]
    unique = sorted(set(known))
    return dict(requested_slots=9, statuses=dict(Counter(s['status'] for s in slots)),
        recorded_attempts=sum(s['record'] is not None for s in slots), valid_references=len(accepted),
        distinct_known_types=len(unique), known_types=unique, known_duplicate_references=len(known)-len(unique),
        unknown_valid_references=sum(r['actual_route_type'] is None for r in accepted),
        known_over_references=sum('over' in value for value in known),
        distinct_lateral_types=sum(all(x in pilot.PASSAGES for x in value) for value in unique),
        has_more_than_K4_known_types=len(unique)>4, total_solution_count=None)


def trace_diagnostics(reader, plan, slot, goals):
    record = slot['record']; detail = {k: v for k, v in slot.items() if k != 'record'}
    if record is None:
        return detail, None
    detail.update(error=record.get('error'), collision_pair=record.get('collision_pair'),
        actual_accepted_type=record.get('actual_route_type') if record['success'] else None,
        strict_restore_passed=bool(record.get('strict_restore', {}).get('passed')),
        camera_flags=record.get('camera_flags'), seconds=record.get('seconds'),
        planning_calls=sum(s['get_path_calls'] for s in record.get('planning_segments', [])),
        planning_seconds=sum(s['planning_seconds'] for s in record.get('planning_segments', [])),
        simulation_seconds=sum(s['simulation_seconds'] for s in record.get('planning_segments', [])),
        first_failed_stage=next((dict(segment=s['segment'], stage=stage)
            for s in record.get('planning_segments', []) for stage in ('planning','simulation')
            if s[stage+'_status']=='failed'), None))
    trace = record.get('trajectory') or record.get('failed_partial_trajectory')
    if not trace:
        return detail, None
    filename = plan['parent_id']+'/'+trace['file']
    with np.load(reader.checked(filename, trace['sha256']), allow_pickle=False) as arrays:
        xyz = arrays['gripper_pose'][:, :3].copy()
        events = arrays['gripper_open'].copy()
        saved_h24 = arrays['xyz_24'].copy() if 'xyz_24' in arrays else None
    if len(xyz) != trace['samples'] or not np.isfinite(xyz).all():
        raise ValueError('trace sample count/finiteness differs from evidence')
    detail.update(trace_file=filename, trace_sha256=trace['sha256'], samples=len(xyz),
        event_transitions=int(np.count_nonzero(np.diff(events>.5))))
    if len(xyz) > 1:
        fields, h24, accepted_proxy = physical.route_acceptance(xyz, goals, slot['target'], plan['config'])
        detail.update(recomputed_fields=fields, diagnostic_tip_endpoint_h24_passed=accepted_proxy,
            actual_row_crossings=diagnostic.row_crossings(xyz, plan['config']),
            guide_segments=diagnostic.segment_diagnostics(xyz, record.get('planning_segments', []), plan['config']))
        if record['success']:
            if not accepted_proxy or fields['actual_route_type'] != (tuple(record['actual_route_type']) if record['actual_route_type'] is not None else None):
                raise ValueError('accepted reference fails unchanged raw/H24/type checks')
            if saved_h24 is None or not np.array_equal(h24, saved_h24):
                raise ValueError('accepted stored H24 differs from collector resampling')
    elif record['success']:
        raise ValueError('accepted reference lacks a full polyline')
    return detail, xyz


def plot_target(plan, target, slots, paths, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    config=plan['config'];centers,halves=pilot.geometry(config);goals=np.asarray(config['goal_xyz'])
    all_points=[centers-halves,centers+halves,goals,np.asarray(config['entry_xyz'])[None]]
    all_points.extend(paths.values());all_points=np.concatenate(all_points)
    low=all_points.min(0)-.035;high=all_points.max(0)+.035
    fig, axes=plt.subplots(3,6,figsize=(21,10),squeeze=False)
    for slot in slots:
        attempt=slot['attempt'];xyz=paths.get(attempt);record=slot['record']
        route_type=record.get('actual_route_type') if record else None
        label='/'.join(route_type) if route_type is not None else 'unknown' if slot['status']=='accepted' else slot['status']
        for projection,(a,b) in enumerate(((0,1),(0,2))):
            ax=axes[attempt//3,(attempt%3)*2+projection]
            for c,h in zip(centers,halves):
                ax.add_patch(Rectangle((c[a]-h[a],c[b]-h[b]),2*h[a],2*h[b],color='gray',alpha=.3))
            if xyz is not None:
                ax.plot(xyz[:,a],xyz[:,b],c='#087f5b' if slot['status']=='accepted' else '#c92a2a',lw=1.2)
                ax.scatter(xyz[-1,a],xyz[-1,b],s=16,marker='x',c='black')
            else:
                ax.text(.5,.5,'NO RECORDED PATH',transform=ax.transAxes,ha='center',fontsize=7)
            ax.scatter(goals[target,a],goals[target,b],s=38,marker='*',c='black')
            ax.set(xlim=(low[a],high[a]),ylim=(low[b],high[b]),xlabel='xyz'[a]+' (m)',ylabel='xyz'[b]+' (m)')
            ax.set_title('%d %s\n%s'%(attempt,slot['status'],label),fontsize=8)
            ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.2);ax.tick_params(labelsize=7)
    fig.suptitle('%s target%d | all 9 registered slots, XY/XZ pairs\nGreen=accepted; red=failed full/partial; gray=post projections; star=goal label'%(plan['parent_id'],target),fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(output,dpi=130);plt.close(fig)


def analyze(corpus, output, plots=True):
    started=time.perf_counter();corpus=Path(corpus);output=Path(output)
    if output.exists():raise FileExistsError('analysis requires a fresh output directory')
    plans,closures,manifest,gate=closed_selection(corpus)
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
        analysis_dependencies_sha256={Path(diagnostic.__file__).name:batch.digest(diagnostic.__file__)},
        requested_parents=16,requested_conditions=48,requested_slots=432,slot_status_counts=dict(status_counts),
        attempted_lower=sum(c['attempted_lower'] for c in closures),attempted_upper=sum(c['attempted_upper'] for c in closures),
        unattempted_lower=sum(c['unattempted_lower'] for c in closures),unattempted_upper=sum(c['unattempted_upper'] for c in closures),
        accepted_reference_count=len(accepted),accepted_per_requested_slot=len(accepted)/432,
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
        plots_written=48 if plots else 0,analysis_elapsed_seconds=time.perf_counter()-started,
        dev_or_higher_raw_opened=False,all_solution_count=None,
        interpretation='Narrow-ID TRAIN collection quality only. Known reference support is incomplete; unknown stays positive. Tip/H24 rechecks are not a continuous whole-robot certificate. No label filtering, method comparison or final-test claim.')
    batch.write(output/'analysis.json',result);batch.write(output/'all_432_slots.json',details)
    batch.write(output/'artifact_hashes.json',{p.relative_to(output).as_posix():batch.digest(p) for p in output.rglob('*') if p.is_file()})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-plots',action='store_true',help='Testing only; registered actual audit uses all 48 plots.')
    args=parser.parse_args();result=analyze(args.corpus,args.output,not args.no_plots)
    print(json.dumps({k:result[k] for k in ('requested_slots','slot_status_counts','accepted_reference_count','conditions_with_more_than_K4_known_types','analysis_elapsed_seconds')}))


if __name__=='__main__':main()
