"""Read sealed A*32/64 artifacts locally. No planner/model/raw-data imports."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x]


def opportunity(candidates, types):
    known = {tuple(x) for x in types if x is not None}
    valid = [c for c in candidates if c['TipValid']]
    classified = [tuple(c['declared_passage_type']) for c in valid if c['declared_passage_type'] is not None]
    unique = set(classified)
    duplicate = len(classified) - len(unique)
    missing = known - unique
    return dict(known_reference_types=sorted(known), valid_predicted_types=sorted(unique),
        missing_known_types=sorted(missing), valid_classified_duplicate_slots=duplicate,
        unknown_valid_slots=sum(c['declared_passage_type'] is None for c in valid),
        duplicate_replacement_count_upper_bound=min(duplicate, len(missing)))


def analyze(prefix):
    family = 'observation_two_row_astar_prefix' + prefix + '_v1'
    p = ROOT / 'reports' / family
    binary = ROOT / 'runs' / family
    report, original_index = read(p/'dev_model/report.json'), read(p/'dev_model/artifact_index.json')
    sync = read(p/'SYNC_SHA256_INDEX.json')
    for r in sync['entries']:
        path = ROOT/r['local_path']
        assert sha(path) == r['sha256'] and path.stat().st_size == r['bytes']
    for name, entry in original_index.items():
        f = (binary if name.endswith('.npz') else p)/'dev_model'/name
        assert sha(f) == entry['sha256'] and f.stat().st_size == entry['bytes'], name
    metadata = p/'input_metadata'
    manifest = read(metadata/'export_manifest.json')
    assert sha(metadata/'export_manifest.json') == report['source_export_manifest_sha256']
    for name in ('observations.jsonl', 'supervision.jsonl'):
        assert sha(metadata/name) == manifest['output_files_sha256'][name]
    observations = {r['id']: r for r in rows(metadata/'observations.jsonl') if r['split'] == 'DEV_MODEL'}
    labels = {r['id']: r for r in rows(metadata/'supervision.jsonl') if r['split'] == 'DEV_MODEL'}
    assert observations.keys() == labels.keys() and len(observations) == 36
    with np.load(binary/'dev_model/predictions.npz', allow_pickle=False) as a:
        ids = list(map(str, a['scene_ids'])); paths = a['paths'].copy(); events = a['gripper_open'].copy()
    assert len(ids) == len(set(ids)) == 36 and set(ids) == set(observations)
    assert paths.shape == (36,4,24,3) and events.shape == (36,4,24)
    assert len(report['per_scene']) == len({r['id'] for r in report['per_scene']}) == 36
    cases = []; candidate_outcomes = Counter(); search_statuses = Counter(); joint_groups = Counter()
    for raw in report['per_scene']:
        identifier = raw['id']; i = ids.index(identifier)
        q = p/'dev_model/requests'/identifier; b = binary/'dev_model/requests'/identifier
        assert read(q/'result.json') == raw
        seal = read(q/'generation_seal.json')
        assert seal['prediction_sha256'] == raw['prediction_sha256'] == sha(b/'predictions.npz')
        assert seal['raw_paths_sha256'] == raw['raw_paths_sha256'] == sha(b/'raw_paths.npz')
        with np.load(b/'predictions.npz', allow_pickle=False) as a:
            np.testing.assert_array_equal(paths[i], a['paths'][0])
            np.testing.assert_array_equal(events[i], a['gripper_open'][0])
        generation = raw['generation']; attempts = generation['attempts']
        assert len(attempts) == len(raw['candidates']) == 4
        per_slot=[]
        for k,(attempt,candidate) in enumerate(zip(attempts,raw['candidates'])):
            finite = bool(np.isfinite(paths[i,k]).all())
            assert finite == candidate['finite_xyz']
            if candidate['TipValid']:
                outcome='tip_valid'
            elif not finite:
                outcome='no_complete_path:'+attempt['status']
            else:
                failed=[name for name in ('semantic_goal_correct','tip_segments_clear','starts_at_current_state','event_state_sequence_correct','finite_event_values') if not candidate[name]]
                assert failed
                outcome='finite_invalid:'+','.join(failed)
            candidate_outcomes[outcome]+=1;search_statuses[attempt['status']]+=1
            per_slot.append(dict(slot=k,search_status=attempt['status'],outcome=outcome,candidate=candidate))
        outcomes=Counter(x['outcome'] for x in per_slot)
        joint='|'.join(key+'='+str(value) for key,value in sorted(outcomes.items()))
        joint_groups[joint]+=1
        op=opportunity(raw['candidates'],labels[identifier]['route_types'])
        case=dict(id=identifier,parent_id=raw['parent_id'],instruction=observations[identifier]['instruction'],
            metric=raw['metrics'],outcomes=dict(outcomes),per_slot=per_slot,opportunity=op,
            exact_grid_duplicate_slots=sum(a.get('exact_grid_path_duplicate',False) for a in attempts),
            request_seconds=raw['timing']['observation_to_checked_pool_seconds'])
        goal=generation.get('virtual_goal_attachments')
        if goal is not None:
            spec=labels[identifier]['semantic_targets'];endpoint=np.asarray(goal['exact_endpoint'])
            d=np.linalg.norm(np.asarray(spec['centers'])-endpoint,axis=1)
            eligible=[c for c in goal['candidates'] if c['within_grid'] and c['within_unchanged_contact_radius'] and c['original_grid_free']]
            case['posthoc_goal_attachment']=dict(endpoint_center_error_m=float(d[spec['target_index']]),
                inferred_endpoint_correct=bool(d.argmin()==spec['target_index'] and d[spec['target_index']]<=spec['tolerance']),
                original_free_local_candidates=len(eligible),
                finite_point_rejected=sum(not c['checks']['finite_point_cloud_segment_clear'] for c in eligible),
                ray_rejected=sum(not c['checks']['ray_check']['passed'] for c in eligible),
                diagnosis_only=True)
        cases.append(case)
    assert sum(candidate_outcomes.values()) == sum(search_statuses.values()) == 144
    valid=sum(c['metric']['TipValidAtK']*4 for c in cases)
    any_valid=sum(c['metric']['AnyTipValidAtK'] for c in cases)
    unique=sum(len(c['opportunity']['valid_predicted_types']) for c in cases)
    duplicate=sum(c['opportunity']['valid_classified_duplicate_slots'] for c in cases)
    unknown=sum(c['opportunity']['unknown_valid_slots'] for c in cases)
    bound=sum(c['opportunity']['duplicate_replacement_count_upper_bound'] for c in cases)
    for key,value in [('TipValidAtK',valid/144),('AnyTipValidAtK',any_valid/36),('UniqueClassifiedTipValidAtK',unique/36),('DuplicateClassifiedTipValidCount',duplicate/36),('UnknownTypeTipValidCount',unknown/36)]:
        assert np.isclose(report['metrics'][key],value)
    assert report['search_status_counts'] == dict(search_statuses)
    found=search_statuses.get('path_found',0)
    assert found == report['metrics']['raw_complete_paths']
    assert report['metrics']['failed_candidate_slots'] == 144-found
    job=read(p/'dev_model.status.json');assert job['status']=='completed' and job['exit_code']==0
    result=dict(protocol='two_row_astar_scaling_saved_pool_audit_v1',source_report_sha256=sha(p/'dev_model/report.json'),
        original_artifacts_verified=len(original_index),synced_artifacts_verified=len(sync['entries']),
        input_metadata_sha256={f:sha(metadata/f) for f in ('export_manifest.json','observations.jsonl','supervision.jsonl')},
        candidate_outcomes=dict(candidate_outcomes),condition_joint_outcomes=dict(joint_groups),search_statuses=dict(search_statuses),
        complete_paths=found,failed_generation_slots=144-found,finite_but_tip_invalid_slots=int(found-valid),
        total_tip_invalid_slots=int(144-valid),all_requested_dev_conditions=36,all_requested_candidate_slots=144,
        actual_metrics=report['metrics'],all_conditions=cases,
        duplicates=dict(valid_conditions=int(any_valid),known_unique_count=unique,classified_duplicate_count=duplicate,
            valid_unknown_count=unknown,known_duplicate_replacement_count_upper_bound=bound,
            counterfactual_known_unique_mean=(unique+bound)/36,
            conditions_with_positive_upper_bound=sum(c['opportunity']['duplicate_replacement_count_upper_bound']>0 for c in cases),
            valid_conditions_with_no_predicted_known_reference_match=sum(bool(c['opportunity']['valid_predicted_types']) and not (set(map(tuple,c['opportunity']['valid_predicted_types'])) & set(map(tuple,c['opportunity']['known_reference_types']))) for c in cases),
            warning='Privileged known-positive count bound, not generation or attainable gain. No invalid/unknown replacement, no complete-reference claim.'),
        actual_job=dict(code_commit=job['code_commit'],pid=job['pid'],child_pid=job['child_pid'],start_utc=job['start_utc'],end_utc=job['end_utc'],exit_code=0,
            wall_seconds=(datetime.fromisoformat(job['end_utc'])-datetime.fromisoformat(job['start_utc'])).total_seconds(),report_seconds=report['elapsed_seconds'],fit_seconds=report['fitting_seconds'],
            cpu_affinity=report['cpu_affinity'],threads=report['cpu_threads'],gpu_hours=0),
        local_analysis_script_sha256=sha(__file__),new_candidates_generated=0,new_model_calls=0,new_planner_calls=0,thresholds_changed=False)
    (p/'SAVED_POOL_ANALYSIS.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result,report


def main():
    results={};reports={}
    for prefix in ('44','76'):results[prefix],reports[prefix]=analyze(prefix)
    old=read(ROOT/'reports/observation_two_row_astar_v2/dev_model/report.json')
    assert old['planner_config']==reports['44']['planner_config']==reports['76']['planner_config']
    assert old['prototype_config']==reports['44']['prototype_config']==reports['76']['prototype_config']
    assert old['frozen_planner_source_integrity']==reports['44']['frozen_planner_source_integrity']==reports['76']['frozen_planner_source_integrity']
    ids=sorted(r['id'] for r in old['per_scene'])
    assert ids==sorted(r['id'] for r in reports['44']['per_scene'])==sorted(r['id'] for r in reports['76']['per_scene'])
    pairs=[]
    for identifier in ids:
        row=dict(id=identifier)
        for size,r in [('16',old),('32',reports['44']),('64',reports['76'])]:
            case=next(x for x in r['per_scene'] if x['id']==identifier)
            row[size]=dict(valid_slots=int(case['metrics']['TipValidAtK']*4),known_unique=case['metrics']['UniqueClassifiedTipValidAtK'],semantic=case['metrics']['semantic_goal_accuracy'],statuses=dict(Counter(a['status'] for a in case['generation']['attempts'])))
        pairs.append(row)
    combined=dict(protocol='fixed_astar16_32_64_saved_comparison_v1',same_planner_and_prototype_config=True,same_three_frozen_planner_sources=True,same36_dev_ids=True,
        comparisons=pairs,results={k:{name:v[name] for name in ('candidate_outcomes','condition_joint_outcomes','duplicates','actual_job')} for k,v in results.items()},
        note='TRAIN-fitted workspace/prototypes change together. No causal decomposition or new planning. All raw failed slots retained.')
    (Path(__file__).parent/'SCALING_COMPARISON.json').write_text(json.dumps(combined,indent=2)+'\n')
    print(json.dumps(combined['results'],indent=2))


if __name__=='__main__':main()
