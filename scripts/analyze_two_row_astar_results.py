"""Read saved A* artifacts only; no planning, model forward or threshold tuning."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_two_row_ordinary import opportunity


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def analyze(run,metadata,output):
    run,metadata,output=map(Path,(run,metadata,output))
    if output.exists():raise FileExistsError('Preserve existing diagnostic')
    report=json.loads((run/'report.json').read_text());index=json.loads((run/'artifact_index.json').read_text())
    manifest=json.loads((metadata/'export_manifest.json').read_text())
    assert sha(metadata/'export_manifest.json')==report['source_export_manifest_sha256']
    for name in ('observations.jsonl','supervision.jsonl'):
        assert sha(metadata/name)==manifest['output_files_sha256'][name]
    for name,entry in index.items():
        assert sha(run/name)==entry['sha256'] and (run/name).stat().st_size==entry['bytes'],name
    obs={r['id']:r for r in read_rows(metadata/'observations.jsonl') if r['split']=='DEV_MODEL'}
    labels={r['id']:r for r in read_rows(metadata/'supervision.jsonl') if r['split']=='DEV_MODEL'}
    assert len(obs)==len(labels)==36 and obs.keys()==labels.keys()
    with np.load(run/'predictions.npz',allow_pickle=False) as a:
        ids=list(map(str,a['scene_ids']));paths=a['paths'].copy();events=a['gripper_open'].copy()
    assert set(ids)==set(obs) and len(ids)==36 and paths.shape==(36,4,24,3) and events.shape==(36,4,24)
    rows=[];groups=Counter();attachments=[]
    for raw in report['per_scene']:
        identifier=raw['id'];label=labels[identifier];position=ids.index(identifier)
        request=run/'requests'/identifier
        saved=json.loads((request/'result.json').read_text());assert saved==raw
        seal=json.loads((request/'generation_seal.json').read_text())
        assert seal['prediction_sha256']==raw['prediction_sha256']==sha(request/'predictions.npz')
        assert seal['raw_paths_sha256']==raw['raw_paths_sha256']==sha(request/'raw_paths.npz')
        with np.load(request/'predictions.npz',allow_pickle=False) as a:
            np.testing.assert_array_equal(paths[position],a['paths'][0]);np.testing.assert_array_equal(events[position],a['gripper_open'][0])
        assert sum(c['finite_xyz'] for c in raw['candidates'])==np.isfinite(paths[position]).all((1,2)).sum()
        record=raw['generation'];statuses=Counter(a['status'] for a in record['attempts'])
        if record['localization']['status']=='unsupported_exact_instruction':kind='unsupported_exact_instruction'
        elif set(statuses)=={'no_admissible_virtual_goal_attachment'}:kind='goal_attachment_failure'
        elif set(statuses)=={'path_found'} and raw['metrics']['TipValidAtK']==1.:kind='all_four_tip_valid'
        else:raise ValueError('Unregistered failure grouping; inspect without silently dropping: '+identifier)
        groups[kind]+=1
        gap=opportunity(raw['candidates'],label['route_types'])
        row=dict(id=identifier,parent_id=raw['parent_id'],instruction=obs[identifier]['instruction'],group=kind,
            actual_metric=raw['metrics'],reference_count=len(label['routes']),opportunity=gap,
            exact_grid_duplicates=sum(a.get('exact_grid_path_duplicate',False) for a in record['attempts']),
            request_seconds=raw['timing']['observation_to_checked_pool_seconds'],search_status_counts=dict(statuses))
        if kind=='goal_attachment_failure':
            target=record['virtual_goal_attachments'];endpoint=np.asarray(target['exact_endpoint'])
            specification=label['semantic_targets'];distance=np.linalg.norm(np.asarray(specification['centers'])-endpoint,axis=1)
            eligible=[a for a in target['candidates'] if a['within_grid'] and a['within_unchanged_contact_radius'] and a['original_grid_free']]
            detail=dict(id=identifier,eligible_original_free_attachments=len(eligible),
                total_considered_attachments=len(target['candidates']),
                finite_point_rejected=sum(not a['checks']['finite_point_cloud_segment_clear'] for a in eligible),
                ray_rejected=sum(not a['checks']['ray_check']['passed'] for a in eligible),
                goal_center_error_m=float(distance[specification['target_index']]),
                inferred_endpoint_semantic_correct=bool(distance.argmin()==specification['target_index'] and distance[specification['target_index']]<=specification['tolerance']),
                label_scope='Post-generation diagnosis only, no target used to rerun or repair attachment.')
            row['goal_attachment_diagnosis']=detail;attachments.append(detail)
        rows.append(row)
    assert len(rows)==36 and len({r['id'] for r in rows})==36
    valid=[r for r in rows if r['group']=='all_four_tip_valid']
    addition=sum(r['opportunity']['duplicate_replacement_count_upper_bound'] for r in valid)
    unknown=sum(r['opportunity']['unknown_valid_slots'] for r in valid)
    duplicate=sum(r['opportunity']['valid_classified_duplicate_slots'] for r in valid)
    metric=report['metrics']
    assert (len(valid),unknown,duplicate,addition)==(19,10,47,43)
    assert np.isclose(metric['UniqueClassifiedTipValidAtK'],sum(len(r['opportunity']['valid_predicted_types']) for r in rows)/36)
    assert all(r['opportunity']['valid_predicted_types']==[('middle','middle')] for r in valid)
    unsupported=Counter(r['instruction'] for r in rows if r['group']=='unsupported_exact_instruction')
    found_unseen=sum(not (set(map(tuple,r['opportunity']['valid_predicted_types'])) & set(map(tuple,r['opportunity']['known_reference_types']))) for r in valid)
    corrected=json.loads((run.parent/'targeted_tests_corrected.status.json').read_text())
    job=json.loads((run.parent/'dev_model.status.json').read_text())
    wall=(datetime.fromisoformat(job['end_utc'])-datetime.fromisoformat(job['start_utc'])).total_seconds()
    result=dict(protocol='two_row_astar_saved_failure_and_duplicate_audit_v1',groups=dict(groups),metrics=metric,
        source_files_sha256={str(run/'report.json'):sha(run/'report.json'),str(run/'artifact_index.json'):sha(run/'artifact_index.json'),
            **{str(metadata/name):sha(metadata/name) for name in ('observations.jsonl','supervision.jsonl','export_manifest.json')},
            str(Path(__file__)):sha(__file__),str(Path(__file__).with_name('analyze_two_row_ordinary.py')):sha(Path(__file__).with_name('analyze_two_row_ordinary.py'))},
        verified_actual_artifacts=len(index),verified_condition_pools=36,submitted_slots=144,
        unsupported_exact_instruction_conditions=dict(unsupported),goal_attachment_failures=attachments,
        duplicate_diagnostic=dict(valid_conditions=len(valid),conditions_with_positive_duplicate_count_upper_bound=sum(r['opportunity']['duplicate_replacement_count_upper_bound']>0 for r in valid),
            valid_classified_duplicate_slots=duplicate,valid_unknown_slots_retained=unknown,
            exact_grid_duplicate_slots=sum(r['exact_grid_duplicates'] for r in rows),
            valid_conditions_with_predicted_type_absent_from_known_reference=found_unseen,
            replacement_count_upper_bound_sum=addition,existing_classified_types_sum=19,
            counterfactual_total_classified_types=19+addition,counterfactual_unique_mean_over_all36=(19+addition)/36,
            counterfactual_is_not_actual_prediction=True,
            definition='For each already valid pool, min(duplicate classified valid slots, missing known positive types); invalid and unknown slots untouched. Privileged reference-count diagnosis, no retrieval/replacement executed.',
            warning='No claim of attainable gain under the observation/search budget; known reference types are incomplete and may miss valid observed planner types.'),
        all_conditions=rows,actual_job=dict(code_commit=job['code_commit'],pid=job['pid'],child_pid=job['child_pid'],exit_code=job['exit_code'],
            wall_seconds=wall,fit_seconds=report['fitting_seconds'],report_seconds=report['elapsed_seconds'],
            cpu_affinity=report['cpu_affinity'],cpu_threads=report['cpu_threads'],gpu_hours=0),
        corrected_test_job=corrected,new_candidates_generated=0,new_model_forwards=0,
        thresholds_changed=False,source_report_modified=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(groups=result['groups'],duplicate_diagnostic=result['duplicate_diagnostic'],actual_job=result['actual_job']),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','metadata','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();analyze(a.run,a.metadata,a.output)
