"""Local sealed-pool audit; no raw observations, labels, planner or model calls."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
from scripts import analyze_observed_spatial_penalty_astar as audit

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).parent
BINARY=ROOT/'runs/observed_two_row_spatial_penalty_v1'
OLD=ROOT/'reports/observation_two_row_astar_prefix44_v1/dev_model'
OLD_BINARY=ROOT/'runs/observation_two_row_astar_prefix44_v1/dev_model'


def run():
    report=audit.read(BASE/'dev/report.json');old=audit.read(OLD/'report.json')
    assert report['status']=='completed' and report['stage']=='dev'
    assert report['completed_requests']==36 and report['requested_candidate_slots']==144
    assert report['failed_requests']==report['unattempted_requests']==0
    assert report['prior_preflight_sha256']==audit.sha(BASE/'train_preflight/report.json')
    assert report['config']['original_dev_report_sha256']==audit.sha(OLD/'report.json')
    verified=audit.verify_artifacts(report,BASE/'dev',BINARY/'dev')
    fit=audit.read(BASE/'dev/fitted_train_model.json')
    assert fit['canonical_sha256']==report['shared_fit_sha256']==report['config']['original_shared_fit_sha256']
    assert report['planner_config']==old['planner_config'] and report['prototype_config']==old['prototype_config']
    assert report['config']['export_manifest_sha256']==old['source_export_manifest_sha256']
    expected=sorted('two_row_reach_%d_target%d'%(p,t) for p in range(283264,283276) for t in range(3))
    assert sorted(report['input_ids'])==expected and len(report['input_ids'])==36
    assert audit.sha(OLD_BINARY/'predictions.npz')==old['prediction_sha256']
    output={};arrays={};raw_arrays={}
    for arm,rows,meta,binary,pool_name in [
        ('edge',old['per_scene'],OLD,OLD_BINARY,'predictions.npz'),
        ('spatial',report['per_request']['spatial'],BASE/'dev',BINARY/'dev','spatial_predictions.npz')]:
        assert len(rows)==36 and sorted(r['id'] for r in rows)==expected
        with np.load(binary/pool_name,allow_pickle=False) as pool:
            ids=pool['scene_ids'].astype(str).tolist();parents=pool['parent_ids'].astype(str).tolist()
            paths=pool['paths'].copy();opened=pool['gripper_open'].copy()
        assert sorted(ids)==expected and paths.shape==(36,4,24,3) and opened.shape==(36,4,24)
        all_cases=[];outcomes=Counter();statuses=Counter();timeouts=[];raw_arrays[arm]={}
        for row in rows:
            identifier=row['id'];i=ids.index(identifier)
            parent=identifier.rsplit('_target',1)[0]
            assert row['parent_id']==parents[i]==parent and row['split']=='DEV_MODEL'
            relative=Path('requests')/identifier if arm=='edge' else Path('spatial')/identifier
            q,b=meta/relative,binary/relative
            assert audit.read(q/'result.json')==row
            seal=audit.read(q/'generation_seal.json')
            assert seal['evaluation_labels_opened'] is False and seal['submitted_candidate_budget']==4
            for key,name in [('prediction_sha256','predictions.npz'),('raw_paths_sha256','raw_paths.npz')]:
                assert row[key]==seal[key]==audit.sha(b/name)
            if arm=='spatial':
                for name in ('predictions.npz','raw_paths.npz','result.json','generation_seal.json','status.json'):
                    assert (relative/name).as_posix() in verified
            with np.load(b/'predictions.npz',allow_pickle=False) as request:
                assert request['paths'].shape==(1,4,24,3) and request['gripper_open'].shape==(1,4,24)
                assert request['scene_ids'].tolist()==[identifier] and request['parent_ids'].tolist()==[parent]
                np.testing.assert_array_equal(paths[i],request['paths'][0]);np.testing.assert_array_equal(opened[i],request['gripper_open'][0])
            with np.load(b/'raw_paths.npz',allow_pickle=False) as raw:
                assert set(raw.files)=={'candidate_%d'%k for k in range(4)}
                raw_paths=[raw['candidate_%d'%k].copy() for k in range(4)]
            raw_arrays[arm][identifier]=raw_paths[0]
            generation=row['generation'];attempts=generation['attempts']
            detail=audit.summarize_candidates(row['candidates'],paths[i],opened[i],raw_paths,attempts)
            assert detail['raw_complete']==generation['raw_complete_paths']==seal['raw_complete_paths']
            assert detail['failed_generation']==generation['failed_slots']
            outcomes.update(detail['outcomes']);statuses.update(detail['search_statuses'])
            for attempt in attempts:
                if attempt['status']=='search_time_budget_exhausted':
                    timeouts.append(dict(id=identifier,slot=attempt['slot'],expanded_nodes=attempt['expanded_nodes'],
                        search_seconds=attempt['search_seconds'],neighbor_edges_examined=attempt['neighbor_edges_examined'],
                        field_seconds=attempt.get('spatial_penalty',{}).get('preprocessing_seconds',0.)))
            for key,value in [('TipValidAtK',detail['valid']/4),('AnyTipValidAtK',float(detail['valid']>0)),
                              ('UniqueClassifiedTipValidAtK',detail['known_unique']),('UnknownTypeTipValidCount',detail['unknown_valid']),
                              ('DuplicateClassifiedTipValidCount',detail['known_duplicate'])]:
                audit.equal_number(row['metrics'][key],value,'Per-request '+key)
            field=sum(a.get('spatial_penalty',{}).get('preprocessing_seconds',0.) for a in attempts)
            if arm=='spatial':
                audit.equal_number(field,generation['spatial_adapter']['preprocessing_seconds'],'Field cost')
                count=calls=0
                for attempt in attempts:
                    if 'spatial_penalty' in attempt:
                        assert attempt['spatial_penalty']['previous_returned_raw_paths']==count
                        assert attempt['spatial_penalty']['all_returned_paths_included_without_postcheck_filter']
                        calls+=1
                    count+=attempt['complete_raw_paths_emitted']
                assert count==generation['spatial_adapter']['previous_complete_raw_paths']
                assert calls==generation['spatial_adapter']['actual_astar_calls']<=4
            assert 0<=field<=row['timing']['observation_to_checked_pool_seconds']
            all_cases.append(dict(id=identifier,parent_id=parent,metrics=row['metrics'],**detail,
                field_seconds=field,request_seconds=row['timing']['observation_to_checked_pool_seconds']))
        metrics=old['metrics'] if arm=='edge' else report['results']['spatial']
        for key in audit.METRICS:
            values=[r['metrics'][key] for r in all_cases if r['metrics'][key] is not None]
            audit.equal_number(metrics[key],np.mean(values) if values else None,'Aggregate '+key)
        output[arm]=dict(cases=all_cases,totals={key:sum(c[key] for c in all_cases) for key in
            ('valid','known_unique','known_duplicate','unknown_valid','raw_complete','failed_generation','finite_invalid')},
            outcomes=dict(outcomes),search_statuses=dict(statuses),timeouts=timeouts,
            field_seconds=sum(c['field_seconds'] for c in all_cases),request_seconds=sum(c['request_seconds'] for c in all_cases),
            metrics=metrics)
        arrays[arm]={identifier:paths[ids.index(identifier),0] for identifier in expected}
    first=[dict(id=i,h24_equal=bool(np.array_equal(arrays['edge'][i],arrays['spatial'][i],equal_nan=True)),
        raw_equal=bool(np.array_equal(raw_arrays['edge'][i],raw_arrays['spatial'][i],equal_nan=True))) for i in expected]
    result=dict(protocol='observed_spatial_penalty_saved_dev_pair_v1',source_report_sha256=audit.sha(BASE/'dev/report.json'),
        original_edge_report_sha256=audit.sha(OLD/'report.json'),verified_artifacts=verified,
        arms=output,all_pairs=audit.paired_cases(output['edge'],output['spatial']),first_path_comparison=first,
        requested_conditions=36,submitted_slots_per_arm=144,new_generation_calls=0,new_raw_labels_opened=False,
        analysis_source_sha256=audit.sha(__file__),shared_helper_sha256=audit.sha(audit.__file__),
        limitation='Existing edge DEV reused after exact source/data/config/fit identity. Fixed K, different actual CPU time; saved checker flags audited, no fresh geometric evaluation.')
    path=BASE/'DEV_PAIRED_ANALYSIS.json';assert not path.exists()
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({a:{k:v[k] for k in ('totals','outcomes','field_seconds','request_seconds')} for a,v in output.items()}))


if __name__=='__main__':run()
