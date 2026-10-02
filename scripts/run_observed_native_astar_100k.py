"""Three separately launched stages for the sole100k-node native budget check.

Tests/build -> TRAIN12 both arms -> DEV36 both arms. No Python search fallback,
parameter sweep, candidate replacement, source edits, or automatic next stage.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET

import numpy as np

from scripts import observation_native_astar_100k as native
from scripts import observation_spatial_penalty_astar as spatial

PROTOCOL = 'observed_two_row_native_100k_paired_runner_v1'
SOURCE = Path(__file__).resolve().parents[1]
EXPORT_SHA = '6ed7786823275f26dba38fff9039bd33127b71567de4e4c0c80a37cfd91e46f9'
FIT_SHA = '384bc02b2d0237dfa7a1331257502a6858db1b5df62e62d463010ae8921d6b1b'
TEST_FILES = ('tests/test_observation_native_astar.py', 'tests/test_observed_native_runner.py',
              'tests/test_observation_spatial_penalty_astar.py', 'tests/test_two_row_astar_control.py',
              'tests/test_spatial_penalty_saved_analysis.py',
              'tests/test_observation_native_astar_100k.py','tests/test_observed_native_runner_100k.py')


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    spatial.control.write_json(path, value)


def config_guard(config, old):
    expected = dict(protocol=native.PROTOCOL, candidates=4, horizon=24, voxel_m=.025,
        grid_edge_penalty_multiplier=4, spatial_sigma_m=.05, maximum_expanded_nodes_per_candidate=100000,
        search_deadline_seconds=2., clock_check_interval_expanded_nodes=64,
        ctypes_preparation_inside_deadline=True, spatial_field_preprocessing_inside_astar_deadline=False,
        all_preprocessing_in_request_walltime=True, arms=['edge','spatial'],
        train_preflight_indices=list(range(4)), dev_indices=list(range(64,76)), targets_per_parent=3,
        registered_train_fit_parents=32, actual_train_fit_parents=31, actual_train_fit_inputs=93,
        export_manifest_sha256=EXPORT_SHA, shared_fit_sha256_excluding_only_training_seconds=FIT_SHA,
        train_preflight_requests=24, train_preflight_slots=96, dev_requests=72, dev_slots=288,
        new_total_request_upper_bound=96, new_total_slot_upper_bound=384,
        historical_maximum_nodes=20000, node_budget_multiplier=5,
        training_updates=0, qwen_calls=0, simulator_calls=0)
    require(all(config.get(k) == v for k,v in expected.items()), 'Registered native experiment changed')
    require(config['compile_flags'] == native.FLAGS+['-fPIC'], 'Linux build flags changed')
    require(old['protocol'] == spatial.PROTOCOL and old['export_manifest_sha256'] == EXPORT_SHA and
            old['original_shared_fit_sha256'] == FIT_SHA, 'Original spatial registration changed')


def execution_context(commit):
    require(re.fullmatch('[0-9a-f]{40}',commit) and SOURCE.name == commit and
            SOURCE.parent.name == 'releases' and SOURCE.parent.parent.name == 'research_v2',
            'Execute only an immutable registered release')
    require(os.name == 'posix' and os.environ.get('CODE_COMMIT') == commit and
            len(os.sched_getaffinity(0)) == 1, 'Actual Linux source commit /one CPU affinity required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') in ('','-1') and
            all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS')),
            'GPU hidden and all numerical libraries one thread required')
    project = SOURCE.parents[2]
    return project, project/'runs/observed_two_row_native_astar_100k_v1', project/'research_v2/native_builds'/commit


def source_inventory():
    names = [Path(__file__).relative_to(SOURCE).as_posix(), 'scripts/launch_observed_two_row_native_astar_100k_v1.sh',
        'scripts/observation_native_astar.py', 'scripts/observation_native_astar_100k.py',
        'routeset/native/observed_astar.cpp', 'scripts/run_observed_native_astar.py',
        'configs/observed_two_row_native_astar_100k_v1.json', 'configs/observed_two_row_spatial_penalty_v1.json',
        *TEST_FILES, 'scripts/observation_spatial_penalty_astar.py', 'scripts/evaluate_two_row_astar_v2.py',
        'scripts/export_two_row_observations.py', 'scripts/evaluate_observed_two_row.py',
        'scripts/evaluate_observed_obstacles.py', 'scripts/collect_observed_two_row_pilot.py',
        'scripts/collect_obstacle_reach.py', 'scripts/collect_two_row_formal.py',
        'scripts/snapshot_multitask_observations.py', 'scripts/record_job.py',
        'scripts/analyze_observed_spatial_penalty_astar.py']
    names += ['scripts/'+name for name in spatial.control.FROZEN_SOURCES]
    return {str(SOURCE/name): native.digest(SOURCE/name) for name in names}


def unchanged(hashes):
    require(all(native.digest(path) == value for path,value in hashes.items()), 'Frozen source/input changed')


def artifacts(output):
    return {str(p):native.digest(p) for p in sorted(Path(output).rglob('*')) if p.is_file()}


def junit_passed(path):
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == 'testsuite' else list(root.iter('testsuite'))
    counts = {k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
    require(counts['tests'] >= 24 and counts['errors'] == counts['failures'] == counts['skipped'] == 0,
            'At least24 actual native differential tests required; no skips or failed tests')
    cases = list(root.iter('testcase'))
    native_ids = sorted(c.get('classname','')+'::'+c.get('name','') for c in cases
        if c.get('classname','').split('.')[-1] == 'test_observation_native_astar')
    require(len(cases) == counts['tests'] and len(native_ids) == len(set(native_ids)) == 24 and
            not any(list(c) for c in cases if any(x.tag in ('failure','error','skipped') for x in c)),
            'Exactly24 distinct actual native test cases required; other test files cannot replace them')
    budget_ids = sorted(c.get('classname','')+'::'+c.get('name','') for c in cases
        if c.get('classname','').split('.')[-1] == 'test_observation_native_astar_100k')
    require(len(budget_ids) == len(set(budget_ids)) == 12,
        'Exactly12 distinct actual100k budget/differential cases required')
    require(any('actual100000_node_cap' in x for x in budget_ids), 'Actual100k boundary test required')
    counts.update(native_test_count=24,native_test_ids=native_ids,
        native100k_test_count=12,native100k_test_ids=budget_ids)
    return counts


def same_build(left, right):
    keys = ('protocol','cpp_sha256','adapter_sha256','compiler_path','compiler_sha256','compiler_version_sha256',
            'flags','flags_sha256','system','machine')
    require(left['exit_code'] == right['exit_code'] == 0 and all(left[k] == right[k] for k in keys),
            'Test and production compiler/source/flags differ')


def find_actual_test_build(folder, fixture_name):
    require(fixture_name in ('native_astar','native_astar100k'), 'Only two actual compiler fixtures accepted')
    required = {'protocol','test_only_build','exit_code','cpp_sha256','adapter_sha256','compiler_path',
        'compiler_sha256','compiler_version_sha256','flags','flags_sha256','system','machine','library_path','library_sha256'}
    candidates=[]
    # Each session compiler fixture owns only direct <fixture_name><N>/build.
    # Nested gate tests deliberately create similarly shaped fake receipts;
    # do not interpret their fixture payloads as extra compiler executions.
    paths=[child/'build/build_receipt.json' for child in Path(folder).iterdir()
           if re.fullmatch(re.escape(fixture_name)+r'[0-9]+',child.name) and child.is_dir() and not child.is_symlink()]
    for path in paths:
        require(path.is_file(), 'Actual native compiler fixture lacks its receipt')
        value=read(path)
        if value.get('protocol') != native.ABI_PROTOCOL or value.get('test_only_build') is not True:
            continue  # Gate unit-test fixtures are not actual compiler receipts.
        require(required.issubset(value) and value['exit_code']==0, 'Actual test compiler receipt incomplete or failed')
        candidates.append(path)
    require(len(candidates)==1, 'Exactly one actual differential-test compiler build required')
    return candidates[0]


def gate_identity(report, stage, commit, sources, build):
    require(report.get('protocol') == PROTOCOL and report.get('status') == 'completed' and
            report.get('stage') == stage and report.get('source_commit') == commit and
            report.get('source_files_sha256') == sources and
            report.get('effective_search_config') == native.EFFECTIVE_SEARCH_CONFIG, 'Stage source/protocol/commit identity differs')
    require(report['build_receipt_sha256'] == native.digest(build/'build_receipt.json') and
            report['library_sha256'] == native.digest(read(build/'build_receipt.json')['library_path']),
            'Stage compiler/binary receipt differs')
    if stage == 'tests':
        require(report.get('actual_linux_differential_tests_passed') is True, 'Actual server differential test gate required')
    elif stage == 'train_preflight':
        require(report.get('mechanical_preflight_passed') is True and report['completed_requests'] == 24 and
                report['requested_candidate_slots'] == 96 and report['shared_fit_sha256'] == FIT_SHA,
                'Full TRAIN mechanical gate required')
    unchanged(report['artifact_sha256'])


def completed_stage(run, stage, commit, sources, build):
    status = read(run/(stage+'.status.json'))
    require(status.get('status') == 'completed' and status.get('exit_code') == 0 and
            status.get('code_commit') == commit, 'Previous recorded stage did not exit successfully')
    path = run/stage/'report.json'
    report = read(path)
    gate_identity(report,stage,commit,sources,build)
    return report, dict(path=str(path),sha256=native.digest(path),status_sha256=native.digest(run/(stage+'.status.json')))


def run_tests(output, build, commit, sources):
    output.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    command = [sys.executable,'-m','pytest',*[str(SOURCE/f) for f in TEST_FILES],'-q','-p','no:cacheprovider',
               '--junitxml='+str(output/'pytest.xml'),'--basetemp='+str(output/'pytest_tmp')]
    result = subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (output/'pytest.log').write_text(result.stdout)
    write(output/'test_process.json',dict(command=command,exit_code=result.returncode,elapsed_seconds=time.perf_counter()-started))
    require(result.returncode == 0,'Actual differential tests failed; preserve logs, no compile/launch continuation')
    counts = junit_passed(output/'pytest.xml')
    test_receipt = find_actual_test_build(output/'pytest_tmp','native_astar100k')
    old_test_receipt = find_actual_test_build(output/'pytest_tmp','native_astar')
    old_test_library = native.NativeLibrary(old_test_receipt,for_testing=True)
    test_library = native.NativeLibrary(test_receipt,for_testing=True)
    test_build = test_library.receipt
    require(test_build['test_only_build'] is True and test_build['system'] == 'Linux', 'Actual Linux test build required')
    compile_started = time.perf_counter();existed = build.exists()
    built = native.build_library(build,source_commit=commit)
    same_build(test_build,built)
    same_build(old_test_library.receipt,built)
    library = native.NativeLibrary(build/'build_receipt.json')
    write(output/'production_build_receipt.json',built)
    unchanged(sources)
    report = dict(protocol=PROTOCOL,status='completed',stage='tests',source_commit=commit,source_files_sha256=sources,
        actual_linux_differential_tests_passed=True,pytest=counts,test_command=command,
        effective_search_config=dict(native.EFFECTIVE_SEARCH_CONFIG),
        old20k_test_build_receipt_path=str(old_test_receipt),old20k_test_build_receipt_sha256=native.digest(old_test_receipt),
        test_build_receipt_path=str(test_receipt),test_build_receipt_sha256=native.digest(test_receipt),
        build_receipt_path=str(build/'build_receipt.json'),build_receipt_sha256=library.receipt_sha256,
        library_sha256=built['library_sha256'],production_build_reused=existed,
        production_build_check_or_compile_seconds=time.perf_counter()-compile_started,
        production_compile_seconds_new=0. if existed else built['elapsed_seconds'],
        test_and_build_elapsed_seconds=time.perf_counter()-started,artifact_sha256=artifacts(output),
        new_candidate_slots=0,qwen_calls=0,simulator_calls=0,training_steps=0,gpu_hours=0)
    write(output/'report.json',report)
    return report


def check_first_pair(output, identifier):
    with np.load(output/'edge'/identifier/'predictions.npz',allow_pickle=False) as left, np.load(output/'spatial'/identifier/'predictions.npz',allow_pickle=False) as right:
        h24 = bool(np.array_equal(left['paths'][0,0],right['paths'][0,0],equal_nan=True))
    with np.load(output/'edge'/identifier/'raw_paths.npz',allow_pickle=False) as left, np.load(output/'spatial'/identifier/'raw_paths.npz',allow_pickle=False) as right:
        raw = bool(np.array_equal(left['candidate_0'],right['candidate_0'],equal_nan=True))
    require(h24 and raw,'Same native zero-field first path differs; preserve failure and stop')
    return dict(id=identifier,h24_equal=h24,raw_equal=raw)


def run_pair(project, run, output, build, commit, stage, sources, config, old):
    tests,test_link = completed_stage(run,'tests',commit,sources,build)
    preflight_link = None
    if stage == 'dev':
        _,preflight_link = completed_stage(run,'train_preflight',commit,sources,build)
    started = time.perf_counter()
    data = project/old['data_relative_to_project']
    manifest,gate = spatial.verify_export_metadata(data,old)
    observations = [json.loads(line) for line in (data/'observations.jsonl').read_text().splitlines()]
    selected = spatial.validate_registration(old,manifest,observations,stage)
    planner = spatial.control.dependencies()
    model,prior,fit = spatial.fit_train(data,observations,manifest,planner)
    require(fit['observed_train_inputs'] == 93 and len(fit['parents']) == 31 and
            spatial.fit_identity(model,prior) == FIT_SHA,'Same fixed31-parent TRAIN fit required')
    library = native.NativeLibrary(build/'build_receipt.json')
    output.mkdir(parents=True,exist_ok=False)
    write(output/'fitted_train_model.json',dict(prototype=model,workspace=prior,fit=fit,canonical_sha256=FIT_SHA,
        canonical_excludes_only='prototype.training_seconds'))
    write(output/'production_build_receipt.json',library.receipt)
    rows,pools,timing = ({a:[] for a in ('edge','spatial')} for _ in range(3))
    input_hashes=dict(fit['source_files_sha256'])
    input_hashes[str(data/'export_manifest.json')]=native.digest(data/'export_manifest.json')
    first=[];requested=len(selected)*2;interrupted=None
    try:
        for row in selected:
            for arm in ('edge','spatial'):
                interrupted=dict(id=row['id'],arm=arm,reserved_candidate_slots=4)
                request_start=time.perf_counter()
                with native.native_planner(planner,library):
                    result,paths,events=spatial.one_request(data,row,manifest,model,prior,planner,output/arm/row['id'],arm)
                duration=time.perf_counter()-request_start
                # Keep old result.json byte-identical to one_request's receipt.
                wrapper=dict(id=row['id'],parent_id=row['parent_id'],split=row['split'],arm=arm,
                    native_context_through_checked_pool_seconds=duration,
                    inner_observation_to_checked_pool_seconds=result['timing']['observation_to_checked_pool_seconds'],
                    includes_neighbor_table_setup_and_context_exit=True,build_receipt_sha256=library.receipt_sha256,
                    library_sha256=library.receipt['library_sha256'],
                    base_planner_config_historical=dict(planner.CONFIG),
                    effective_search_config=dict(native.EFFECTIVE_SEARCH_CONFIG),
                    historical_node_cap_is_not_executed=True)
                require(duration>=wrapper['inner_observation_to_checked_pool_seconds'],'Incomplete request clock')
                write(output/arm/row['id']/'native_request_receipt.json',wrapper)
                for attempt in result['generation']['attempts']:
                    if attempt['status'] in ('localization_failure','grid_construction_rejected'):
                        require(attempt['complete_raw_paths_emitted']==0 and 'native' not in attempt,
                            'Pre-search failure cannot contain an emitted route')
                    else:
                        require('native' in attempt,'Actual search must use native backend; no Python fallback')
                        require(attempt['native']['test_only'] is False and
                            attempt['native']['protocol']==native.PROTOCOL and
                            attempt['native']['effective_search_config']==native.EFFECTIVE_SEARCH_CONFIG and
                            attempt['native']['budget_adapter_sha256']==native.digest(native.__file__) and attempt['maximum_expanded_nodes']==100000 and
                            attempt['search_deadline_seconds']==2. and attempt['native']['clock_check_interval_expanded_nodes']==64 and
                            attempt['native']['library_sha256']==library.receipt['library_sha256'], 'Production native search drift')
                rows[arm].append(result);pools[arm].append((paths,events));timing[arm].append(wrapper)
                input_hashes.update(result['source_files_sha256']);interrupted=None
                print(json.dumps(dict(stage=stage,arm=arm,id=row['id'],failed_slots=result['generation']['failed_slots'],
                    native_request_seconds=duration)),flush=True)
            first.append(check_first_pair(output,row['id']))
        for arm,pool in pools.items():
            np.savez_compressed(output/(arm+'_predictions.npz'),paths=np.stack([p[0] for p in pool]),
                gripper_open=np.stack([p[1] for p in pool]),scene_ids=[r['id'] for r in selected],
                parent_ids=[r['parent_id'] for r in selected])
        _,final_gate=spatial.verify_export_metadata(data,old)
        unchanged(sources);unchanged(input_hashes)
        require(native.digest(build/'build_receipt.json')==library.receipt_sha256 and
            native.digest(library.receipt['library_path'])==library.receipt['library_sha256'],'Native build changed during run')
        metrics={arm:spatial.aggregate(value) for arm,value in rows.items()}
        for arm in metrics:
            durations=[x['native_context_through_checked_pool_seconds'] for x in timing[arm]]
            metrics[arm]['native_full_request_latency_seconds']=dict(first=durations[0],median=float(np.median(durations)),
                p95=float(np.quantile(durations,.95)),total=sum(durations))
        report=dict(protocol=PROTOCOL,status='completed',stage=stage,source_commit=commit,source_files_sha256=sources,
            input_source_files_sha256=input_hashes,config=config,original_spatial_config=old,base_planner_config_historical=planner.CONFIG,
            effective_search_config=dict(native.EFFECTIVE_SEARCH_CONFIG),
            prototype_config=planner.prototype.CONFIG,source_export_manifest_sha256=EXPORT_SHA,shared_fit_sha256=FIT_SHA,
            train_fit=fit,registered_train_parents=32,actual_train_parents=31,requested_inputs_per_arm=len(selected),
            input_ids=[r['id'] for r in selected],requested_candidate_slots=requested*4,completed_requests=requested,
            failed_requests=0,unattempted_requests=0,results=metrics,per_request=rows,native_request_timing=timing,
            first_path_equal=first,mechanical_preflight_passed=(stage=='train_preflight' and len(first)==12),
            tests_gate=test_link,train_preflight_gate=preflight_link,
            build_receipt_path=str(build/'build_receipt.json'),build_receipt_sha256=library.receipt_sha256,
            library_sha256=library.receipt['library_sha256'],new_compile_seconds=0.,
            initial_mechanical_gate=gate,final_mechanical_gate=final_gate,elapsed_seconds=time.perf_counter()-started,
            training_steps=0,new_qwen_calls=0,new_simulator_calls=0,gpu_hours=0,artifact_sha256=artifacts(output),
            scope='Both arms get100k nodes (5x the historical20k), with unchanged2s/64 checks; no equal-node or end-to-end-time claim. Both arms use the same production native kernel, original observation-derived grid and endpoint attachment, and all K4 slots. Verification geometry is read only after sealing predictions. No Python fallback. Field and ctypes costs remain measured; no automatic next stage.')
        write(output/'report.json',report)
        return report
    except Exception:
        completed=sum(len(value) for value in rows.values())
        write(output/'failure.json',dict(protocol=PROTOCOL,status='failed',stage=stage,traceback=traceback.format_exc(),
            requested_requests=requested,reserved_candidate_slots=requested*4,completed_requests=completed,
            interrupted_request=interrupted,interrupted_request_upper_bound_slots=4 if interrupted else 0,
            later_unattempted_requests=max(0,requested-completed-int(interrupted is not None)),automatic_retries=0))
        raise


def main(commit, stage):
    project,run,build=execution_context(commit)
    config=read(SOURCE/'configs/observed_two_row_native_astar_100k_v1.json')
    old=read(SOURCE/'configs/observed_two_row_spatial_penalty_v1.json');config_guard(config,old)
    output=run/stage
    require(not output.exists(),'Fresh stage required; preserve existing artifacts and do not replay')
    sources=source_inventory()
    return run_tests(output,build,commit,sources) if stage=='tests' else run_pair(
        project,run,output,build,commit,stage,sources,config,old)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-commit',required=True)
    parser.add_argument('--stage',choices=('tests','train_preflight','dev'),required=True)
    args=parser.parse_args();result=main(args.source_commit,args.stage)
    print(json.dumps({k:result[k] for k in ('protocol','status','stage')}),flush=True)
