from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_observed_native_astar as runner


def config_pair():
    root=Path(__file__).resolve().parents[1]
    return (json.loads((root/'configs/observed_two_row_native_astar_v1.json').read_text()),
            json.loads((root/'configs/observed_two_row_spatial_penalty_v1.json').read_text()))


def junit_cases(count,native_count=24):
    return ''.join('<testcase classname="%s" name="case%d" />'%(
        'tests.test_observation_native_astar' if i<native_count else 'tests.test_other',i) for i in range(count))


@pytest.mark.parametrize('key,value',[('candidates',8),('search_deadline_seconds',4.),
    ('maximum_expanded_nodes_per_candidate',40000),('spatial_sigma_m',.1),('dev_requests',36)])
def test_registered_limits_and_both_arm_budget_cannot_change(key,value):
    config,old=config_pair();runner.config_guard(config,old)
    config[key]=value
    with pytest.raises(ValueError,match='Registered'):
        runner.config_guard(config,old)


def test_old_fit_and_export_registration_cannot_be_replaced():
    config,old=config_pair();old['original_shared_fit_sha256']='changed'
    with pytest.raises(ValueError,match='Original spatial'):
        runner.config_guard(config,old)


@pytest.mark.parametrize('attribute,value',[('failures','1'),('errors','1'),('skipped','1'),('tests','23')])
def test_differential_gate_rejects_failure_skip_or_missing_test(tmp_path,attribute,value):
    numbers=dict(tests='24',failures='0',errors='0',skipped='0');numbers[attribute]=value
    path=tmp_path/'pytest.xml';path.write_text('<testsuites><testsuite '+
        ' '.join('%s="%s"'%x for x in numbers.items())+'>'+junit_cases(24)+'</testsuite></testsuites>')
    with pytest.raises(ValueError,match='actual native differential'):
        runner.junit_passed(path)


def test_completed_actual_junit_and_same_compiler_flags_required(tmp_path):
    path=tmp_path/'pytest.xml';path.write_text('<testsuites><testsuite tests="38" failures="0" errors="0" skipped="0">'+
        junit_cases(38)+'</testsuite></testsuites>')
    assert runner.junit_passed(path)['tests']==38
    common=dict(protocol='native',cpp_sha256='cpp',adapter_sha256='adapter',compiler_path='g++',compiler_sha256='compiler',
        compiler_version_sha256='version',flags=['-fno-fast-math'],flags_sha256='flags',system='Linux',machine='x86_64',exit_code=0)
    runner.same_build(common,dict(common,test_only_build=False))
    with pytest.raises(ValueError,match='compiler/source/flags'):
        runner.same_build(common,dict(common,compiler_sha256='different'))
    with pytest.raises(ValueError,match='compiler/source/flags'):
        runner.same_build(common,dict(common,flags=['-ffast-math']))


@pytest.mark.parametrize('total,native_count',[(45,0),(69,0),(69,23)])
def test_many_other_tests_cannot_replace_actual_native_differentials(tmp_path,total,native_count):
    path=tmp_path/'pytest.xml';path.write_text('<testsuites><testsuite tests="%d" failures="0" errors="0" skipped="0">'%total+
        junit_cases(total,native_count)+'</testsuite></testsuites>')
    with pytest.raises(ValueError,match='distinct actual native'):
        runner.junit_passed(path)


def test_stage_gate_binds_sources_binary_full_train_budget_and_artifacts(tmp_path):
    build=tmp_path/'build';build.mkdir()
    binary=build/'kernel.so';binary.write_bytes(b'compiled')
    receipt=build/'build_receipt.json';receipt.write_text(json.dumps(dict(library_path=str(binary))))
    artifact=tmp_path/'saved.json';artifact.write_text('{}')
    source={'source':'sha'}
    report=dict(protocol=runner.PROTOCOL,status='completed',stage='train_preflight',source_commit='a'*40,
        source_files_sha256=source,build_receipt_sha256=runner.native.digest(receipt),library_sha256=runner.native.digest(binary),
        mechanical_preflight_passed=True,completed_requests=24,requested_candidate_slots=96,shared_fit_sha256=runner.FIT_SHA,
        artifact_sha256={str(artifact):runner.native.digest(artifact)})
    runner.gate_identity(report,'train_preflight','a'*40,source,build)
    for key,value in [('source_commit','b'*40),('source_files_sha256',{}),('mechanical_preflight_passed',False),
                      ('completed_requests',12),('requested_candidate_slots',48)]:
        bad=deepcopy(report);bad[key]=value
        with pytest.raises(ValueError):runner.gate_identity(bad,'train_preflight','a'*40,source,build)
    artifact.write_text('{"changed":true}')
    with pytest.raises(ValueError,match='changed'):
        runner.gate_identity(report,'train_preflight','a'*40,source,build)


def test_first_complete_and_failed_path_exact_gate_no_tolerance(tmp_path):
    identifier='test'
    for arm in ('edge','spatial'):
        folder=tmp_path/arm/identifier;folder.mkdir(parents=True)
        np.savez(folder/'predictions.npz',paths=np.zeros((1,4,24,3)))
        np.savez(folder/'raw_paths.npz',candidate_0=np.zeros((2,3)))
    assert runner.check_first_pair(tmp_path,identifier)['raw_equal']
    np.savez(tmp_path/'spatial'/identifier/'raw_paths.npz',candidate_0=np.full((2,3),1e-14))
    with pytest.raises(ValueError,match='first path differs'):
        runner.check_first_pair(tmp_path,identifier)
    for arm in ('edge','spatial'):
        np.savez(tmp_path/arm/identifier/'predictions.npz',paths=np.full((1,4,24,3),np.nan))
        np.savez(tmp_path/arm/identifier/'raw_paths.npz',candidate_0=np.empty((0,3)))
    assert runner.check_first_pair(tmp_path,identifier)['h24_equal']


def test_actual_recorded_stage_required_not_only_a_report(tmp_path):
    run=tmp_path/'runs';run.mkdir()
    (run/'tests.status.json').write_text(json.dumps(dict(status='failed',exit_code=1,code_commit='a'*40)))
    with pytest.raises(ValueError,match='recorded stage'):
        runner.completed_stage(run,'tests','a'*40,{},tmp_path/'build')


def test_compiler_receipt_discovery_ignores_gate_fixture_but_rejects_multiple_actual_builds(tmp_path):
    fake=tmp_path/'gate_fixture';fake.mkdir()
    (fake/'build_receipt.json').write_text(json.dumps(dict(library_path='not_a_compile')))
    actual=tmp_path/'native_astar0/build';actual.mkdir(parents=True)
    receipt=dict(protocol=runner.native.PROTOCOL,test_only_build=True,exit_code=0,
        cpp_sha256='cpp',adapter_sha256='adapter',compiler_path='g++',compiler_sha256='compiler',
        compiler_version_sha256='version',flags=['-O3'],flags_sha256='flags',system='Linux',machine='x86_64',
        library_path='actual.so',library_sha256='compiled')
    (actual/'build_receipt.json').write_text(json.dumps(receipt))
    assert runner.find_actual_test_build(tmp_path)==actual/'build_receipt.json'
    another=tmp_path/'native_astar1/build';another.mkdir(parents=True)
    (another/'build_receipt.json').write_text(json.dumps(receipt))
    with pytest.raises(ValueError,match='Exactly one'):
        runner.find_actual_test_build(tmp_path)


def test_incomplete_actual_compiler_receipt_cannot_be_ignored(tmp_path):
    actual=tmp_path/'native_astar0/build';actual.mkdir(parents=True)
    (actual/'build_receipt.json').write_text(json.dumps(dict(protocol=runner.native.PROTOCOL,test_only_build=True,exit_code=0)))
    with pytest.raises(ValueError,match='incomplete'):
        runner.find_actual_test_build(tmp_path)
