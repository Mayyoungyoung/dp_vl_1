"""Metadata controls only: synthetic fixtures are not experimental evidence."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from routeset.observed_multitask import (REFERENCE_FIELDS,aggregate_task_parent_reference,
    check_multitask_model_gate,draw_observation_batch,validate_multitask_resume)


def test_uniform_sampling_is_exact_historical_stream_and_balanced_resume():
    data=dict(tasks=['a','a','a','a','b'],parent_ids=np.array(['a0','a0','a0','a1','b0']))
    ids=np.arange(5);old=np.random.default_rng(78);new=np.random.default_rng(78)
    for _ in range(3):
        np.testing.assert_array_equal(draw_observation_batch(data,ids,new,17),old.choice(ids,17,replace=True))
    balanced=np.random.default_rng(17)
    samples=draw_observation_batch(data,ids,balanced,20000,'task_parent_language')
    # 50% task b; task a shares its half between two parents, not 4 wording rows.
    proportions=np.bincount(samples,minlength=5)/len(samples)
    np.testing.assert_allclose(proportions,[1/12,1/12,1/12,1/4,1/2],atol=.015)
    state=copy.deepcopy(balanced.bit_generator.state)
    expected=draw_observation_batch(data,ids,balanced,31,'task_parent_language')
    resumed=np.random.default_rng(999);resumed.bit_generator.state=state
    np.testing.assert_array_equal(expected,draw_observation_batch(data,ids,resumed,31,'task_parent_language'))
    assert set(draw_observation_batch(data,np.array([1,4]),resumed,200,'task_parent_language'))=={1,4}


def test_macro_reference_metrics_equalize_tasks_parents_and_keep_missing_unknown():
    parents=['a0','a0','a0','a1','b0','b1'];tasks=['a']*4+['b']*2;values=[1.,1.,1.,3.,10.,None]
    rows=[dict(parent_id=parent,reference_count=int(value is not None),
               **{field:value for field in REFERENCE_FIELDS}) for parent,value in zip(parents,values)]
    result={field:3.2 for field in REFERENCE_FIELDS}
    result.update(examples=6,reference_evaluation_examples=5,semantic_evaluation_examples=0,semantic_goal_accuracy=None)
    aggregate_task_parent_reference(result,rows,tasks)
    assert result['candidate_matched_ADE_m']==6. # ((1+3)/2 + 10)/2
    assert result['instruction_weighted_reference_metrics']['candidate_matched_ADE_m']==3.2
    assert result['evaluated_tasks']==2 and result['reference_evaluable_parents']==3
    assert result['examples']==6 and result['reference_evaluation_examples']==5
    assert result['semantic_goal_accuracy'] is None
    assert all(value is None for value in result['unsupported_task_metrics'].values())
    assert result['task_parent_reference_metrics']['per_parent'][-1]['candidate_matched_ADE_m'] is None


def test_balanced_resume_options_fail_closed_and_legacy_defaults_remain_valid():
    validate_multitask_resume(dict(sampling_mode='uniform',metric_aggregation='instruction'),{})
    for key,value in [('sampling_mode','task_parent_language'),('metric_aggregation','task_parent')]:
        with pytest.raises(ValueError,match=key):validate_multitask_resume({key:value},{})
    with pytest.raises(ValueError,match='task metadata'):
        draw_observation_batch(dict(tasks=[None],parent_ids=['p']),[0],np.random.default_rng(0),1,'task_parent_language')


def gate_fixture(tmp_path):
    source=tmp_path/'source';other=tmp_path/'other';snapshot=tmp_path/'snapshot'
    for path in (source,other,snapshot):path.mkdir()
    selected=[dict(parent_id='p0',split='TRAIN',task='lift'),dict(parent_id='p1',split='DEV_MODEL',task='lift')]
    def save(path,value):
        path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value),encoding='utf-8')
    save(source/'partition_manifest.json',dict(parents=selected))
    save(other/'partition_manifest.json',dict(parents=[]))
    for row in selected:
        save(source/row['split']/'parents'/row['parent_id']/'mechanical_fingerprint.json',
             dict(row,protocol='multitask_initial_layout_fingerprint_v1',
                 physical_layout_sha256=hashlib.sha256(row['parent_id'].encode()).hexdigest(),
                 physical_layout_quantized_sha256=hashlib.sha256(row['parent_id'].encode()).hexdigest()))
    inputs=[dict(id=row['parent_id'],image='never-opened.png',instruction='lift',**row) for row in selected]
    # task belongs to supervision only, never the observation input.
    for row in inputs:row.pop('task')
    labels=[dict(id=row['parent_id'],observation='never-opened.npz',routes=[],semantic_targets=None,**row) for row in selected]
    obs=snapshot/'observations.jsonl';sup=snapshot/'supervision.jsonl'
    for path,rows in ((obs,inputs),(sup,labels)):path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n',encoding='utf-8')
    (snapshot/'attempts.jsonl').write_text('')
    (snapshot/'parent_inventory.json').write_text('[]')
    manifest=dict(protocol='multitask_closed_prefix_snapshot_v1',current_gate_required_before_model_use=True,
        selected_requested_parents=selected,source_dataset=str(source),compare_sources=[str(other)],
        requested_parents=2,requested_attempts=6,actual_attempt_records=6,setup_failed_parents=0,
        parents_with_observation=2,parents_with_zero_reference=2,successful_reference_routes=0,
        source_files_sha256={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source.glob('*/parents/*/mechanical_fingerprint.json')},
        snapshot_files_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in
            (obs,sup,snapshot/'attempts.jsonl',snapshot/'parent_inventory.json')})
    save(snapshot/'snapshot_manifest.json',manifest)
    return obs,sup,source,other,save


def test_model_gate_rechecks_later_cross_role_duplicates_without_raw_read(tmp_path,monkeypatch):
    obs,sup,source,other,save=gate_fixture(tmp_path)
    result=check_multitask_model_gate(obs,sup)
    assert result['selected_tasks']==['lift'] and result['collection_denominators']['requested_attempts']==6
    # A subsequently collected parent can make yesterday's snapshot ineligible.
    duplicate=dict(parent_id='later_locked',split='TEST_LOCKED',task='lift')
    save(other/'partition_manifest.json',dict(parents=[duplicate]))
    save(other/'TEST_LOCKED/parents/later_locked/mechanical_fingerprint.json',
        dict(duplicate,physical_layout_sha256=hashlib.sha256(b'p0').hexdigest()))
    original=Path.read_bytes
    def guarded(path):
        assert 'TEST_LOCKED' not in path.parts,'No locked raw contents may be opened'
        return original(path)
    monkeypatch.setattr(Path,'read_bytes',guarded)
    with pytest.raises(ValueError,match='current mechanical'):check_multitask_model_gate(obs,sup)


def test_model_gate_rejects_changed_snapshot_and_legacy_has_no_added_io(tmp_path):
    obs,sup,_,_,_=gate_fixture(tmp_path)
    obs.write_text(obs.read_text()+'\n')
    with pytest.raises(ValueError,match='snapshot artifact changed'):check_multitask_model_gate(obs,sup)
    assert check_multitask_model_gate(tmp_path/'legacy/obs.jsonl',tmp_path/'legacy/labels.jsonl') is None


@pytest.mark.parametrize('parents',[32,64])
def test_real_legacy_learning_curve_manifests_are_ignored_unless_explicit(tmp_path,parents):
    saved=Path(__file__).resolve().parents[1]/'reports'/('observation_learning_curve_new%d_manifest.json'%parents)
    manifest=json.loads(saved.read_text(encoding='utf-8'))
    assert manifest['evaluation_protocol']=='observation_eval_v2' and manifest['train_parents']==parents
    assert 'protocol' not in manifest and 'current_gate_required_before_model_use' not in manifest
    path=tmp_path/'snapshot_manifest.json';path.write_text(json.dumps(manifest))
    # No referenced data are opened by automatic legacy detection.
    assert check_multitask_model_gate(tmp_path/'observations.jsonl',tmp_path/'supervision.jsonl') is None
    with pytest.raises(ValueError,match='unrecognized'):
        check_multitask_model_gate(tmp_path/'observations.jsonl',tmp_path/'supervision.jsonl',path)


def test_declared_multitask_gate_cannot_be_disabled_or_malformed(tmp_path):
    path=tmp_path/'snapshot_manifest.json'
    for manifest in (dict(protocol='multitask_closed_prefix_snapshot_v1',current_gate_required_before_model_use=False),
                     dict(protocol='multitask_closed_prefix_snapshot_v1'),
                     dict(protocol='wrong',current_gate_required_before_model_use=True)):
        path.write_text(json.dumps(manifest))
        with pytest.raises(ValueError,match='unrecognized'):
            check_multitask_model_gate(tmp_path/'observations.jsonl',tmp_path/'supervision.jsonl')


def test_live_gate_rejects_selected_fingerprint_loss_despite_legacy_pointer(tmp_path):
    obs,sup,source,_,save=gate_fixture(tmp_path)
    folder=source/'TRAIN/parents/p0';(folder/'mechanical_fingerprint.json').unlink()
    save(folder/'reference_pointer.json',dict(image_sha256='legacy-image-hash'))
    with pytest.raises(ValueError,match='no current mechanical fingerprint'):check_multitask_model_gate(obs,sup)


def test_live_gate_rejects_incomplete_or_changed_selected_fingerprint(tmp_path):
    obs,sup,source,_,save=gate_fixture(tmp_path);path=source/'TRAIN/parents/p0/mechanical_fingerprint.json'
    original=json.loads(path.read_text());changed=dict(original);changed.pop('physical_layout_quantized_sha256');save(path,changed)
    with pytest.raises(ValueError,match='complete physical fingerprint'):check_multitask_model_gate(obs,sup)
    changed=dict(original,physical_layout_sha256='a'*64);save(path,changed)
    with pytest.raises(ValueError,match='changed since snapshot'):check_multitask_model_gate(obs,sup)


def test_live_gate_keeps_future_and_comparison_unknowns_without_blocking_selected_data(tmp_path):
    obs,sup,source,other,save=gate_fixture(tmp_path)
    plan=json.loads((source/'partition_manifest.json').read_text())
    plan['parents'].append(dict(parent_id='future',split='TEST_LOCKED',task='lift'));save(source/'partition_manifest.json',plan)
    old=dict(parent_id='legacy',split='TRAIN',task='lift');save(other/'partition_manifest.json',dict(parents=[old]))
    save(other/'TRAIN/parents/legacy/reference_pointer.json',dict(image_sha256='legacy-image-hash'))
    result=check_multitask_model_gate(obs,sup)
    assert result['current_audit']['parents_with_only_legacy_image_hash']==1
    assert len(result['current_audit']['missing_mechanical_metadata'])==1
    assert len(result['selected_mechanical_files_sha256'])==2
