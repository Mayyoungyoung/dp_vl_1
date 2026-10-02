import copy
import json
from pathlib import Path

import pytest

from routeset.legacy_multitask_transfer import (LEGACY_SOURCE,mechanical_gate,validate_selection,
    validate_checkpoint_plan,verify_export,aggregate,validate_cache_compatibility,audit_model_source,
    MODEL_SOURCES,EVALUATION_FUNCTIONS)
from scripts.export_legacy_multitask_dev import build
from scripts.observation_collect_multitask import registration
from scripts.snapshot_multitask_observations import digest,write_json
from test_multitask_snapshot import make_corpus,save_json


@pytest.fixture
def corpus(tmp_path):
    config=json.loads((Path(__file__).resolve().parents[1]/'configs/observed_multitask_legacy12_transfer_v1.json').read_text())
    plan=registration(281000)
    selected=[row for row in plan['parents'] if row['split']=='DEV_MODEL']
    source,other,_=make_corpus(tmp_path,selected)
    write_json(source/'partition_manifest.json',plan);write_json(source/'source_manifest.json',LEGACY_SOURCE)
    # No native physical summary in this historical source.
    for path in source.glob('DEV_MODEL/parents/*/mechanical_fingerprint.json'):path.unlink()
    comparison=registration(282000,'validated-five-v1','interleaved-early-dev-v1')
    write_json(other/'partition_manifest.json',comparison)
    for i,spec in enumerate(comparison['parents']):
        if spec['split'] not in ('TRAIN','DEV_MODEL'):continue
        value=('%064x'%(i+1000))
        save_json(other/spec['split']/'parents'/spec['parent_id']/'mechanical_fingerprint.json',
            dict(spec,physical_layout_sha256=value,physical_layout_quantized_sha256=value,rgb_file_sha256=value))
    # Any accidental locked access would fail JSON/image parsing.
    locked=next(row for row in comparison['parents'] if row['split']=='TEST_LOCKED')
    f=other/'TEST_LOCKED'/'parents'/locked['parent_id'];f.mkdir(parents=True)
    (f/'mechanical_fingerprint.json').write_text('DO NOT OPEN LOCKED')
    config.update(source_dataset=str(source),compare_dataset=str(other),source_manifest_sha256=digest(source/'partition_manifest.json'),compare_manifest_sha256=digest(other/'partition_manifest.json'),
        selected_requested_parents=selected,output_dataset='transfer')
    return config,source,other,tmp_path/'transfer'


def test_export_preserves_failures_null_semantics_and_strict_input(corpus):
    config,source,_,out=corpus
    result=build(config,out)
    assert result['requested_parents']==12 and result['requested_attempts']==36
    assert result['actual_attempt_records']==33 and result['successful_reference_routes']==10
    assert result['parents_with_zero_reference']==1
    inputs=[json.loads(line) for line in (out/'observations.jsonl').read_text().splitlines()]
    labels=[json.loads(line) for line in (out/'supervision.jsonl').read_text().splitlines()]
    assert len(inputs)==23 and all(set(r)=={'id','parent_id','split','image','instruction'} for r in inputs)
    assert all(r['split']=='DEV_MODEL' and r['semantic_targets'] is None for r in labels)
    assert sum(not r['routes'] for r in labels)==3
    assert not list(source.glob('DEV_MODEL/parents/*/mechanical_fingerprint.json'))
    assert verify_export(out,config)[0]['protocol']==result['protocol']


def test_no_partial_parent_or_locked_substitution(corpus):
    config,source,_,out=corpus
    changed=copy.deepcopy(config);changed['selected_requested_parents'][-1]['split']='TEST_LOCKED'
    with pytest.raises(ValueError,match='twelve original DEV'):mechanical_gate(changed)
    last=config['selected_requested_parents'][-1]
    (source/'DEV_MODEL'/'parents'/last['parent_id']/'worker.lock').write_text('still active')
    with pytest.raises(ValueError,match='still running'):build(config,out)
    assert not out.exists()


def test_duplicate_against_training_blocks_transfer(corpus):
    config,_,other,out=corpus
    gate=mechanical_gate(config);legacy=next(row for row in gate['rows'] if row.get('derived_legacy'))
    plan=json.loads((other/'partition_manifest.json').read_text())
    spec=next(row for row in plan['parents'] if row['split']=='TRAIN' and row['task']==legacy['task'])
    path=other/'TRAIN'/'parents'/spec['parent_id']/'mechanical_fingerprint.json'
    row=json.loads(path.read_text());row['physical_layout_quantized_sha256']=legacy['physical_layout_quantized_sha256'];write_json(path,row)
    with pytest.raises(ValueError,match='duplicate blocks'):build(config,out)


def test_missing_comparison_physical_hash_is_not_legacy_unknown(corpus):
    config,_,other,_=corpus
    path=next(other.glob('TRAIN/parents/*/mechanical_fingerprint.json'))
    row=json.loads(path.read_text());del row['physical_layout_sha256'];write_json(path,row)
    with pytest.raises(ValueError,match='Complete physical hash'):mechanical_gate(config)


def test_label_corruption_and_live_source_change_rejected(corpus):
    config,source,_,out=corpus
    build(config,out)
    path=next(source.glob('DEV_MODEL/parents/*/attempts/*/route.npz'))
    path.write_bytes(path.read_bytes()+b'corrupt')
    with pytest.raises(ValueError,match='committed source changed'):verify_export(out,config)


def test_frozen_plan_rejects_missing_last_reselection_or_wrong_step(corpus):
    config,*_=corpus
    assert len(validate_checkpoint_plan(config))==12
    bad=copy.deepcopy(config);bad['requested_checkpoints'].pop()
    with pytest.raises(ValueError,match='twelve frozen'):validate_checkpoint_plan(bad)
    bad=copy.deepcopy(config);bad['checkpoint_reselection']=True
    with pytest.raises(ValueError,match='Frozen inference'):validate_checkpoint_plan(bad)
    bad=copy.deepcopy(config);next(r for r in bad['requested_checkpoints'] if r['checkpoint_kind']=='last')['step']=1000
    with pytest.raises(ValueError,match='last1500'):validate_checkpoint_plan(bad)


def test_old_snapshot_and_model_gate_are_not_modified(corpus):
    config,_,_,out=corpus
    build(config,out)
    assert (out/'export_manifest.json').exists() and not (out/'snapshot_manifest.json').exists()
    inputs=out/'observations.jsonl';inputs.write_text(inputs.read_text()+'\n')
    with pytest.raises(ValueError,match='Export file changed'):verify_export(out,config)


def test_registered_rows_reordering_keeps_seed_pairs():
    entries=[]
    for arm in ('ordinary','event_supported'):
        for seed in range(3):
            for kind in ('best','last'):
                value=(seed+1)*(.1 if arm=='ordinary' else .07)
                entries.append(dict(arm=arm,seed=seed,checkpoint_kind=kind,metrics={k:value for k in
                    ('candidate_matched_ADE_m','candidate_endpoint_error_m','event_state_accuracy','event_sequence_accuracy')}))
    reordered=entries[8:]+list(reversed(entries[:8]))
    assert aggregate(reordered)==aggregate(entries)
    assert aggregate(entries)['best']['candidate_matched_ADE_m']['paired_aux_minus_ordinary']['seeds']==[0,1,2]


def test_cache_all_production_and_pooling_fields_must_match():
    path=Path(__file__).resolve().parents[1]/'reports/observed_multitask_prefix108_v1/free_offset_seed0/config.json'
    # Fixture values mirror the pinned historical producer, without requiring reports in code-only releases.
    cache=dict(model='Qwen/Qwen3-VL-2B-Instruct',revision='89644892e4d85e24eaac8bacfd4f463576704203',
        processor='89644892e4d85e24eaac8bacfd4f463576704203',manifest_sha256='old',torch='2.4.1+cu121',
        transformers='4.57.1',max_pixels=262144,dtype='torch.bfloat16',model_trainable_parameter_count=0)
    cfg=dict(cache_config=cache,pooling='both',feature_dim=4096,pixel_stride=2,geometry_pooling='spatial')
    new=dict(cache,manifest_sha256='new')
    assert validate_cache_compatibility(new,[cfg],'new')['output_dimension']==4096
    for key in ('revision','processor','max_pixels','dtype','transformers'):
        bad=dict(new);bad[key]='changed'
        with pytest.raises(ValueError,match='preprocessing/runtime'):validate_cache_compatibility(bad,[cfg],'new')
    with pytest.raises(ValueError,match='pooling/dimension'):validate_cache_compatibility(new,[dict(cfg,feature_dim=2048)],'new')
    with pytest.raises(ValueError,match='exported inputs'):validate_cache_compatibility(new,[cfg],'wrong')


def test_original_source_and_reachable_evaluation_guard(tmp_path):
    old=tmp_path/'research_v2/releases'/('a'*40);runtime=tmp_path/'runtime'
    for root in (old,runtime):
        for name in MODEL_SOURCES:
            path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('original')
        (root/'scripts/train_observed_geometry.py').write_text(''.join('def '+name+'():\n    return 1\n' for name in EVALUATION_FUNCTIONS))
    cfg=dict(code_commit='a'*40,source_script_sha256=digest(old/'scripts/train_observed_geometry.py'))
    assert audit_model_source(tmp_path,runtime,cfg)['training_commit']=='a'*40
    path=runtime/'scripts/train_observed_geometry.py';path.write_text(path.read_text()+'\ndef train():\n    pass\n')
    audit_model_source(tmp_path,runtime,cfg) # Unused new training flags do not change frozen inference.
    path.write_text(path.read_text().replace('return 1','return 2',1))
    with pytest.raises(ValueError,match='evaluation/geometry'):audit_model_source(tmp_path,runtime,cfg)


def test_reference_symlink_escape_is_rejected_before_read(corpus,tmp_path):
    config,source,_,_=corpus
    path=next(source.glob('DEV_MODEL/parents/*/reference_pointer.json'))
    pointer=json.loads(path.read_text());ref=path.parent/pointer['directory']/'reference.json'
    outside=tmp_path/'outside.json';outside.write_bytes(ref.read_bytes());ref.unlink()
    try:ref.symlink_to(outside)
    except OSError:pytest.skip('Host does not permit symlinks; Linux run must cover this')
    with pytest.raises(ValueError,match='escapes registered parent'):mechanical_gate(config)
