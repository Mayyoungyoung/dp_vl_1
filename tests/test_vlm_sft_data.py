"""Pure SFT interface tests; no real-Qwen training or robotics claims."""
import json
from copy import deepcopy

import numpy as np
from PIL import Image
import pytest

from routeset.vlm_sft_data import (alternating_k, fixed_development_plan, load_sft_data,
    parent_language_groups, prepare_prefix, prepare_teacher_forcing, read_observation,
    reference_indices, sample_parent_language, validate_resume_config)
from routeset.vlm_sft_data import exclusive_training_output,validate_reserved_observation_manifest


class FakeProcessor:
    """Explicitly synthetic reversible character-token fixture."""
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = 'USER:' + messages[0]['content'][-1]['text'] + '\nASSISTANT:'
        if len(messages) == 2:
            text += messages[1]['content'][0]['text'] + '<eos>'
        return text

    def __call__(self, text, images, return_tensors):
        assert len(images) == 2 and return_tensors == 'pt'
        ids = np.asarray([[ord(char) % 256 for char in text[0]]], dtype=np.int64)
        return dict(input_ids=ids, attention_mask=np.ones_like(ids))


def fixture_resample(poses, opened, horizon):
    # Constant-event synthetic fixture only; production uses the existing
    # actual event-preserving resampler, never this injected shortcut.
    return (np.linspace(poses[0,:3], poses[-1,:3], horizon).astype(np.float32),
            np.full(horizon, opened[0], dtype=np.float32))


def make_fixture(root):
    root.mkdir(parents=True, exist_ok=True)
    observations, labels = [], []
    for number,(split,refs,languages) in enumerate([('TRAIN',2,2),('TRAIN',3,1),('DEV_MODEL',1,1),('DEV_MODEL',0,1)]):
        folder = root / ('parent%d' % number); folder.mkdir()
        image = folder/'rgb.png'; Image.fromarray(np.zeros((4,4,3),np.uint8)).save(image)
        np.savez(folder/'observation.npz', gripper_pose=np.array([.1,.2,.3,0,0,0,1]), gripper_open=np.array(1.),
            depth=np.ones((4,4)),camera_intrinsics=np.eye(3),camera_extrinsics=np.eye(4),
            task_low_dim_state=np.array([9999.]), future_path=np.array([8888.]))
        routes=[]
        for ref in range(refs):
            path=folder/('route%d.npz'%ref); routes.append(str(path))
            xyz=np.linspace([.1,.2,.3],[.5, .3+ref*.1,.7],4)
            np.savez(path,gripper_pose=np.c_[xyz,np.tile([0,0,0,1],(4,1))],gripper_open=np.ones(4))
        for language in range(languages):
            identity='parent%d_language%d'%(number,language)
            observations.append(dict(id=identity,parent_id='parent%d'%number,split=split,image=str(image),instruction='reach target %d'%language))
            labels.append(dict(id=identity,parent_id='parent%d'%number,split=split,
                observation=str(folder/'observation.npz'),routes=routes,semantic_targets=[7777.,7777.,7777.],route_types=['private']*refs))
    obs,sup=root/'observations.jsonl',root/'supervision.jsonl'
    for path, rows in ((obs,observations),(sup,labels)):
        path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n',encoding='utf-8')
    return obs,sup


def test_budget_target_selection_all_positives_and_explicit_rng_restore():
    rng=np.random.default_rng(17)
    for count in (1,2,3,4):
        for _ in range(20):
            chosen=reference_indices(count,4,rng)
            assert len(chosen)==4 and set(chosen)==set(range(count))
    state=deepcopy(rng.bit_generator.state)
    expected=[reference_indices(3,alternating_k(i),rng).tolist() for i in range(1,13)]
    restored=np.random.default_rng();restored.bit_generator.state=state
    assert expected==[reference_indices(3,alternating_k(i),restored).tolist() for i in range(1,13)]
    with pytest.raises(ValueError,match='at most four'):reference_indices(5,4,rng)
    with pytest.raises(ValueError,match='Positive'):reference_indices(0,1,rng)


def test_parent_then_language_uniform_and_dev_plan_does_not_consume_training_rng(tmp_path):
    obs,sup=make_fixture(tmp_path/'data');data=load_sft_data(obs,sup,4,fixture_resample)
    groups=parent_language_groups(data['samples']);assert list(map(len,groups))==[2,1]
    rng=np.random.default_rng(18);state=deepcopy(rng.bit_generator.state)
    first=fixed_development_plan(data['samples'],10)
    assert first==fixed_development_plan(data['samples'],10) and len(first)==2
    assert rng.bit_generator.state==state
    draws=[sample_parent_language(groups,rng) for _ in range(12000)]
    freq=np.bincount(draws,minlength=3)/len(draws)
    np.testing.assert_allclose(freq,[.25,.25,.5],atol=.02)
    assert len(data['unreferenced'])==1


def test_answer_free_prefix_and_exact_teacher_forced_boundary(tmp_path):
    obs,sup=make_fixture(tmp_path/'data');data=load_sft_data(obs,sup,4,fixture_resample)
    sample=data['samples'][0];observed=read_observation(sample);processor=FakeProcessor()
    prefix=prepare_prefix(processor,observed,4,4)
    selected=np.array([0,1,0,1])
    full,labels,receipt=prepare_teacher_forcing(processor,observed,sample['paths'][selected],sample['events'][selected],4,4)
    length=prefix['input_ids'].shape[1]
    np.testing.assert_array_equal(prefix['input_ids'],full['input_ids'][:,:length])
    assert (labels[:,:length]==-100).all() and (labels[:,length:]!=-100).all()
    assert receipt['k']==4 and receipt['max_coordinate_quantization_error_m']<=.000501
    assert 'paths' not in observed and 'semantic_targets' not in observed and 'task_low_dim_state' not in observed
    # Passing an answer field is rejected even if a caller accidentally has it.
    with pytest.raises(ValueError,match='whitelist'):
        prepare_prefix(processor,dict(observed,paths=sample['paths']),4,4)


def test_data_join_leakage_and_config_changes_are_refused(tmp_path):
    obs,sup=make_fixture(tmp_path/'data')
    rows=[json.loads(line) for line in obs.read_text().splitlines()]
    labels=[json.loads(line) for line in sup.read_text().splitlines()]
    rows[-1]['parent_id']=rows[0]['parent_id'];labels[-1]['parent_id']=rows[0]['parent_id']
    obs.write_text('\n'.join(map(json.dumps,rows)));sup.write_text('\n'.join(map(json.dumps,labels)))
    with pytest.raises(ValueError,match='Parent leakage'):load_sft_data(obs,sup,4,fixture_resample)
    base=dict(lr=.0001,steps=1500,eval_every=250,data_fingerprint='original',dev_plan_sha256='original',source_sha256={'trainer':'original'})
    validate_resume_config(dict(base),base)
    for key in base:
        changed=dict(base);changed[key]='changed'
        with pytest.raises(ValueError,match=key):validate_resume_config(changed,base)


def test_k1_samples_all_positive_references_including_unknown_without_type_inputs():
    rng=np.random.default_rng(7)
    selected=[int(reference_indices(3,1,rng)[0]) for _ in range(6000)]
    np.testing.assert_allclose(np.bincount(selected)/6000,[1/3]*3,atol=.025)


def test_output_exclusion_blocks_distinct_jobs_and_cleans_up_on_exception(tmp_path):
    output=tmp_path/'run'
    with pytest.raises(RuntimeError,match='synthetic stop'):
        with exclusive_training_output(output,False):
            assert (output/'train.lock').exists()
            (output/'last.pt').write_text('placeholder for metadata-only lock test')
            with pytest.raises(FileExistsError):
                with exclusive_training_output(output,True):pass
            assert (output/'train.lock').exists()
            raise RuntimeError('synthetic stop')
    assert not (output/'train.lock').exists() and (output/'last.pt').exists()
    with exclusive_training_output(output,True):pass


def test_parent_preread_gate_rejects_locked_relabel_before_any_raw_file(tmp_path):
    rows=[dict(id=f'obstacle_reach_{n}_target0',parent_id=f'obstacle_reach_{n}',
               split='TRAIN' if n<272096 else 'DEV_MODEL',image='does-not-exist.png',instruction='reach')
          for n in range(272000,272104)]
    path=tmp_path/'observations.jsonl'
    path.write_text('\n'.join(map(json.dumps,rows)))
    assert len(validate_reserved_observation_manifest(path))==104
    rows[-1].update(parent_id='obstacle_reach_272120')
    path.write_text('\n'.join(map(json.dumps,rows)))
    with pytest.raises(ValueError,match='raw contents were not opened'):
        validate_reserved_observation_manifest(path)
