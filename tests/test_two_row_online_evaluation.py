"""Pure interface/ordering tests: no torch, model download or real forward."""
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import evaluate_observed_two_row_online as online


def observation():
    return dict(id=online.DEV_IDS[0], parent_id=online.DEV_PARENTS[0], split='DEV_MODEL',
        image='/recorded/'+online.DEV_PARENTS[0]+'/front.png', instruction='Touch the red sphere.')


def prediction():
    value=online.failed_prediction()
    value.update(paths=np.zeros((4,24,3),np.float32),gripper_open=np.ones((4,24),np.float32),
        current=np.r_[np.zeros(7),1.].astype(np.float32),mean_hidden=np.zeros(2048,np.float32),
        last_hidden=np.zeros(2048,np.float32),input_tokens=np.array(10))
    return value


def test_exact_dev36_and_input_whitelist():
    assert len(online.DEV_IDS)==36 and len(set(online.DEV_IDS))==36
    row=observation();assert online.observation_contract(row)==row
    for changed in (dict(row,semantic_targets=[1,2,3]),dict(row,split='TEST_LOCKED'),
                    dict(row,parent_id='another_parent'),dict(row,id='two_row_reach_283200_target0')):
        with pytest.raises(ValueError):online.observation_contract(changed)


def test_missing_inputs_are_retained_not_replaced():
    rows=online.select_observations([observation()])
    assert len(rows)==1 and len([i for i in online.DEV_IDS if i not in rows])==35
    with pytest.raises(ValueError):online.select_observations([observation(),observation()])


@pytest.mark.parametrize('count',(16,32,64))
def test_registered_scaling_keeps_identical_dev_and_rejects_extra_train(count):
    parent='two_row_reach_%d'%(283200+count-1)
    train=dict(observation(),id=parent+'_target2',parent_id=parent,split='TRAIN')
    assert online.select_observations([train,observation()],count)=={observation()['id']:observation()}
    wrong=dict(train,parent_id='two_row_reach_%d'%(283200+count))
    with pytest.raises(ValueError):online.select_observations([wrong],count)
    with pytest.raises(ValueError):online.select_observations([dict(train,id=parent+'_target3')],count)


def test_strict_current_path_hashes_and_no_label_pointer(tmp_path):
    row=observation();folder=tmp_path/row['parent_id'];folder.mkdir()
    image=folder/'front.png';image.write_bytes(b'image')
    current=folder/'observation.npz';current.write_bytes(b'current')
    row['image']=str(image)
    hashes={str(p.resolve()):online.digest(p) for p in (image,current)}
    assert online.input_paths(row,hashes)==(image.resolve(),current.resolve())
    current.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):online.input_paths(row,hashes)


def test_full_head_options_reject_unbudgeted_or_changed_models():
    config=dict(feature_dim=4096,horizon=24,candidates=4,width=128,depth=2,point_width=64,pooling='both',
        pixel_stride=2,endpoint_residual_bound=.05,anchor_mode='straight_through_peak',endpoint_mode='surface_anchor',
        refinement_mode='none',geometry_pooling='spatial',objective='saturation',checkpoint_selection='tip_unique_valid')
    args=online.head_options(config)
    assert args['anchor_mode']=='straight_through_peak' and args['endpoint_mode']=='surface_anchor'
    assert args['max_candidates']==4 and args['refinement_mode']=='none'
    for key,value in [('candidates',8),('endpoint_mode','free_offset'),('refinement_mode','local'),('refinement_sigma',.1)]:
        with pytest.raises(ValueError):online.head_options(dict(config,**{key:value}))


def test_actual_request_seal_precedes_label_and_walltime_is_continuous(tmp_path):
    order=[];times=iter([10.,11.,12.,15.])
    def generate(row):
        assert set(row)==online.INPUT_KEYS;order.append('generate');return prediction(),{'fixture_ms':10.}
    def evaluate(identifier,row,path,sha):
        assert order==['generate'] and path.exists() and path.with_suffix('.seal.json').exists()
        assert online.digest(path)==sha
        with np.load(path) as a:assert a['paths'].shape==(4,24,3)
        order.append('labels');return online.unavailable_metrics(),[],{}
    record,_=online.process_request(online.DEV_IDS[0],observation(),generate,evaluate,tmp_path,clock=lambda:next(times))
    assert order==['generate','labels']
    assert record['generation_ms']==1000 and record['prediction_seal_ms']==1000
    assert record['label_and_check_ms']==3000 and record['continuous_request_wall_ms']==5000
    assert record['retries']==record['additional_complete_paths']==0


def test_changed_labels_never_change_generation_arguments(tmp_path):
    seen=[];outputs=[]
    def generator(row):seen.append(dict(row));return prediction(),{}
    for index,label in enumerate([{'target':[0,0,0]},{'target':[999,999,999]}]):
        output=tmp_path/str(index);output.mkdir()
        def evaluator(identifier,row,path,sha):
            # Label can alter reported checking, but has no generator argument.
            return dict(online.unavailable_metrics(),fixture_label=label),[],{}
        record,value=online.process_request(online.DEV_IDS[0],observation(),generator,evaluator,output)
        outputs.append(value['paths'])
    assert seen[0]==seen[1]==observation() and np.array_equal(*outputs)
    assert set(seen[0])==online.INPUT_KEYS


def test_failure_is_four_nan_slots_without_retry(tmp_path):
    calls=[]
    def generator(row):calls.append(row);raise RuntimeError('fixture CUDA failure')
    def evaluator(identifier,row,path,sha):
        with np.load(path) as a:assert not np.isfinite(a['paths']).any()
        return online.unavailable_metrics(),[],{}
    record,value=online.process_request(online.DEV_IDS[0],observation(),generator,evaluator,tmp_path)
    assert len(calls)==1 and record['requested_candidates']==4 and record['finite_path_candidates']==0
    assert record['generation_error']=='RuntimeError: fixture CUDA failure'
    assert value['paths'].shape==(4,24,3)


def test_missing_observation_never_calls_model_or_invents_label(tmp_path):
    def generator(row):raise AssertionError('No image means no actual model call')
    def evaluator(identifier,row,path,sha):
        assert row is None;return online.unavailable_metrics(),[],{}
    record,_=online.process_request(online.DEV_IDS[0],None,generator,evaluator,tmp_path)
    assert not record['generation_attempted'] and record['requested_candidates']==4
    assert record['metrics']['KnownReferenceTypeCoverageAtK'] is None


def test_no_repair_or_overwrite_of_wrong_horizon_or_existing_pool(tmp_path):
    value=prediction();value['paths']=value['paths'][:,:23]
    with pytest.raises(ValueError,match='no post-hoc'):online.validate_prediction(value)
    online.seal_prediction(tmp_path,online.DEV_IDS[0],prediction())
    with pytest.raises(FileExistsError):online.seal_prediction(tmp_path,online.DEV_IDS[0],prediction())


def test_evaluator_cannot_mutate_sealed_candidate_pool(tmp_path):
    def evaluator(identifier,row,path,sha):
        path.write_bytes(b'mutated');return online.unavailable_metrics(),[],{}
    with pytest.raises(ValueError,match='modified'):
        online.process_request(online.DEV_IDS[0],observation(),lambda row:(prediction(),{}),evaluator,tmp_path)


def test_request_label_deserializes_only_selected_payload(tmp_path,monkeypatch):
    path=tmp_path/'supervision.jsonl'
    wanted=dict(id=online.DEV_IDS[0],semantic_targets={'target_index':0})
    other=dict(id=online.DEV_IDS[1],forbidden_label='should not deserialize')
    path.write_text(json.dumps(other)+'\n'+json.dumps(wanted)+'\n')
    loads=json.loads;opened=[]
    def checked(text,*args,**kwargs):
        value=loads(text,*args,**kwargs)
        if isinstance(value,dict):
            assert value['id']==wanted['id'];opened.append(value)
        return value
    monkeypatch.setattr(online.json,'loads',checked)
    assert online.request_label(path,wanted['id'])==wanted and opened==[wanted]


def test_summary_includes_all_registered_failure_slots_and_null_coverage():
    records=[dict(metrics=online.unavailable_metrics(),generation_attempted=False,generation_error='missing',
        continuous_request_wall_ms=0.) for _ in online.DEV_IDS]
    summary=online.summarize(records)
    assert summary['requested_candidate_slots']==144 and summary['requested_input_success_denominator']==36
    assert summary['TipValidAtK']==0 and summary['KnownReferenceTypeCoverageAtK'] is None
    assert summary['actual_attempted_generation_requests']==0 and summary['all_attempted_request_ms_median'] is None


def test_training_receipt_binds_best_comparison_and_rejects_tampering(tmp_path):
    config=dict(two_row_export_sha256='export',two_row_driver_sha256='driver')
    (tmp_path/'summary.json').write_text('{}')
    folder=tmp_path/'dev_model';folder.mkdir();artifacts={}
    for filename in ('predictions.npz','per_scene.json','metrics.json'):
        path=folder/filename;path.write_bytes(b'fixed')
        artifacts['dev_model/'+filename]=dict(path=str(path),sha256=online.digest(path),bytes=path.stat().st_size)
    receipt=dict(protocol='ordinary_two_row_frozen_qwen_saturation_v1',
        actual_training_summary_sha256=online.digest(tmp_path/'summary.json'),export_sha256='export',
        source_sha256='driver',prediction_artifacts=artifacts)
    online.write_json(tmp_path/'two_row_driver_receipt.json',receipt)
    result=online.verify_training_artifacts(tmp_path,config)
    assert len(result['best_comparison_artifact_sha256'])==3
    (folder/'per_scene.json').write_bytes(b'changed')
    with pytest.raises(ValueError,match='comparison artifact'):online.verify_training_artifacts(tmp_path,config)


def test_cached_array_comparison_uses_fixed_ids_hashes_and_no_extra_forward(tmp_path):
    import hashlib
    row=observation();image=tmp_path/'front.png';image.write_bytes(b'actual recorded image');row['image']=str(image)
    run=tmp_path/'run';folder=run/'dev_model';folder.mkdir(parents=True)
    cache=tmp_path/'cache';cache.mkdir();value=prediction()
    np.savez_compressed(folder/'predictions.npz',paths=value['paths'][None],gripper_open=value['gripper_open'][None],
        scene_ids=np.array([row['id']]),parent_ids=np.array([row['parent_id']]))
    online.write_json(run/'summary.json',dict(prediction_sha256=online.digest(folder/'predictions.npz')))
    key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
    cache_file=cache/(key+'.npz')
    np.savez_compressed(cache_file,mean_hidden=value['mean_hidden'],last_hidden=value['last_hidden'],
        id=row['id'],parent_id=row['parent_id'],split=row['split'],image_sha256=online.digest(image),input_tokens=10)
    hashes={str(cache_file.resolve()):online.digest(cache_file)}
    report=online.compare_cache(run,dict(cache_dir=str(cache)),{row['id']:row},{row['id']:value},hashes)
    assert report['additional_model_requests']==0 and report['examples']==1
    assert report['per_scene'][0]['xyz_allclose_atol1e_6_rtol1e_5']
    assert report['per_scene'][0]['feature_max_abs_difference']==dict(mean_hidden=0.,last_hidden=0.)
    cache_file.write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):online.compare_cache(run,dict(cache_dir=str(cache)),{row['id']:row},{row['id']:value},hashes)
