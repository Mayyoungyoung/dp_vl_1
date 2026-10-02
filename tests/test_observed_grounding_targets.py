import copy
import numpy as np
import pytest
from routeset.observed_grounding_targets import choose_landmark,prepare_event_grounding_targets,validate_grounding_target_resume


def fixture():
    paths=np.zeros((4,2,4,3),np.float32);paths[0,0,:,0]=[0,.1,.2,.9];paths[0,1,:,0]=[0,.3,.4,.5];paths[1]=paths[0]
    paths[2:]=np.nan
    events=np.array([[[1,1,0,0],[1,1,1,1]],[[1,1,0,0],[1,1,1,1]],[[0]*4]*2,[[0]*4]*2],np.float32)
    mask=np.array([[1,1],[1,1],[0,0],[1,0]],bool)
    data=dict(paths=paths,events=events,path_mask=mask,splits=np.array(['TRAIN','TRAIN','TRAIN','DEV_MODEL']),
        parent_ids=np.array(['p','p','empty','dev']),scene_ids=np.array(['p_a','p_b','empty_a','dev_a']),tasks=['lift','lift','cup','reach'])
    geometry=dict(index=np.array([0,0,1,2]),points=dict(world_xyz=np.array([[[.2,0,0],[.5,0,0],[np.nan]*3],[[0,0,0]]*3,[[np.nan]*3]*3]),valid_mask=np.array([[1,1,0],[1,1,1],[1,1,1]],bool)),fingerprint='fixed-test-input')
    return data,geometry


def test_closer_first_close_but_endpoint_on_exact_tie_no_close_and_masked_nan():
    path=np.array([[0,0,0],[.2,0,0],[.9,0,0]])
    point,record=choose_landmark(path,[1,0,0],[[.2,0,0],[np.nan]*3],[1,0])
    np.testing.assert_array_equal(point,path[1]);assert record['selected_kind']=='first_close_after'
    path[-1]=path[1];_,record=choose_landmark(path,[1,0,0],[[.2,0,0]],[1])
    assert record['selected_kind']=='endpoint'
    point,record=choose_landmark(path,[1,1,1],[[.2,0,0]],[1])
    assert record['first_close_index'] is None and record['selected_kind']=='endpoint'


def test_all_positive_train_no_dev_and_no_rng_or_task_id_in_point_selection():
    data,geometry=fixture();before=np.random.get_state();targets,metadata=prepare_event_grounding_targets(data,geometry,np.array([0,1]))
    after=np.random.get_state();assert before[0]==after[0];np.testing.assert_array_equal(before[1],after[1]);assert before[2:]==after[2:]
    assert np.isfinite(targets).all() and not targets[2:].any()
    assert metadata['unique_parent_reference_slots']==2 and metadata['positive_reference_observation_slots']==4
    assert metadata['zero_reference_train_observations']==1 and not metadata['reference_filtering']
    changed=copy.deepcopy(data);changed['tasks']=['different']*4
    other,_=prepare_event_grounding_targets(changed,geometry,np.array([0,1]));np.testing.assert_array_equal(targets,other)
    with pytest.raises(ValueError,match='only positive TRAIN'):prepare_event_grounding_targets(data,geometry,np.array([0,1,3]))


def test_reference_changes_fingerprint_and_inconsistent_paraphrases_fail():
    data,geometry=fixture();_,before=prepare_event_grounding_targets(data,geometry,np.array([0,1]))
    changed=copy.deepcopy(data);changed['paths'][:2,0,2,0]=.21
    _,after=prepare_event_grounding_targets(changed,geometry,np.array([0,1]));assert before['target_fingerprint']!=after['target_fingerprint']
    changed['paths'][1,0,2,0]=.22
    with pytest.raises(ValueError,match='share the exact'):prepare_event_grounding_targets(changed,geometry,np.array([0,1]))


def test_resume_legacy_default_and_new_target_fingerprint_required():
    validate_grounding_target_resume({'grounding_target':'endpoint'}, {})
    c={'grounding_target':'event_supported','grounding_target_fingerprint':'a'}
    validate_grounding_target_resume(c,c)
    with pytest.raises(ValueError,match='grounding_target'):validate_grounding_target_resume(c,{})
    with pytest.raises(ValueError,match='fingerprint'):validate_grounding_target_resume(c,dict(c,grounding_target_fingerprint='b'))
