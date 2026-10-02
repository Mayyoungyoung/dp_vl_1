"""Synthetic train-only bank and complete-budget retrieval checks."""
import copy
import numpy as np
import pytest
from scripts.evaluate_observed_retrieval import NearestReferenceBank,fit_train_bank


def fixture():
    features=np.array([[1.,0.],[1.,0.],[0.,1.],[1.,0.]])
    paths=np.zeros((4,2,3,3),np.float32)
    for index in range(4):
        paths[index,0,:,0]=[5.,6.,7.+index]
        paths[index,1,:,0]=[4.,6.,8.+index]
    return dict(features=features,paths=paths,events=np.tile([1.,0.,0.],(4,2,1)),
        path_mask=np.array([[True,True],[True,True],[False,False],[True,True]]),
        splits=np.array(['TRAIN','TRAIN','TRAIN','DEV_MODEL']),scene_ids=np.array(['z','a','empty','dev']),
        parent_ids=np.array(['p0','p1','p2','p3']),tasks=['one','two','three','query'],
        semantic_targets=[None]*4)


def test_stable_id_tie_fixed_reference_order_translation_and_full_k_budget():
    data=fixture();bank,ids=fit_train_bank(data)
    assert ids.tolist()==[0,1]
    current=np.array([[.1,.2,.3,0,0,0,1,0.]])
    paths,events,details=bank.predict(np.array([[1.,0.]]),current,4)
    assert details[0]['nearest_train_id']=='a' and details[0]['original_reference_indices']==[0,1,0,1]
    assert details[0]['complete_proposals']==4 and details[0]['duplicated_reference_slots']==2
    np.testing.assert_array_equal(paths[0,:,0],np.tile(current[0,:3],(4,1)).astype(np.float32))
    np.testing.assert_array_equal(events[0,:,0],np.zeros(4))
    np.testing.assert_array_equal(events[0,:,1:],np.zeros((4,2)))
    np.testing.assert_allclose(paths[0,:,-1,0],[3.1,5.1,3.1,5.1],atol=1e-6)


def test_dev_labels_and_all_task_metadata_do_not_enter_bank_or_predictions():
    data=fixture();current=np.array([[0,0,0,0,0,0,1,1]],np.float32)
    first,ids=fit_train_bank(data);prediction=first.predict(data['features'][3:],current)
    changed=copy.deepcopy(data);changed['paths'][3]=np.nan;changed['events'][3]=np.nan
    changed['path_mask'][3]=False;changed['tasks']=['wrong']*4;changed['semantic_targets']=[{'oracle':999}]*4
    second,other=fit_train_bank(changed);counterfactual=second.predict(changed['features'][3:],current)
    np.testing.assert_array_equal(ids,other)
    for index in (0,1):np.testing.assert_array_equal(prediction[index],counterfactual[index])
    assert prediction[2]==counterfactual[2]
    with pytest.raises(TypeError):second.predict(data['features'][3:],current,task='oracle')


def test_feature_condition_changes_neighbor_and_zero_reference_train_cannot_be_selected():
    data=fixture();data['features'][1]=[0.,1.];bank,_=fit_train_bank(data)
    current=np.zeros((2,8),np.float32)
    _,_,details=bank.predict(np.eye(2),current,4)
    assert [row['nearest_train_id'] for row in details]==['z','a']
    assert all(row['nearest_train_id']!='empty' for row in details)
    data['path_mask'][:3]=False
    with pytest.raises(ValueError,match='nonempty'):fit_train_bank(data)


def test_invalid_features_and_nonpositive_budget_fail_explicitly():
    bank,_=fit_train_bank(fixture());current=np.zeros((1,8))
    with pytest.raises(ValueError,match='zero-norm'):bank.predict(np.zeros((1,2)),current)
    with pytest.raises(ValueError,match='positive integer'):bank.predict(np.ones((1,2)),current,0)


def test_all_dev_including_no_reference_use_same_macro_reference_metrics():
    pytest.importorskip('torch')
    from scripts.train_observed_routes import observation_metrics
    from routeset.observed_multitask import aggregate_task_parent_reference
    data=fixture()
    for key in ('features','paths','events','path_mask','splits','scene_ids','parent_ids'):
        data[key]=np.concatenate([data[key],data[key][-1:]],axis=0)
    data['scene_ids'][-1]='none';data['parent_ids'][-1]='new';data['path_mask'][-1]=False
    data['semantic_targets']=[None]*5;data['tasks']=['one','two','three','query','query']
    bank,_=fit_train_bank(data);ids=np.array([3,4]);pred,event,_=bank.predict(data['features'][ids],np.zeros((2,8)))
    metrics,rows=observation_metrics(pred,event,data,ids)
    aggregate_task_parent_reference(metrics,rows,[data['tasks'][idx] for idx in ids])
    assert metrics['examples']==2 and metrics['reference_evaluation_examples']==1
    assert metrics['semantic_goal_accuracy'] is None and metrics['semantic_evaluation_examples']==0
    assert rows[-1]['candidate_matched_ADE_m'] is None and metrics['candidates']==4
