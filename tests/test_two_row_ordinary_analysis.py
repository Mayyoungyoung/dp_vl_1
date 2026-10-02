import numpy as np
import pytest
from scripts.analyze_two_row_ordinary import opportunity,cross_goal_consistency,read_pool,digest


def candidate(mode,valid=True):
    return dict(declared_passage_type=mode,classified_tip_valid=valid and mode is not None,TipValid=valid,
        finite_xyz=True,finite_event_values=True,semantic_goal_correct=valid,starts_at_current_state=True,
        tip_segments_clear=True,event_state_sequence_correct=True)


def test_duplicate_gap_needs_known_missing_and_valid_duplicate():
    n,m,p=['negative_y']*2,['middle']*2,['positive_y']*2
    result=opportunity([candidate(n),candidate(n),candidate(p,False),candidate(None)],[n,m,None])
    assert result['duplicate_replacement_count_upper_bound']==1
    assert result['valid_classified_duplicate_slots']==1 and result['unknown_valid_slots']==1
    assert result['missing_known_types']==[tuple(m)]
    assert result['nonexclusive_failure_counts']['semantic_goal_correct']==1


def test_unseen_valid_type_is_not_duplicate_or_reference_error():
    n,m,p=['negative_y']*2,['middle']*2,['positive_y']*2
    result=opportunity([candidate(n),candidate(p)],[n,m])
    assert result['duplicate_replacement_count_upper_bound']==0
    assert tuple(p) in result['valid_predicted_types'] and result['missing_known_types']==[tuple(m)]


def test_cross_goal_uses_only_known_positive_intersection():
    n,m,p=['negative_y']*2,['middle']*2,['positive_y']*2
    a=dict(id='a',parent_id='parent',opportunity=opportunity([candidate(n),candidate(m)],[n,m,None]))
    b=dict(id='b',parent_id='parent',opportunity=opportunity([candidate(p)],[n,p,None]))
    result=cross_goal_consistency([a,b])
    assert result['directed_observed_pairs']==2 and result['eligible_pairs']==1
    assert result['mean_known_common_type_consistency']==0
    assert result['pairs'][0]['known_common_types']==[tuple(n)]
    assert result['pairs'][1]['fraction'] is None


def test_no_known_references_cannot_support_correspondence():
    a=dict(id='a',parent_id='parent',opportunity=opportunity([candidate(None)],[None]))
    b=dict(id='b',parent_id='parent',opportunity=opportunity([candidate(['middle']*2)],[]))
    result=cross_goal_consistency([a,b])
    assert result['eligible_pairs']==0 and result['mean_known_common_type_consistency'] is None


def test_saved_pool_cannot_drop_condition_or_change_registered_parent(tmp_path):
    path=tmp_path/'predictions.npz'
    np.savez(path,scene_ids=['a'],parent_ids=['p'],paths=np.zeros((1,4,24,3)),gripper_open=np.ones((1,4,24)))
    labels={'a':{'parent_id':'p'}}
    with pytest.raises(ValueError,match='exact registered split'):
        read_pool(path,labels,{'a','b'},'weight',digest(path))
    with pytest.raises(ValueError,match='parent identity'):
        read_pool(path,{'a':{'parent_id':'wrong'}},{'a'},'weight',digest(path))
    with pytest.raises(ValueError,match='hash differs'):
        read_pool(path,labels,{'a'},'weight','changed')
