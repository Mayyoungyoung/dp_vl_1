import copy

import numpy as np
import pytest

from scripts import analyze_native_astar_100k_saved as audit


def slot(identifier='parent_target0',k=0):
    return dict(id=identifier,slot=k,raw=np.zeros((3,3)),h24=np.zeros((24,3)),opened=np.ones(24),
        attempt=dict(status='path_found',expanded_nodes=10,search_seconds=.1,complete_raw_paths_emitted=1),
        candidate=dict(TipValid=True,declared_passage_type=None))


def test_exact_paths_and_counters_ignore_only_runtime():
    old=slot();new=copy.deepcopy(old);new['attempt']['search_seconds']=.001
    new['attempt']['native']={'library_sha256':'compiled'}
    result=audit.compare_slot(old,new)
    assert result['raw']['equal_values'] and result['h24']['equal_values']
    assert result['deterministic_search_equal'] and result['saved_candidate_decisions_equal']


def test_same_h24_cannot_hide_different_complete_raw_path():
    old=slot();new=copy.deepcopy(old);new['raw'][1,0]=1e-12
    result=audit.compare_slot(old,new)
    assert not result['raw']['equal_values'] and result['h24']['equal_values']
    assert result['raw']['max_absolute_error_on_joint_finite_entries']==1e-12


def test_different_raw_vertex_count_kept_without_alignment_or_resampling():
    result=audit.array_difference(np.zeros((3,3)),np.zeros((4,3)))
    assert not result['equal_values'] and result['max_absolute_error_on_joint_finite_entries'] is None


def test_failed_nan_slots_and_finite_failure_mask_are_distinct():
    failed=np.full((24,3),np.nan)
    assert audit.array_difference(failed,failed)['equal_values']
    result=audit.array_difference(failed,np.zeros((24,3)))
    assert not result['equal_values'] and not result['finite_masks_equal']


def test_deterministic_count_drift_not_hidden_by_equal_path():
    old=slot();new=copy.deepcopy(old);new['attempt']['expanded_nodes']=11
    result=audit.compare_slot(old,new)
    assert result['raw']['equal_values'] and not result['deterministic_search_equal']
    assert result['deterministic_search_differences']['expanded_nodes']=={'old20k':10,'new100k':11}


def test_timeout_recovery_is_recorded_not_declared_identical():
    old=slot();new=copy.deepcopy(old)
    old.update(raw=np.empty((0,3)),h24=np.full((24,3),np.nan))
    old['attempt'].update(status='search_time_budget_exhausted',complete_raw_paths_emitted=0)
    result=audit.compare_slot(old,new)
    assert result['old_search_timed_out'] and not result['new_search_timed_out']
    assert not result['raw']['equal_values'] and not result['deterministic_search_equal']


def test_pairing_uses_id_and_slot_not_list_order_and_rejects_missing_duplicate():
    old=[slot(k=0),slot(k=1)];new=list(reversed(copy.deepcopy(old)))
    assert [r['slot'] for r in audit.pair_slots(old,new)]==[0,1]
    with pytest.raises(ValueError,match='every paired'):
        audit.pair_slots(old,new[:1])
    with pytest.raises(ValueError,match='every paired'):
        audit.pair_slots(old,[new[0],new[0]])


def test_value_equality_does_not_claim_byte_equality():
    result=audit.array_difference(np.zeros(1,dtype=np.float64),np.array([-0.],dtype=np.float64))
    assert result['equal_values'] and not result['equal_dtype_and_bytes']


def test_registered_node_cap_change_is_separate_from_actual_node_counts():
    old=slot();new=copy.deepcopy(old)
    old['attempt']['maximum_expanded_nodes']=20000;new['attempt']['maximum_expanded_nodes']=100000
    assert audit.compare_slot(old,new)['deterministic_search_equal']
    new['attempt']['expanded_nodes']=11
    assert not audit.compare_slot(old,new)['deterministic_search_equal']


def test_cap_transition_keeps_new_finite_invalid_and_changed_raw_history():
    old=slot();new=copy.deepcopy(old)
    old['attempt'].update(status='expanded_node_budget_exhausted',expanded_nodes=20000,complete_raw_paths_emitted=0)
    old.update(raw=np.empty((0,3)),h24=np.full((24,3),np.nan))
    old['candidate']['TipValid']=False;new['candidate']['TipValid']=False
    rows=audit.transition_slots([old],[new])
    assert rows[0]['new_raw_emitted']==1 and rows[0]['new_candidate']['TipValid'] is False
    assert not rows[0]['raw_difference']['equal_values']
    with pytest.raises(ValueError,match='transition'):
        audit.transition_slots([old],[])
