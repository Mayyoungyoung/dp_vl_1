import numpy as np
import pytest
from scripts.analyze_multitask_event_locations import compare,grouped_metrics,validate_prediction_shapes


def test_missing_close_is_failed_location_and_no_close_reference_is_separate():
    path=np.zeros((1,4,3))
    result=compare(path,np.ones((1,4)),path,np.array([[1,1,0,0]]))[0]
    assert result['presence_match'] is False
    assert result['close_within_3cm'] is False and result['close_location_error_m'] is None
    noevent=compare(path,np.ones((1,4)),path,np.ones((1,4)))[0]
    assert noevent['presence_match'] is True and noevent['close_within_3cm'] is None


def test_matching_uses_path_ade_not_nearest_event_and_uses_actual_close_frame():
    prediction=np.zeros((1,4,3));prediction[0,3,0]=.4
    refs=np.concatenate((prediction.copy(),prediction.copy()+1))
    events=np.array([[1,0,0,0],[1,1,1,0]])
    result=compare(prediction,np.array([[1,1,1,0]]),refs,events)[0]
    assert result['reference_index']==0 and result['absolute_close_index_error']==2
    assert result['close_location_error_m']==pytest.approx(.4)
    with pytest.raises(ValueError):compare(prediction*np.nan,np.ones((1,4)),refs,events)


def test_absent_references_retained_and_macro_does_not_weight_paraphrases():
    path=np.zeros((1,4,3));opened=np.ones((1,4))
    absent=compare(path,opened,np.empty((0,4,3)),np.empty((0,4)))[0]
    assert absent['reference_available'] is False and absent['presence_match'] is None
    positive=dict(absent,presence_match=True,close_location_error_m=.1,close_within_3cm=False,absolute_close_index_error=0)
    negative=dict(positive,close_location_error_m=.9)
    rows=[dict(task='t',parent_id='a',candidates=[positive]) for _ in range(4)]
    rows.append(dict(task='t',parent_id='b',candidates=[negative]))
    assert grouped_metrics(rows)['macro']['close_location_error_m']==pytest.approx(.5)


def test_extra_rows_or_dropped_candidate_slots_cannot_change_budget():
    saved=dict(paths=np.zeros((48,4,24,3)),gripper_open=np.ones((48,4,24)),scene_ids=np.zeros(48),parent_ids=np.zeros(48))
    validate_prediction_shapes(saved,48,4,24)
    for bad in (dict(saved,paths=np.zeros((49,4,24,3))),dict(saved,paths=np.zeros((48,3,24,3))),
                dict(saved,parent_ids=np.zeros((48,1)))):
        with pytest.raises(ValueError,match='budget'):validate_prediction_shapes(bad,48,4,24)
