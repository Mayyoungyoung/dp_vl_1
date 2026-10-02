import json
import numpy as np
import pytest

from routeset.vlm_route_serialization import (depth_image,prompt,serialize_paths,parse_paths,assistant_only_labels)


def test_roundtrip_and_millimeter_error_does_not_repair_start_or_filter_duplicates():
    rng=np.random.default_rng(19)
    paths=rng.uniform(-.5,1,(4,24,3));paths[3]=paths[0]
    events=np.ones((4,24));events[:,9:]=0
    xyz,opened,receipt=parse_paths(serialize_paths(paths,events),4,24)
    assert np.max(np.abs(xyz-paths))<=.0005001
    np.testing.assert_array_equal(xyz[3],xyz[0]);np.testing.assert_array_equal(opened,events)
    assert receipt['charged_candidate_slots']==receipt['format_valid_candidates']==4


def test_missing_malformed_and_extra_candidates_all_remain_in_budget():
    route=[[100,200,300,1]]*3
    paths,_,receipt=parse_paths(json.dumps([route,[[100,200,True,1]]*3]),4,3)
    assert np.isfinite(paths[0]).all() and np.isnan(paths[1:]).all()
    assert receipt['format_valid_candidates']==1 and receipt['charged_candidate_slots']==4
    paths,_,receipt=parse_paths(json.dumps([route]*5),4,3)
    assert np.isnan(paths).all() and receipt['budget_exceeded'] and receipt['charged_candidate_slots']==5
    paths,_,receipt=parse_paths('```json\n[]\n```',4,3)
    assert np.isnan(paths).all() and receipt['charged_candidate_slots']==4
    paths,_,receipt=parse_paths(json.dumps([[[-9223372036854775808,0,0,1]]*3]),1,3)
    assert np.isnan(paths).all() and receipt['format_valid_candidates']==0


def test_depth_validity_and_quantization_without_hidden_clipping():
    original=np.array([[1.23456,0.,np.nan],[2.5012,-1.,np.inf]])
    image=depth_image(original)
    decoded=(image[...,0].astype(float)*256+image[...,1])/1000
    valid=np.isfinite(original)&(original>0)
    assert np.array_equal(image[...,2]==255,valid)
    assert np.max(np.abs(decoded[valid]-original[valid]))<.0005
    assert not image[~valid].any()
    with pytest.raises(ValueError,match='clip'):depth_image(np.array([[66.]]))


def test_actual_prefix_is_entirely_masked_and_mismatch_rejected():
    prefix=np.array([151644,300,900,800,151645,151644,77091,198])
    answer=np.array([58,16,11,17,60,151645])
    labels=assistant_only_labels(np.concatenate([prefix,answer]),prefix)
    assert (labels[:len(prefix)]==-100).all()
    np.testing.assert_array_equal(labels[len(prefix):],answer)
    with pytest.raises(ValueError,match='prefix differs'):
        assistant_only_labels(np.concatenate([prefix,answer]),np.append(prefix[:-1],3))


def test_prompt_contains_only_supplied_current_fields_and_changes_requested_budget():
    current=np.array([.1,.2,.3,0.,0.,0.,1.,1.])
    text=prompt('reach the red sphere',current,np.eye(3),np.eye(4),4,24)
    assert 'exactly 4 routes' in text and 'exactly 24 points' in text
    assert 'true_target' not in text and 'future_path' not in text
    with pytest.raises(ValueError,match='current observation'):
        prompt('reach red',current,np.full((3,3),np.nan),np.eye(4),4,24)
