import numpy as np
import pytest
from routeset.geometric_modes import portal_word, summarize_modes

C = dict(row_x=[1., 2.], post_y=[[-1., 1.], [-1., 1.]],
         post_size_xyz=[.2, .2, 1.], post_base_z=0., tip_clearance_m=.02)

def test_jitter_and_vertex_sampling_same_corridor():
    a = np.array([[0, 0, .5], [3, 0, .5]])
    b = np.array([[0, 0, .5], [1, .02, .6], [2, -.02, .3], [3, 0, .5]])
    assert portal_word(a,C) == portal_word(b,C)

def test_distinct_3d_routes_same_endpoints():
    a = [[0,0,.5],[.5,0,.5],[2.5,0,.5],[3,0,.5]]
    b = [[0,0,.5],[.5,-1,1.3],[2.5,-1,1.3],[3,0,.5]]
    assert portal_word(a,C) != portal_word(b,C)

def test_small_backtrack_cancels():
    a = [[0,0,.5],[3,0,.5]]
    b = [[0,0,.5],[1.1,0,.5],[.9,0,.5],[3,0,.5]]
    assert portal_word(a,C) == portal_word(b,C)

def test_alternate_portal_backtrack_retained():
    a = [[0,0,.5],[1.2,0,.5],[1.2,2,.5],[.8,2,.5],[.8,0,.5],[3,0,.5]]
    assert len(portal_word(a,C)) == 4

def test_blocked_portal_fails():
    with pytest.raises(ValueError): portal_word([[0,1,.5],[3,1,.5]],C)

def test_unknown_duplicates_and_failed_routes_not_new_modes():
    result = summarize_modes([True,True,False],['a','a','b'],['a','b'],range(3))
    assert result == dict(ValidCount=2,GeometricModeCount=1,TwoDistinctValid=0,ReferenceModesHit=1,ReferenceModeCoverage=.5)
