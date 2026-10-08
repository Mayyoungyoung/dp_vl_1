import numpy as np
from routeset.research_v3_modes import passage_signature
from routeset.geometric_modes import portal_word


def test_above_and_middle_are_distinct_and_boundary_strip_is_classified():
    cfg=dict(row_x=[0.],post_y=[[-.1,.1]],post_heights=[.15],post_base_z=.755,tip_clearance_m=.02)
    low=np.array([[-.1,0.,.84],[.1,0.,.84]])
    high=low.copy();high[:,2]=.94
    assert passage_signature(low,cfg)==('gap1',)
    assert passage_signature(high,cfg)==('over',)
    assert portal_word(low,cfg)==portal_word(high,cfg)  # Preserved historical metric caveat.
    near=low.copy();near[:,2]=.90
    assert passage_signature(near,cfg)==('gap1',)
    split=np.vstack([high[0],high.mean(0),high[1]])
    assert passage_signature(split,cfg)==passage_signature(high,cfg)


def test_closed_gap_is_not_a_certificate_that_over_is_closed():
    cfg=dict(row_x=[0.],post_y=[[-.028,.028]],post_heights=[.15],post_base_z=.755,tip_clearance_m=.02)
    low=np.array([[-.1,0.,.84],[.1,0.,.84]])
    assert passage_signature(low,cfg) is None
    low[:,2]=.94
    assert passage_signature(low,cfg)==('over',)
