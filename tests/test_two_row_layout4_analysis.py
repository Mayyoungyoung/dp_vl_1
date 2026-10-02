import numpy as np
from scripts.analyze_two_row_layout4 import obb_overlap


def box(position,quaternion=(0,0,0,1),half=(.5,.5,.5)):
    return dict(pose=list(position)+list(quaternion),bounding_box=np.stack([-np.asarray(half),half],axis=1).ravel())


def test_bbox_overlap_includes_contact_but_not_separated_boxes():
    assert obb_overlap(box([0,0,0]),box([1,0,0]))
    assert not obb_overlap(box([0,0,0]),box([1.001,0,0]))
    assert obb_overlap(box([0,0,0]),box([.4,.2,0]))


def test_bbox_rotation_and_offset_are_respected():
    half=(1,.1,.1);rot=(0,0,np.sqrt(.5),np.sqrt(.5))
    assert obb_overlap(box([0,0,0],rot,half),box([0,.8,0],half=(.1,.1,.1)))
    assert not obb_overlap(box([0,0,0],rot,half),box([.8,0,0],half=(.1,.1,.1)))
