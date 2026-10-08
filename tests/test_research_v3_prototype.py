import numpy as np
from scripts.research_v3_prototype import strided_loader


def test_stride_preserves_point_correspondence_and_exposes_no_new_fields():
    rgb=np.arange(4*6*3).reshape(4,6,3);xyz=rgb+.5;valid=np.ones((4,6),bool)
    valid[2,2]=False
    def load(*args):return (rgb,xyz,valid),['rgb','depth']
    grid,files=strided_loader(load)('observed-only')
    assert len(grid)==3 and files==['rgb','depth']
    np.testing.assert_array_equal(grid[0],rgb[::2,::2])
    np.testing.assert_array_equal(grid[1],xyz[::2,::2])
    assert not grid[2][1,1]
