import numpy as np
from scripts.research_v3_audit import exact_clear, probes_clear


def test_continuous_multibox_check_finds_collision_between_clear_probes():
    paths = np.array([[[0.,0.,0.],[1.,0.,0.]],[[0.,1.,0.],[1.,1.,0.]]])
    cs = np.array([[.25,0.,0.],[.75,2.,0.]])
    hs = np.ones((2,3))*.01
    assert probes_clear(paths,cs,hs).tolist()==[True,True]
    assert exact_clear(paths,cs,hs).tolist()==[False,True]
    refined = np.stack([paths[:,0],(paths[:,0]+paths[:,1])/2,paths[:,1]],1)
    np.testing.assert_array_equal(exact_clear(paths,cs,hs),exact_clear(refined,cs,hs))
