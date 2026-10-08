import numpy as np
from scripts.research_v3_train_edit_retention import nearest_mean_point_distance


def test_distance_is_pointwise_mean_and_handles_missing_mode():
    path=np.zeros((24,3));near=np.zeros((24,3));near[:,0]=.03
    far=np.zeros((24,3));far[:12,1]=.08
    assert np.isclose(nearest_mean_point_distance(path,np.array([far,near])),.03)
    assert np.isclose(nearest_mean_point_distance(path,np.array([far])),.04)
    assert nearest_mean_point_distance(path,np.empty((0,24,3))) is None
