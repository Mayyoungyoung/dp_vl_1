import numpy as np
import pytest
from scripts.audit_two_row_last_train import select_train,summarize_residual


def data():
    ids=['two_row_reach_%d_target%d'%(p,t) for p in range(283200,283216) for t in range(3)]
    return dict(scene_ids=np.array(ids+['two_row_reach_283264_target0']),
        splits=np.array(['TRAIN']*48+['DEV_MODEL']),path_mask=np.ones((49,9),bool))


def test_exact_train_and_no_dev_selected():
    d=data();assert select_train(d).tolist()==list(range(48))
    d['splits'][-1]='TRAIN'
    with pytest.raises(ValueError):select_train(d)


def test_missing_positive_or_wrong_parent_rejected():
    d=data();d['path_mask'][0]=False
    with pytest.raises(ValueError):select_train(d)
    d=data();d['scene_ids'][0]=d['scene_ids'][1]
    with pytest.raises(ValueError):select_train(d)


def test_residual_is_full_vertex_distance_not_endpoint_only():
    pred=np.array([[0.,0.,0.],[0.,3.,4.],[0.,0.,2.]])
    r=summarize_residual(pred,np.zeros_like(pred))
    assert r['per_vertex_distance_m']==[0.,5.,2.]
    assert r['max_vertex_distance_m']==5 and r['endpoint_distance_m']==2
