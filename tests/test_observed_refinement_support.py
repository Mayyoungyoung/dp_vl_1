import numpy as np
import pytest
from scripts.audit_observed_refinement_support import support_at_draft,aggregate,read_saved_stage
from scripts.export_observation_roles import digest


def test_only_valid_visible_points_define_support_and_keep_missing_support():
    draft=np.zeros((1,6,3),np.float32);draft[0,:,0]=np.linspace(0,1,6)
    points=np.array([[.2,0,0],[.21,0,0],[.4,0,0],[np.nan,0,0]])
    mask=np.array([True,True,False,False])
    rows=support_at_draft(draft,draft.copy(),points,mask,.03,.5)
    assert rows[0]['nearest_visible_point_m']<1e-6 and rows[0]['visible_points_within_sigma']==2
    assert rows[1]['visible_points_within_sigma']==rows[1]['visible_points_within_2sigma']==0
    result=aggregate(rows)
    assert result['eligible_queries']==2 and result['eligible_without_points_within_2sigma']==1
    assert result['actual_update_l2_m']['max']==0


def test_saved_endpoint_and_nonprefix_mutation_rejected():
    draft=np.zeros((1,6,3),np.float32);draft[0,:,0]=np.linspace(0,1,6)
    points=np.array([[.2,0,0]]);mask=np.ones(1,bool)
    changed=draft.copy();changed[0,-1,1]=.1
    with pytest.raises(ValueError,match='endpoint'):
        support_at_draft(draft,changed,points,mask,.03,.5)
    changed=draft.copy();changed[0,4,1]=.1
    with pytest.raises(ValueError,match='non-prefix'):
        support_at_draft(draft,changed,points,mask,.03,.5)


def test_saved_stage_rejects_truncation_parent_budget_and_changed_sha(tmp_path):
    path=tmp_path/'prediction.npz';by_id={'a':{'parent_id':'p0'},'b':{'parent_id':'p1'}}
    arrays=dict(scene_ids=np.array(['a','b']),parent_ids=np.array(['p0','p1']),
                draft_paths=np.zeros((2,2,4,3)),paths=np.zeros((2,2,4,3)))
    np.savez(path,**arrays);correct=digest(path)
    assert len(read_saved_stage(path,by_id,2,4,correct)[0])==2
    with pytest.raises(ValueError,match='SHA'):
        read_saved_stage(path,by_id,2,4,'wrong')
    with pytest.raises(ValueError,match='exact saved K/H'):
        read_saved_stage(path,by_id,4,4,correct)
    shortened=dict(arrays,draft_paths=arrays['draft_paths'][:1],paths=arrays['paths'][:1])
    np.savez(path,**shortened)
    with pytest.raises(ValueError,match='all DEV'):
        read_saved_stage(path,by_id,2,4,digest(path))
    mismatched=dict(arrays,parent_ids=np.array(['p1','p0']))
    np.savez(path,**mismatched)
    with pytest.raises(ValueError,match='parent identity'):
        read_saved_stage(path,by_id,2,4,digest(path))
