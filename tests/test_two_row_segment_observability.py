"""Pure contracts; no repository model, real data, server, or Torch calls."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import audit_two_row_segment_observability as probe


def policy():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/two_row_segment_observability_v1.json').read_text())


def test_fixed_population_and_budget():
    cfg = policy(); probe.validate_config(cfg)
    assert len(probe.IDS) == 48
    assert set(probe.FIT_PARENTS).isdisjoint(probe.HOLDOUT_PARENTS)
    assert cfg['geometry_forward_budget'] == 48
    assert (48*4+cfg['positive_references'])*23 == 10971
    for key, value in [('ridge_lambda',.1),('geometry_forward_budget',96),('pool_stage','best'),('prediction_threshold',.6)]:
        changed=copy.deepcopy(cfg); changed[key]=value
        with pytest.raises(ValueError): probe.validate_config(changed)


@pytest.mark.parametrize('field', ['parents','fit_parents','holdout_parents'])
def test_no_replacement_parent(field):
    cfg=policy(); cfg[field][-1]='two_row_reach_283264'
    with pytest.raises(ValueError): probe.validate_config(cfg)


def test_opaque_unselected_json(tmp_path):
    p=tmp_path/'rows.jsonl'
    rows=[json.dumps(dict(id=i,parent_id=i.rsplit('_target',1)[0],split='TRAIN')) for i in probe.IDS]
    # Valid canonical ID header, deliberately undecodable body: must never parse.
    rows.insert(0,'{"id": "two_row_reach_283264_target0", UNREADABLE_DEV_PAYLOAD')
    p.write_text('\n'.join(rows)+'\n')
    assert len(probe.selected_rows(p)) == 48
    rows[-1]=rows[-1].replace('TRAIN','TEST_LOCKED'); p.write_text('\n'.join(rows)+'\n')
    with pytest.raises(ValueError): probe.selected_rows(p)


def test_cache_metadata_selection_without_role_payload(tmp_path):
    p=tmp_path/'cache.jsonl'
    p.write_text(''.join(json.dumps(dict(id=i,file=i+'.npz',sha256='x'))+'\n' for i in probe.IDS))
    assert len(probe.selected_rows(p,identity=False)) == 48
    p.write_text(p.read_text()+json.dumps(dict(id=probe.IDS[0]))+'\n')
    with pytest.raises(ValueError): probe.selected_rows(p,identity=False)


def test_path_escape_and_wrong_role_rejected(tmp_path):
    with pytest.raises(ValueError): probe.child(tmp_path,'../escape')
    with pytest.raises(ValueError): probe.train_path(tmp_path,'two_row_reach_283264','route.npz')


def test_symlink_escape_rejected(tmp_path):
    base=tmp_path/'base'; outside=tmp_path/'outside'; base.mkdir(); outside.mkdir()
    link=base/'redirect'
    try: link.symlink_to(outside,target_is_directory=True)
    except (OSError,NotImplementedError): pytest.skip('Symlinks unavailable on this host')
    with pytest.raises(ValueError): probe.child(base,'redirect/file')


def pool_arrays():
    names=np.array(['two_row_reach_%d_target%d'%(p,t) for p in list(range(283200,283264))+list(range(400000,400032)) if p != 283220 for t in range(3)])
    return dict(scene_ids=names,parent_ids=np.array([s.rsplit('_target',1)[0] for s in names]),
        paths=np.zeros((285,4,24,3)),gripper_open=np.ones((285,4,24)))


def test_sealed_train_pool_keeps_failures_but_rejects_dev(tmp_path):
    p=tmp_path/'pool.npz'; arrays=pool_arrays(); arrays['paths'][0,0,2,0]=np.nan
    np.savez(p,**arrays); values=probe.load_pool(p)
    assert len(values) == 48 and np.isnan(values[probe.IDS[0]][0][0,2,0])
    arrays['scene_ids'][0]='two_row_reach_283264_target0'; np.savez(p,**arrays)
    with pytest.raises(ValueError): probe.load_pool(p)


def test_signed_camera_ray_unknown_is_not_free():
    k=np.array([[-2.,0,2],[0,-2.,2],[0,0,1.]])
    depth=np.full((5,5),2.); depth[0,0]=np.nan
    points=np.array([[0,0,1.],[0,0,3.],[0,0,-1.],[10,0,1.],[1,1,1.]])
    result=probe.ray_descriptor(points,depth,k,np.eye(4))
    np.testing.assert_array_equal(result[:,1:],[ [1,1],[0,1],[0,0],[0,0],[0,0]])
    assert result[1,0] == -.25 and result[2,0] == 0


def test_local_representation_depends_on_alignment_and_no_labels():
    segments=np.array([[[0.,0,1],[.2,0,1]],[[.5,0,1],[.7,0,1]]])
    xyz=np.array([[0,0,1.],[.1,0,1.],[.2,0,1.]])
    local=probe.local_features(segments,xyz,np.full((5,5),2.),np.eye(3),np.eye(4))
    assert local.shape == (2,40) and np.isfinite(local).all()
    assert local[0,0] == 0 and local[1,0] == .25
    coordinates=probe.segment_coordinates(segments,np.zeros(8))
    assert coordinates.shape == (2,20)
    with pytest.raises(TypeError): probe.local_features(segments,xyz,np.ones((5,5)),np.eye(3),np.eye(4),labels=np.ones(2))


def test_shuffle_is_label_free_derangement_inside_condition():
    ids=['a']*5+['b']*4
    local=np.arange(9*40).reshape(9,40)
    shuffled,index=probe.shuffle_local(local,ids)
    assert np.all(index != np.arange(9))
    assert [ids[i] for i in index] == ids
    np.testing.assert_array_equal(np.sort(shuffled[:5],axis=0),np.sort(local[:5],axis=0))
    np.testing.assert_array_equal(shuffled,probe.shuffle_local(local,ids)[0])


def test_same_dim_and_context_preserved_all_three_arms():
    c=np.ones((4,128)); s=np.zeros((4,20)); loc=np.arange(160).reshape(4,40)
    neg,_=probe.shuffle_local(loc,['a']*4)
    design=[probe.design_matrix(c,s,loc,a,neg) for a in probe.ARMS]
    assert {d.shape for d in design} == {(4,188)}
    for d in design: np.testing.assert_array_equal(d[:,:148],np.c_[c,s])
    assert not design[0][:,148:].any()
    assert not np.array_equal(design[1],design[2])


def record(parent,condition,source,route,segment):
    return dict(parent_id=parent,id=condition,source=source,route=route,segment=segment,
        semantic_correct=False,unknown_type=True,tip_valid=False)


def test_weights_do_not_let_many_refs_or_segments_dominate():
    rows=[]
    for parent in ('p','q'):
        for source,count in (('generated_last',4),('positive_H24',9)):
            for route in range(count):
                rows.extend(record(parent,parent+'_t',source,route,i) for i in range(route+1))
    weights=probe.fitting_weights(rows)
    assert weights.sum() == pytest.approx(1)
    for parent in ('p','q'):
        for source in ('generated_last','positive_H24'):
            assert sum(w for w,r in zip(weights,rows) if r['parent_id']==parent and r['source']==source) == pytest.approx(.25)


def test_closed_form_fit_scaling_uses_only_fit_and_is_reproducible():
    x=np.array([[0.,0],[1,0],[2,0],[3,0]])
    y=np.array([0.,0,1,1]); w=np.ones(4)/4
    first=probe.fit_ridge(x,y,w); second=probe.fit_ridge(x,y,w)
    for key in first: np.testing.assert_array_equal(first[key],second[key])
    assert first['mean'][0] == 1.5 and first['scale'][1] == 1
    before=first['mean'].copy(); probe.predict(first,np.array([[1e9,-1e9]]))
    np.testing.assert_array_equal(first['mean'],before)
    with pytest.raises(ValueError): probe.fit_ridge(x,np.zeros(4),w)


def test_auc_ties_single_class_and_fixed_threshold():
    assert probe.auc(np.array([0,1]),np.array([.5,.5])) == .5
    assert probe.auc(np.array([0,1]),np.array([.2,.8])) == 1
    assert probe.auc(np.array([1,1]),np.array([.2,.8])) is None
    assert probe.measures(np.array([0,1]),np.array([.5,.5]))['balanced_accuracy'] == .5


def test_wrong_targets_and_unknowns_are_not_dropped():
    rows=[record(probe.HOLDOUT_PARENTS[0],'id','generated_last',0,i) for i in range(2)]
    result=probe.evaluate(rows,np.array([0.,1.]),np.array([.1,.9]))['holdout']
    assert result['generated']['count'] == result['wrong_target']['count'] == result['unknown_type']['count'] == 2
    assert result['generated_correct_target']['count'] == 0


def test_screen_requires_both_controls_and_support():
    def metric(a,b): return dict(eligible_parents=2,collisions=20,clear=20,parent_macro_auc=a,parent_macro_brier=b)
    results={a:dict(holdout=dict(generated=metric(.7,.2))) for a in probe.ARMS}
    results['aligned']['holdout']['generated']=metric(.8,.18)
    limits=policy()['screen']; assert probe.screen(results,limits)['passed']
    results['shuffled']['holdout']['generated']=metric(.79,.181)
    assert not probe.screen(results,limits)['passed']
    results['aligned']['holdout']['generated']['collisions']=19
    assert probe.screen(results,limits)['decision'] == 'stop_underpowered'


def test_import_does_not_load_torch_transformers_or_simulator():
    code='import sys; import scripts.audit_two_row_segment_observability; assert not any(k in sys.modules for k in ("torch","transformers","pyrep","rlbench"))'
    result=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True)
    assert result.returncode == 0, result.stderr


def test_runtime_rejects_visible_gpu_before_output(tmp_path,monkeypatch):
    p=tmp_path/'config.json'; p.write_text(json.dumps(policy()))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    with pytest.raises(ValueError,match='hide GPU'): probe.run(p,tmp_path/'result')
    assert not (tmp_path/'result').exists()
