"""Analysis rejects unfair pairs and retains null-reference/failed outputs."""
import copy
import json
import numpy as np
import pytest

from scripts.analyze_observed_obstacle_pair import paired_config, require_close, evaluate_saved
from test_observed_selection import fixture, FixedObservedModel
from scripts.train_observed_geometry import evaluate


def test_pair_config_only_ignores_documented_nontraining_differences():
    original=dict(output='a',anchor_mode='soft',lr=.0003,seed=0,
        geometry_preprocessing=dict(load_preprocess_total_s=1.,source_hashes={'image':'abc'}))
    other=copy.deepcopy(original);other.update(output='b',anchor_mode='straight_through_peak')
    other['geometry_preprocessing']['load_preprocess_total_s']=5.
    assert paired_config(original)==paired_config(other)
    other['lr']=.0004
    assert paired_config(original)!=paired_config(other)
    other['lr']=original['lr'];other['geometry_preprocessing']['source_hashes']['image']='different'
    assert paired_config(original)!=paired_config(other)


def test_metric_comparison_preserves_null_denominators():
    require_close(None,None,'no references')
    with pytest.raises(ValueError,match='null mismatch'):
        require_close(None,0.,'no references')
    with pytest.raises(ValueError,match='metric mismatch'):
        require_close(.5,.6,'changed result')


def test_saved_recompute_keeps_all_inputs_and_zero_reference(tmp_path):
    paths,data,points,sources=fixture(tmp_path)
    folder=tmp_path/'predictions'
    evaluate(FixedObservedModel(paths),data,points,np.arange(2),'cpu',folder,
             selection_metric='tip_unique_valid',evaluation_sources=sources)
    rows=[dict(id='p%d_target0'%i,parent_id='p%d'%i,instruction='reach red') for i in range(2)]
    labels={r['id']:r for r in map(json.loads,sources[1].read_text().splitlines())}
    current={r['id']:dict(gripper_pose=data['current'][i,:7],gripper_open=data['current'][i,7]) for i,r in enumerate(rows)}
    geometry={r['id']:dict(obstacle_centers=np.array([[0.,0.,0.]]),obstacle_halfsizes=np.array([[.1,.1,.1]])) for r in rows}
    result,per_scene,failure=evaluate_saved(folder,rows,data,current,geometry,labels,2,tmp_path/'analysis')
    assert result['semantic_evaluation_examples']==result['tip_evaluation_examples']==2
    assert result['reference_evaluation_examples']==1 and per_scene[1]['candidate_matched_ADE_m'] is None
    assert result['TipValidAtK']==1 and result['selection_score']==2.05
    assert failure['no_reference']['candidates']==2 and failure['no_reference']['tip_valid_candidates']==2
    with np.load(folder/'predictions.npz') as archive:
        arrays={key:archive[key][:1] for key in archive.files}
    np.savez(folder/'predictions.npz',**arrays)
    with pytest.raises(ValueError,match='all DEV inputs'):
        evaluate_saved(folder,rows,data,current,geometry,labels,2,tmp_path/'must_not_exist')


def test_saved_recompute_rejects_changed_recorded_score(tmp_path):
    paths,data,points,sources=fixture(tmp_path);folder=tmp_path/'predictions'
    evaluate(FixedObservedModel(paths),data,points,np.arange(2),'cpu',folder,
             selection_metric='tip_unique_valid',evaluation_sources=sources)
    recorded=json.loads((folder/'metrics.json').read_text());recorded['selection_score']=99.
    (folder/'metrics.json').write_text(json.dumps(recorded))
    rows=[dict(id='p%d_target0'%i,parent_id='p%d'%i,instruction='reach red') for i in range(2)]
    labels={r['id']:r for r in map(json.loads,sources[1].read_text().splitlines())}
    current={r['id']:dict(gripper_pose=data['current'][i,:7],gripper_open=data['current'][i,7]) for i,r in enumerate(rows)}
    geometry={r['id']:dict(obstacle_centers=np.array([[0.,0.,0.]]),obstacle_halfsizes=np.array([[.1,.1,.1]])) for r in rows}
    with pytest.raises(ValueError,match='selection_score'):
        evaluate_saved(folder,rows,data,current,geometry,labels,2,tmp_path/'must_not_exist')
