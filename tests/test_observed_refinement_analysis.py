import copy
import json
import numpy as np
import pytest

from scripts.analyze_observed_refinement_pair import paired_config,evaluate_drafts,validate_reference_peak
from scripts.export_observation_roles import digest
from test_observed_selection import fixture


def test_matched_config_only_allows_pooling_mode_change():
    first=dict(output='a',anchor_mode='straight_through_peak',refinement_mode='global',
               refinement_sigma=.1,refinement_prefix_fraction=.5,refinement_bound=.1,lr=.0003)
    second=dict(first,output='b',refinement_mode='local')
    assert paired_config(first)==paired_config(second)
    for key,value in [('refinement_sigma',.2),('refinement_prefix_fraction',.25),('refinement_bound',.2),('lr',.0004)]:
        changed=dict(second);changed[key]=value
        assert paired_config(first)!=paired_config(changed)


def test_draft_metric_includes_every_saved_candidate_and_no_reference(tmp_path):
    paths,data,_,sources=fixture(tmp_path)
    draft=np.tile(paths[None],(2,1,1,1));draft[1,0,:,0]=0.
    # First zero-reference candidate now crosses the physical box. Its endpoint
    # remains correct and it must still be counted among submitted drafts.
    file=tmp_path/'saved.npz'
    np.savez(file,draft_paths=draft,paths=np.tile(paths[None],(2,1,1,1)),
             gripper_open=np.ones((2,2,4)),scene_ids=data['scene_ids'],parent_ids=data['parent_ids'])
    rows=[dict(id='p%d_target0'%i,parent_id='p%d'%i,instruction='reach red') for i in range(2)]
    labels={row['id']:row for row in map(json.loads,sources[1].read_text().splitlines())}
    current={row['id']:dict(gripper_pose=data['current'][i,:7],gripper_open=data['current'][i,7]) for i,row in enumerate(rows)}
    geometry={row['id']:dict(obstacle_centers=np.array([[0.,0.,0.]]),obstacle_halfsizes=np.array([[.1,.1,.1]])) for row in rows}
    result=evaluate_drafts(file,rows,data,current,geometry,labels,tmp_path/'draft_analysis')
    assert result['metrics']['semantic_evaluation_examples']==result['metrics']['tip_evaluation_examples']==2
    assert result['metrics']['reference_evaluation_examples']==1
    assert result['metrics']['total_submitted_candidate_budget']==4
    assert result['metrics']['semantic_goal_accuracy']==1.
    assert result['metrics']['TipValidAtK']==.75
    assert result['failure_breakdown']['independent_gate_failures']['tip_segments_clear']==1


def test_reference_rejects_soft_or_changed_actual_artifacts(tmp_path):
    for filename in ('best.pt','last.pt','dev_model/predictions.npz','last_dev_model/predictions.npz'):
        file=tmp_path/filename;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(filename.encode())
    summary=dict(best_checkpoint_sha256=digest(tmp_path/'best.pt'),last_checkpoint_sha256=digest(tmp_path/'last.pt'),
        prediction_sha256=digest(tmp_path/'dev_model/predictions.npz'),last_prediction_sha256=digest(tmp_path/'last_dev_model/predictions.npz'))
    config=dict(anchor_mode='straight_through_peak')
    validate_reference_peak(tmp_path,config,summary)
    with pytest.raises(ValueError,match='not soft'):
        validate_reference_peak(tmp_path,dict(anchor_mode='soft'),summary)
    with pytest.raises(ValueError,match='not soft'):
        validate_reference_peak(tmp_path,dict(config,refinement_mode='local'),summary)
    (tmp_path/'last_dev_model/predictions.npz').write_bytes(b'changed')
    with pytest.raises(ValueError,match='SHA'):
        validate_reference_peak(tmp_path,config,summary)
