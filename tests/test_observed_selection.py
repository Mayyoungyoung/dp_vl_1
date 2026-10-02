"""Checkpoint selection changes evaluation only, including no-reference DEV."""
import json
import numpy as np
import pytest
import torch

from scripts.train_observed_geometry import (evaluate,checkpoint_selection_score,
    validate_selection_resume,POINT_FIELDS)
from scripts.train_observed_routes import observation_metrics


class FixedObservedModel:
    def __init__(self,paths):
        self.paths=torch.as_tensor(paths)
        self.calls=[]
    def eval(self):return self
    def __call__(self,**inputs):
        assert set(inputs)==set(POINT_FIELDS)|{'features','current'}
        self.calls.append({key:value.clone() for key,value in inputs.items()})
        count=len(inputs['features'])
        paths=self.paths[None].expand(count,-1,-1,-1).clone()
        return paths,torch.ones(paths.shape[:3]),{'anchor_xyz':paths[:,0,-1]}


def fixture(tmp_path):
    left=np.array([[0.,0.,.4],[-.3,0.,.15],[-.3,0.,-.15],[0.,0.,-.4]],np.float32)
    right=left.copy();right[:,0]*=-1
    paths=np.stack([left,right])
    specification=dict(centers=[[0.,0.,-.4]],target_index=0,tolerance=.03)
    current=np.tile(np.array([0.,0.,.4,0.,0.,0.,1.,1.],np.float32),(2,1))
    data=dict(features=np.zeros((2,4),np.float32),current=current,parent_ids=np.array(['p0','p1']),
        scene_ids=np.array(['p0_target0','p1_target0']),splits=np.array(['DEV_MODEL','DEV_MODEL']),
        paths=np.tile(left[None,None],(2,1,1,1)),events=np.ones((2,1,4),np.float32),
        path_mask=np.array([[True],[False]]),semantic_targets=[specification,specification],image_hashes=np.array(['a','b']))
    points=dict(world_xyz=np.zeros((2,1,3),np.float32),rgb=np.zeros((2,1,3),np.float32),uv=np.zeros((2,1,2),np.float32),
                depth=np.ones((2,1),np.float32),valid_mask=np.ones((2,1),bool))
    geometry=dict(points=points,index=np.arange(2))
    (tmp_path/'manifest.json').write_text(json.dumps(dict(acceptance=dict(tip_polyline_clearance_m=.02))))
    labels=[]
    for index in range(2):
        folder=tmp_path/('p'+str(index));folder.mkdir()
        np.savez(folder/'verification.npz',obstacle_centers=np.array([[0.,0.,0.]]),obstacle_halfsizes=np.array([[.1,.1,.1]]))
        labels.append(dict(id='p%d_target0'%index,parent_id='p%d'%index,split='DEV_MODEL',
            verification_only='p%d/verification.npz'%index,semantic_targets=specification,
            routes=['reference'] if index==0 else [],route_types=[['negative_x']] if index==0 else []))
    supervision=tmp_path/'supervision.jsonl';supervision.write_text(''.join(json.dumps(row)+'\n' for row in labels))
    return paths,data,geometry,(tmp_path/'observations.jsonl',supervision)


def test_default_evaluation_exact_original_formula_without_geometry_labels(tmp_path,monkeypatch):
    paths,data,geometry,sources=fixture(tmp_path)
    import scripts.train_observed_geometry as module
    def forbidden(*args,**kwargs):raise AssertionError('default path must not read extra geometry labels')
    monkeypatch.setattr(module,'add_tip_evaluation',forbidden)
    model=FixedObservedModel(paths)
    result=evaluate(model,data,geometry,np.arange(2),'cpu')
    predictions=np.tile(paths[None],(2,1,1,1))
    expected,_=observation_metrics(predictions,np.ones((2,2,4)),data,np.arange(2))
    expected.update(learned_surface_anchor_reference_endpoint_error_m=0.,paired_language_control=None,
                    selection_score=-expected['candidate_matched_ADE_m'])
    assert result==expected
    explicit=evaluate(model,data,geometry,np.arange(2),'cpu',selection_metric='reference_ADE',evaluation_sources=sources)
    assert explicit==expected


def test_tip_selection_keeps_no_reference_inputs_and_formula(tmp_path):
    paths,data,geometry,sources=fixture(tmp_path)
    result=evaluate(FixedObservedModel(paths),data,geometry,np.arange(2),'cpu',selection_metric='tip_unique_valid',evaluation_sources=sources)
    assert result['examples']==result['tip_evaluation_examples']==2
    assert result['reference_evaluation_examples']==1 and result['examples_without_reference']==1
    assert result['UniqueClassifiedTipValidAtK']==2 and result['TipValidAtK']==1
    assert result['selection_score']==2.05
    # Unknown but valid paths retain the secondary validity contribution.
    assert checkpoint_selection_score({'UniqueClassifiedTipValidAtK':0.,'TipValidAtK':1.},'tip_unique_valid')==.05


def test_geometry_labels_read_after_forward_never_enter_inputs(tmp_path,monkeypatch):
    paths,data,geometry,sources=fixture(tmp_path);model=FixedObservedModel(paths)
    original_load=np.load;loads=[]
    def audited_load(path,*args,**kwargs):
        assert model.calls,'privileged boxes must only open after prediction'
        loads.append(str(path))
        return original_load(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',audited_load)
    first=evaluate(model,data,geometry,np.arange(2),'cpu',selection_metric='tip_unique_valid',evaluation_sources=sources)
    for parent in ('p0','p1'):
        np.savez(tmp_path/parent/'verification.npz',obstacle_centers=np.array([[0.,0.,0.]]),obstacle_halfsizes=np.array([[10.,10.,10.]]))
    second=evaluate(model,data,geometry,np.arange(2),'cpu',selection_metric='tip_unique_valid',evaluation_sources=sources)
    assert first['selection_score']==2.05 and second['selection_score']==0
    assert len(loads)==4 and len(model.calls)==2
    assert all(torch.equal(model.calls[0][key],model.calls[1][key]) for key in model.calls[0])


def test_selection_resume_rejects_changed_protocol_and_accepts_history():
    validate_selection_resume({'checkpoint_selection':'reference_ADE'}, {})
    validate_selection_resume({'checkpoint_selection':'tip_unique_valid'}, {'checkpoint_selection':'tip_unique_valid'})
    with pytest.raises(ValueError,match='checkpoint_selection'):
        validate_selection_resume({'checkpoint_selection':'tip_unique_valid'}, {})
    with pytest.raises(ValueError,match='checkpoint_selection'):
        validate_selection_resume({'checkpoint_selection':'reference_ADE'}, {'checkpoint_selection':'tip_unique_valid'})
    with pytest.raises(ValueError,match='unsupported'):
        checkpoint_selection_score({},'unregistered')


def test_selection_resume_rejects_changed_evaluation_interval():
    for protocol in ('reference_ADE','tip_unique_valid'):
        saved=dict(checkpoint_selection=protocol,eval_every=250)
        validate_selection_resume(dict(saved),saved)
        validate_selection_resume(saved,dict(checkpoint_selection=protocol))
        for interval in (100,500):
            with pytest.raises(ValueError,match='eval_every'):
                validate_selection_resume(dict(saved,eval_every=interval),saved)
            with pytest.raises(ValueError,match='eval_every'):
                validate_selection_resume(dict(checkpoint_selection=protocol,eval_every=interval),
                                          dict(checkpoint_selection=protocol))
