"""Tiny CPU checks only; these synthetic features are not experiment evidence."""
import argparse
import copy
import json
import numpy as np
import pytest
import torch
from routeset.common import seed_all
from routeset.observed_geometry import ObservedGeometryRouteHead,positive_endpoint_attention_loss
from routeset.observed_grounding_targets import prepare_event_grounding_targets
from routeset.observed_multitask import draw_observation_batch
from routeset.observed_route_head import load_observed_dataset
from scripts.train_observed_geometry import train,load_geometry,batch_inputs,validate_endpoint_training
from test_observed_multitask import make_generic_fixture


def same(left,right):
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:same(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):same(a,b)
    elif isinstance(left,torch.Tensor):assert torch.equal(left,right)
    elif isinstance(left,np.ndarray):np.testing.assert_array_equal(left,right)
    else:assert left==right


def args(root):
    obs,sup,cache=make_generic_fixture(root/'data')
    return dict(observations=str(obs),supervision=str(sup),cache_dir=str(cache),output=str(root/'full'),steps=4,
        batch_size=2,candidates=2,horizon=12,width=16,depth=1,pooling='both',geometry_pooling='spatial',anchor_mode='soft',
        endpoint_mode='free_offset',sampling_mode='task_parent_language',metric_aggregation='task_parent',
        refinement_mode='none',refinement_sigma=None,refinement_prefix_fraction=None,refinement_bound=None,
        checkpoint_selection='reference_ADE',point_width=8,pixel_stride=2,endpoint_residual_bound=.05,grounding_weight=.02,
        grounding_target='event_supported',grounding_sigma=.025,event_scale=.2,lr=3e-4,seed=31,eval_every=2,
        threads=1,device='cpu',resume=False,stop_after=None)


def test_same_initial_model_forward_rng_and_sampling_no_label_forward(tmp_path):
    values=args(tmp_path);data=load_observed_dataset(values['observations'],values['supervision'],values['cache_dir'],12,'both')
    geometry=load_geometry(data,values['observations'],values['supervision'],2);ids=np.flatnonzero((data['splits']=='TRAIN')&data['path_mask'].any(1))
    seed_all(31);ordinary=ObservedGeometryRouteHead(16,12,2,16,1,8,endpoint_mode='free_offset')
    seed_all(31);targets,_=prepare_event_grounding_targets(data,geometry,ids)
    auxiliary=ObservedGeometryRouteHead(16,12,2,16,1,8,endpoint_mode='free_offset');same(ordinary.state_dict(),auxiliary.state_dict())
    batch=batch_inputs(data,geometry,ids,'cpu');a=ordinary(**batch);b=auxiliary(**batch)
    same(a,b)
    first=np.random.default_rng(100031);second=np.random.default_rng(100031)
    for _ in range(10):np.testing.assert_array_equal(draw_observation_batch(data,ids,first,2,'task_parent_language'),draw_observation_batch(data,ids,second,2,'task_parent_language'))
    loss=positive_endpoint_attention_loss(b[2]['attention'],batch['world_xyz'],batch['valid_mask'],torch.from_numpy(targets[ids]),torch.from_numpy(data['path_mask'][ids]),.025)
    loss.backward();assert auxiliary.geometry.task_query[-1].weight.grad.abs().sum()>0
    with pytest.raises(TypeError):auxiliary(**batch,grounding_targets=torch.from_numpy(targets[ids]))
    with pytest.raises(ValueError):validate_endpoint_training(dict(endpoint_mode='free_offset',grounding_weight=.02))
    validate_endpoint_training(dict(endpoint_mode='free_offset',grounding_weight=.02,grounding_target='event_supported'))


def test_actual_loop_event_aux_resume_bit_exact_and_zero_weight_historical_control(tmp_path,monkeypatch):
    # This unit fixture has no live simulator corpus. Real training still MUST
    # pass its unpatched model-use gate; only this isolated test substitutes it.
    monkeypatch.setattr('scripts.train_observed_geometry.check_multitask_model_gate',lambda *a,**kw:{'unit_test_only':True})
    values=args(tmp_path);train(argparse.Namespace(**values));expected=torch.load(tmp_path/'full/last.pt',weights_only=False)
    values.update(output=str(tmp_path/'resumed'),stop_after=2);train(argparse.Namespace(**values))
    values.update(resume=True,stop_after=None);train(argparse.Namespace(**values));actual=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key in ('model','optimizer','scheduler','sampler_state','rng','trajectory_exposures','step','best'):same(expected[key],actual[key])
    assert expected['config']['grounding_target_fingerprint']==actual['config']['grounding_target_fingerprint']
    metadata=json.loads((tmp_path/'full/grounding_target_selection.json').read_text());assert metadata['positive_train_observations']==3
    values.update(resume=False,grounding_weight=0.,output=str(tmp_path/'zero_aux'));train(argparse.Namespace(**values));zero=torch.load(tmp_path/'zero_aux/last.pt',weights_only=False)
    values.update(grounding_target='endpoint',output=str(tmp_path/'ordinary'));train(argparse.Namespace(**values));ordinary=torch.load(tmp_path/'ordinary/last.pt',weights_only=False)
    for key in ('model','optimizer','scheduler','sampler_state','rng','trajectory_exposures','step','best'):same(zero[key],ordinary[key])
