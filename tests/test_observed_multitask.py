"""Synthetic CPU fixtures, not real-Qwen or RLBench experimental evidence."""
import argparse
import hashlib
import json

import numpy as np
from PIL import Image
import pytest
import torch

from routeset.common import sha256
from routeset.observed_geometry import ObservedGeometryRouteHead
from routeset.observed_route_head import OBSERVATION_KEYS,QWEN_REVISION,load_observed_dataset
from scripts.train_observed_geometry import (batch_inputs,evaluate,load_geometry,train,
    validate_endpoint_resume,validate_endpoint_training)
from scripts.train_observed_routes import paired_language_indices


def make_generic_fixture(root):
    cache=root/'cache';cache.mkdir(parents=True)
    observations=[];labels=[]
    specs=[('TRAIN','a',2,True),('TRAIN','b',1,True),('DEV_MODEL','a',3,True),
           ('DEV_MODEL','b',1,False),('DEV_MODEL','b',2,True)]
    for number,(split,task,languages,has_reference) in enumerate(specs):
        parent='parent%d'%number;folder=root/parent;folder.mkdir()
        image=folder/'rgb.png';Image.fromarray(np.full((6,6,3),30+number,np.uint8)).save(image)
        current=np.array([.1,.2,.3,0,0,0,1],np.float32)
        np.savez(folder/'observation.npz',gripper_pose=current,gripper_open=np.array(1.),
            depth=np.full((6,6),.5,np.float32),camera_intrinsics=np.diag([5.,5.,1.]),camera_extrinsics=np.eye(4),
            forbidden_task_low_dim_state=np.full(3,999.))
        xyz=np.linspace(current[:3],current[:3]+[.1,0,.7],9)
        pose=np.c_[xyz,np.tile(current[3:],(9,1))];opened=np.array([1,1,1,0,0,0,1,1,1])
        np.savez(folder/'route.npz',gripper_pose=pose,gripper_open=opened)
        for language in range(languages):
            row=dict(id=parent+'_lang%d'%language,parent_id=parent,split=split,
                image=parent+'/rgb.png',instruction=('lift with this wording '*(language+1)).strip())
            observations.append(row);labels.append(dict(id=row['id'],parent_id=parent,split=split,task=task,
                observation=parent+'/observation.npz',routes=[parent+'/route.npz'] if has_reference else [],semantic_targets=None))
            key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
            # Deliberately synthetic hidden states for unit tests only.
            feature=np.arange(8,dtype=np.float32)*.01+language*.1
            np.savez(cache/(key+'.npz'),mean_hidden=feature,last_hidden=feature+.1,id=row['id'],parent_id=parent,
                split=split,image_sha256=sha256(image),input_tokens=12+language*5)
    obs=root/'observations.jsonl';sup=root/'supervision.jsonl'
    for path,rows in ((obs,observations),(sup,labels)):path.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    (cache/'cache_config.json').write_text(json.dumps(dict(model='Qwen/Qwen3-VL-2B-Instruct',revision=QWEN_REVISION,
        model_trainable_parameter_count=0,manifest_sha256=sha256(obs),input_contract=sorted(OBSERVATION_KEYS))))
    return obs,sup,cache


def inputs():
    generator=torch.Generator().manual_seed(98)
    return dict(features=torch.randn(2,16,generator=generator),current=torch.randn(2,8,generator=generator),
        world_xyz=torch.randn(2,7,3,generator=generator),rgb=torch.rand(2,7,3,generator=generator),
        uv=torch.rand(2,7,2,generator=generator),depth=torch.ones(2,7),valid_mask=torch.ones(2,7,dtype=torch.bool))


def test_default_surface_forward_and_parameter_initialization_are_unchanged():
    torch.set_num_threads(1)
    torch.manual_seed(11);default=ObservedGeometryRouteHead(16,horizon=8,width=16,point_width=8)
    torch.manual_seed(11);explicit=ObservedGeometryRouteHead(16,horizon=8,width=16,point_width=8,endpoint_mode='surface_anchor')
    torch.manual_seed(11);free=ObservedGeometryRouteHead(16,horizon=8,width=16,point_width=8,endpoint_mode='free_offset')
    for name,value in default.state_dict().items():
        assert torch.equal(value,explicit.state_dict()[name]) and torch.equal(value,free.state_dict()[name])
    batch=inputs();first=default(**batch);second=explicit(**batch)
    assert all(torch.equal(first[i],second[i]) for i in (0,1))
    assert all(torch.equal(first[2][key],second[2][key]) for key in first[2])
    # Verify the historical surface formula independently of the new branch.
    context=default.head.feature_encoder(batch['features'])+default.head.state_encoder(batch['current'])+first[2]['context']
    tokens=context[:,None]+default.head.queries[None]
    for block in default.head.blocks:tokens=block(tokens,context)
    prediction=default.head.output(tokens).reshape(2,4,7,4)
    fraction=torch.linspace(0,1,8)[1:]
    line=batch['current'][:,None,None,:3]+fraction[None,None,:,None]*(first[2]['anchor_xyz']-batch['current'][:,:3])[:,None,None]
    expected=torch.cat([batch['current'][:,None,None,:3].expand(-1,4,1,-1),
        line[:,:,:-1]+prediction[:,:,:-1,:3],first[2]['anchor_xyz'][:,None,None]+.05*prediction[:,:,-1:,:3].tanh()],2)
    assert torch.equal(first[0],expected)


def test_free_airborne_endpoint_event_and_mask_contract_and_trainable_geometry():
    model=ObservedGeometryRouteHead(16,horizon=8,width=16,point_width=8,endpoint_mode='free_offset')
    batch=inputs();batch['current'][:,7]=1
    xyz,opened,details=model(**batch)
    (xyz.square().mean()+opened.square().mean()).backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
    assert model.geometry.point_encoder[0].weight.grad.abs().sum()>0
    assert model.geometry.task_query[-1].weight.grad.abs().sum()>0
    assert torch.equal(xyz[:,:,0],batch['current'][:,None,:3].expand(-1,4,-1))
    assert torch.equal(opened[:,:,0],torch.ones(2,4))
    with torch.no_grad():
        model.head.output[-1].weight.zero_();model.head.output[-1].bias.zero_()
        model.head.output[-1].bias.reshape(7,4)[-1,2]=3.
    xyz,_,_=model(**batch)
    torch.testing.assert_close(xyz[:,:,-1,2],batch['current'][:,2,None].expand(-1,4)+3.)
    assert (xyz[:,:,-1]-details['anchor_xyz'][:,None]).abs().max()>.05
    with pytest.raises(TypeError):model(**batch,semantic_targets=None)
    with pytest.raises(TypeError):model(**batch,task_id=torch.zeros(2))


def test_variable_language_null_semantics_no_reference_preserved(tmp_path):
    obs,sup,cache=make_generic_fixture(tmp_path);data=load_observed_dataset(obs,sup,cache,horizon=12)
    geometry=load_geometry(data,obs,sup,pixel_stride=2)
    assert data['features'].shape==(9,16) and geometry['metadata']['unique_current_observations']==5
    assert len(data['unreferenced'])==1 and data['semantic_targets']==[None]*9
    assert data['events'][0,0].tolist()==[1.,1.,1.,1.,0.,0.,0.,0.,1.,1.,1.,1.]
    ids=np.flatnonzero(data['splits']=='DEV_MODEL')
    assert (paired_language_indices(data,ids)==-1).all()
    model=ObservedGeometryRouteHead(16,horizon=12,width=16,point_width=8,endpoint_mode='free_offset')
    result=evaluate(model,data,geometry,ids,'cpu',metric_aggregation='task_parent')
    assert result['examples']==6 and result['reference_evaluation_examples']==5 and result['semantic_evaluation_examples']==0
    assert result['semantic_goal_accuracy'] is None and result['ValidAtK'] is None
    assert result['paired_language_control'] is None and result['learned_surface_anchor_reference_endpoint_error_m'] is None
    assert result['generation_budget']['total_complete_path_states']==4
    assert result['evaluated_tasks']==2 and result['reference_evaluable_parents']==2
    assert set(batch_inputs(data,geometry,ids[:1],'cpu'))=={'features','current','world_xyz','rgb','uv','depth','valid_mask'}


def test_free_endpoint_rejects_reach_supervision_and_resume_representation_change():
    validate_endpoint_training({});validate_endpoint_resume(dict(endpoint_mode='surface_anchor'),{})
    validate_endpoint_training(dict(endpoint_mode='free_offset',grounding_weight=0.))
    for key,value in [('grounding_weight',.02),('checkpoint_selection','tip_unique_valid'),('refinement_mode','local')]:
        with pytest.raises(ValueError):validate_endpoint_training(dict(endpoint_mode='free_offset',**{key:value}))
    with pytest.raises(ValueError,match='endpoint_mode'):validate_endpoint_resume(dict(endpoint_mode='free_offset'),{})
    with pytest.raises(ValueError,match='endpoint_mode'):ObservedGeometryRouteHead(16,endpoint_mode='bad')


def test_free_baseline_actual_training_resume_bit_exact(tmp_path):
    obs,sup,cache=make_generic_fixture(tmp_path/'data')
    values=dict(observations=str(obs),supervision=str(sup),cache_dir=str(cache),output=str(tmp_path/'full'),steps=4,
        batch_size=2,candidates=2,horizon=12,width=16,depth=1,pooling='both',geometry_pooling='spatial',anchor_mode='soft',
        endpoint_mode='free_offset',sampling_mode='task_parent_language',metric_aggregation='task_parent',
        refinement_mode='none',refinement_sigma=None,refinement_prefix_fraction=None,refinement_bound=None,
        checkpoint_selection='reference_ADE',point_width=8,pixel_stride=2,endpoint_residual_bound=.05,grounding_weight=0.,
        grounding_sigma=.025,event_scale=.2,lr=3e-4,seed=31,eval_every=2,threads=1,device='cpu',resume=False,stop_after=None)
    train(argparse.Namespace(**values))
    expected=torch.load(tmp_path/'full/last.pt',weights_only=False)
    values.update(output=str(tmp_path/'resumed'),stop_after=2);train(argparse.Namespace(**values))
    values.update(resume=True,stop_after=None);train(argparse.Namespace(**values))
    actual=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
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
    for key in ('model','optimizer','scheduler','sampler_state','rng','trajectory_exposures','step','best'):
        same(expected[key],actual[key])
    summary=json.loads((tmp_path/'resumed/summary.json').read_text())
    assert summary['last_step']==4 and summary['checkpoint_selection_protocol']=='task_parent_reference_ADE_v1'
    assert summary['last_metrics']['semantic_goal_accuracy'] is None
    assert (tmp_path/'resumed/last_dev_model/predictions.npz').exists()
