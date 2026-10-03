import json
from pathlib import Path
import numpy as np
import pytest
torch=pytest.importorskip('torch')

from scripts import train_observed_two_row as driver
from scripts.collect_observed_two_row_pilot import geometry


def test_scoped_adapter_restores_original_default_even_on_failure():
    original=driver.base.evaluate
    with pytest.raises(RuntimeError):
        with driver.evaluation_adapter(('obs','sup')):
            assert driver.base.evaluate is not original
            raise RuntimeError('test interruption')
    assert driver.base.evaluate is original


def test_resume_rejects_selection_data_source_or_quality_change():
    keys=('two_row_driver_protocol','two_row_driver_sha256','two_row_export_sha256','two_row_quality_audit_sha256',
          'two_row_selection_sha256','two_row_tip_protocol')
    config={key:'frozen' for key in keys};driver.validate_adapter_resume(config,dict(config))
    for key in keys:
        with pytest.raises(ValueError,match='resume adapter mismatch'):driver.validate_adapter_resume(dict(config,**{key:'changed'}),config)


def fixture(tmp_path,monkeypatch):
    cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/observed_two_row_pilot_v4.json').read_text())
    (tmp_path/'config.json').write_text(json.dumps(cfg));centers,halves=geometry(cfg)
    np.savez(tmp_path/'verification.npz',obstacle_centers=centers,obstacle_halfsizes=halves)
    ids=np.array(['p_target0','p_target1','p_target2']);current=np.tile(np.r_[cfg['entry_xyz'],0.,0.,0.,1.,1.],(3,1)).astype(np.float32)
    paths=np.stack([np.stack([current[0,:3],goal]) for goal in cfg['goal_xyz']]).astype(np.float32)
    labels=[]
    for index,ident in enumerate(ids):
        labels.append(dict(id=ident,parent_id='p',split='DEV_MODEL',verification_only=str(tmp_path/'verification.npz'),
            route_config=str(tmp_path/'config.json'),route_config_sha256=driver.sha256(tmp_path/'config.json'),
            semantic_targets=dict(centers=cfg['goal_xyz'],target_index=index,tolerance=.03),route_types=[None]))
    sup=tmp_path/'supervision.jsonl';sup.write_text(''.join(json.dumps(r)+'\n' for r in labels))
    files={str(tmp_path/name):driver.sha256(tmp_path/name) for name in ('config.json','verification.npz')}
    monkeypatch.setattr(driver,'verify_export',lambda path:({'source_files_sha256':files},{}))
    data=dict(features=np.arange(3,dtype=np.float32)[:,None],current=current,scene_ids=ids,parent_ids=np.array(['p']*3),
        splits=np.array(['DEV_MODEL']*3),image_hashes=np.array(['same']*3),semantic_targets=[r['semantic_targets'] for r in labels],
        paths=paths[:,None],events=np.ones((3,1,2),np.float32),path_mask=np.array([[True],[False],[True]]))
    geometry_pack=dict(index=np.zeros(3,dtype=int),points={
        'world_xyz':np.zeros((1,2,3),np.float32),'rgb':np.zeros((1,2,3),np.float32),'uv':np.zeros((1,2,2),np.float32),
        'depth':np.ones((1,2),np.float32),'valid_mask':np.ones((1,2),bool)})
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.calls=0
        def forward(self,**inputs):
            assert set(inputs)=={'features','current','world_xyz','rgb','uv','depth','valid_mask'}
            self.calls+=1;index=inputs['features'][:,0].long()
            out=torch.as_tensor(paths)[index,None].expand(-1,4,-1,-1)
            return out,torch.ones(out.shape[:3]),{'anchor_xyz':out[:,0,-1]}
    return data,geometry_pack,Model(),(tmp_path/'observations.jsonl',sup),labels


def test_all_slots_no_ref_and_same_parent_language_control_reuse(tmp_path,monkeypatch):
    data,geometry_pack,model,sources,_=fixture(tmp_path,monkeypatch)
    metrics=driver.evaluate(model,data,geometry_pack,np.arange(3),'cpu',evaluation_sources=sources,selection_metric='tip_unique_valid')
    assert model.calls==1 and metrics['examples']==3 and metrics['reference_evaluation_examples']==2
    assert metrics['semantic_evaluation_examples']==3 and metrics['candidates']==4
    assert metrics['generation_budget']['evaluated_complete_path_states']==12
    assert metrics['paired_language_control']['additional_forward_requests']==0
    assert metrics['paired_language_control']['same_image_other_target_language_new_goal_accuracy']==1
    assert metrics['selection_score']==metrics['UniqueClassifiedTipValidAtK']+.05*metrics['TipValidAtK']


def test_language_reuse_requires_identical_observation_geometry(tmp_path,monkeypatch):
    data,geometry_pack,_,_,_=fixture(tmp_path,monkeypatch);geometry_pack['index'][1]=1
    prediction=np.repeat(data['paths'],4,axis=1);events=np.repeat(data['events'],4,axis=1)
    with pytest.raises(ValueError,match='identical current observation'):
        driver.reused_language_control(prediction,events,data,geometry_pack,np.arange(3))


def test_privileged_geometry_only_opens_after_all_model_calls(tmp_path,monkeypatch):
    data,geometry_pack,model,sources,_=fixture(tmp_path,monkeypatch)
    load=np.load;reads=[]
    def audited(*args,**kwargs):
        assert model.calls==3 # batch_size1 must finish every condition first.
        reads.append(str(args[0]));return load(*args,**kwargs)
    monkeypatch.setattr(np,'load',audited)
    driver.evaluate(model,data,geometry_pack,np.arange(3),'cpu',batch_size=1,evaluation_sources=sources)
    assert len(reads)==3


def test_actual_two_row_selection_loop_complete_and_resume_are_bit_equal(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from test_observed_grounding_target_training import same
    data,geometry_pack,_,sources,labels=fixture(tmp_path,monkeypatch)
    train_ids=np.array(['t_target0','t_target1','t_target2'])
    data={key:(np.concatenate([value,value]) if isinstance(value,np.ndarray) else value*2)
          for key,value in data.items()}
    data['scene_ids'][:3]=train_ids;data['parent_ids'][:3]='t';data['splits']=np.array(['TRAIN']*3+['DEV_MODEL']*3)
    data.update(tasks=['reach']*6,source_hashes={},cache_config={},skipped=[],unreferenced=[],evaluation_protocol='observation_eval_v2')
    train_labels=[dict(row,id=str(train_ids[i]),parent_id='t',split='TRAIN') for i,row in enumerate(labels)]
    sources[1].write_text(''.join(json.dumps(r)+'\n' for r in train_labels+labels))
    geometry_pack.update(index=np.zeros(6,dtype=int),fingerprint='fixture',metadata={'source_hashes':{}})
    monkeypatch.setattr(driver.base,'load_observed_dataset',lambda *a,**kw:data)
    monkeypatch.setattr(driver.base,'load_geometry',lambda *a,**kw:geometry_pack)
    monkeypatch.setattr(driver.base,'measure_latency',lambda *a,**kw:{'unit_test_skipped_io_latency':True})
    values=dict(observations=str(sources[0]),supervision=str(sources[1]),cache_dir='fixture',output=str(tmp_path/'full'),
        steps=4,batch_size=2,candidates=2,horizon=2,width=16,depth=1,pooling='both',geometry_pooling='spatial',
        anchor_mode='straight_through_peak',endpoint_mode='surface_anchor',sampling_mode='uniform',metric_aggregation='instruction',
        refinement_mode='none',refinement_sigma=None,refinement_prefix_fraction=None,refinement_bound=None,
        checkpoint_selection='tip_unique_valid',point_width=8,pixel_stride=2,endpoint_residual_bound=.05,
        grounding_weight=.02,grounding_sigma=.025,grounding_target='endpoint',event_scale=.2,lr=3e-4,
        seed=0,eval_every=2,threads=1,device='cpu',resume=False,stop_after=None,sample_stream_audit=True)
    with driver.evaluation_adapter(sources):driver.base.train(SimpleNamespace(**values))
    expected=torch.load(tmp_path/'full/last.pt',weights_only=False)
    values.update(output=str(tmp_path/'resumed'),stop_after=2)
    with driver.evaluation_adapter(sources):driver.base.train(SimpleNamespace(**values))
    values.update(resume=True,stop_after=None)
    with driver.evaluation_adapter(sources):driver.base.train(SimpleNamespace(**values))
    actual=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key in ('model','optimizer','scheduler','rng','sampler_state','best','step','trajectory_exposures','sample_stream_audit'):
        same(expected[key],actual[key])
    result=json.loads((tmp_path/'resumed/summary.json').read_text())
    assert result['last_metrics']['tip_evaluation_protocol']==driver.PROTOCOL
    assert result['last_metrics']['generation_budget']['language_control_additional_requests']==0
    index=driver.prediction_artifact_index(tmp_path/'resumed')
    assert len(index)==12
    for key,record in index.items():
        assert record['sha256']==driver.sha256(tmp_path/'resumed'/key)
    (tmp_path/'resumed/train/paired_language_predictions.npz').unlink()
    with pytest.raises(FileNotFoundError):driver.prediction_artifact_index(tmp_path/'resumed')
