import json
from types import SimpleNamespace

import numpy as np
import pytest

torch=pytest.importorskip('torch')

from scripts import run_observed_two_row_composite as pipeline
from scripts import train_observed_two_row as ordinary
from test_two_row_observation_training import fixture
from test_observed_grounding_target_training import same


def test_scoped_adapter_restores_shared_functions_on_exception(monkeypatch):
    before=(ordinary.verify_export,ordinary.base.evaluate,ordinary.base.new_stream_audit)
    with pytest.raises(RuntimeError):
        with pipeline.training_adapter(ordinary,('obs','sup')):raise RuntimeError('failure')
    assert before==(ordinary.verify_export,ordinary.base.evaluate,ordinary.base.new_stream_audit)


def test_initial_model_guard_runs_before_training_exposure():
    original=ordinary.base.new_stream_audit
    class Model(torch.nn.Module):
        def __init__(self):super().__init__();self.x=torch.nn.Parameter(torch.zeros(1))
    model=Model();sampler=np.random.default_rng(100000)
    with pipeline.training_adapter(ordinary,('obs','sup')):
        with pytest.raises(ValueError,match='Initial model'):
            ordinary.base.new_stream_audit(model,sampler.bit_generator.state,torch.get_rng_state())
    assert ordinary.base.new_stream_audit is original


def test_shared_constant_loop_composite_adapter_continuous_resume_exact(tmp_path,monkeypatch):
    data,geometry,_,sources,labels=fixture(tmp_path,monkeypatch)
    original_verify=ordinary.verify_export
    monkeypatch.setattr(pipeline.exporter,'verify_export',original_verify)
    data={k:(np.concatenate([v,v]) if isinstance(v,np.ndarray) else v*2) for k,v in data.items()}
    train_ids=np.array(['t_target0','t_target1','t_target2'])
    data['scene_ids'][:3]=train_ids;data['parent_ids'][:3]='t';data['splits']=np.array(['TRAIN']*3+['DEV_MODEL']*3)
    data.update(tasks=['reach']*6,source_hashes={},cache_config={},skipped=[],unreferenced=[],evaluation_protocol='observation_eval_v2')
    train_labels=[dict(r,id=str(train_ids[i]),parent_id='t',split='TRAIN') for i,r in enumerate(labels)]
    sources[1].write_text(''.join(json.dumps(r)+'\n' for r in train_labels+labels))
    geometry.update(index=np.zeros(6,dtype=int),fingerprint='fixture',metadata={'source_hashes':{}})
    monkeypatch.setattr(ordinary.base,'load_observed_dataset',lambda *a,**kw:data)
    monkeypatch.setattr(ordinary.base,'load_geometry',lambda *a,**kw:geometry)
    monkeypatch.setattr(ordinary.base,'measure_latency',lambda *a,**kw:{'fixture':True})
    values=dict(pipeline.MODEL_OPTIONS,observations=str(sources[0]),supervision=str(sources[1]),cache_dir='fixture',
        output=str(tmp_path/'continuous'),steps=4,batch_size=2,candidates=2,horizon=2,width=16,depth=1,
        point_width=8,eval_every=2,geometry_pooling='spatial',metric_aggregation='instruction',
        refinement_sigma=None,refinement_prefix_fraction=None,refinement_bound=None,
        checkpoint_selection='tip_unique_valid',device='cpu',threads=1,resume=False,stop_after=None)
    # Tiny fixture changes architecture; production CLI cannot override INITIAL.
    with pipeline.training_adapter(ordinary,sources,initial={}):ordinary.base.train(SimpleNamespace(**values))
    expected=torch.load(tmp_path/'continuous/last.pt',weights_only=False)
    values.update(output=str(tmp_path/'resumed'),stop_after=2)
    with pipeline.training_adapter(ordinary,sources,initial={}):ordinary.base.train(SimpleNamespace(**values))
    values.update(resume=True,stop_after=None)
    with pipeline.training_adapter(ordinary,sources,initial={}):ordinary.base.train(SimpleNamespace(**values))
    actual=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key in ('model','optimizer','scheduler','rng','sampler_state','best','step','trajectory_exposures','sample_stream_audit'):
        same(expected[key],actual[key])
    # Original evaluator adapter + same unchanged constant loop must also agree.
    values.update(output=str(tmp_path/'ordinary'),resume=False)
    with ordinary.evaluation_adapter(sources):ordinary.base.train(SimpleNamespace(**values))
    old=torch.load(tmp_path/'ordinary/last.pt',weights_only=False)
    for key in ('model','optimizer','scheduler','rng','sampler_state','best','step','trajectory_exposures','sample_stream_audit'):
        same(expected[key],old[key])
