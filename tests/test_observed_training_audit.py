import argparse
import copy
import json

import numpy as np
import pytest
import torch

from routeset.observed_training_audit import new_stream_audit, append_indices, restore_stream_audit
from scripts.train_observed_geometry import train
from test_observed_grounding_target_training import args, same


def test_hash_chain_persistence_order_counts_and_no_randomness():
    torch.manual_seed(2);model=torch.nn.Linear(2,1);sampler=np.random.default_rng(3)
    before=torch.get_rng_state().clone();initial=new_stream_audit(model,sampler.bit_generator.state,before)
    state=append_indices(initial,np.array([3,1,3]));saved=json.loads(json.dumps(state))
    restored=restore_stream_audit(True,{'sample_stream_audit':True},saved,initial,1,3)
    assert append_indices(state,np.array([0,2,1]))==append_indices(restored,np.array([0,2,1]))
    assert state['index_chain_sha256']!=append_indices(initial,np.array([1,3,3]))['index_chain_sha256']
    assert torch.equal(before,torch.get_rng_state())
    assert restore_stream_audit(False,{},None,None,12,32) is None
    for enabled,config,saved,step in [(True,{},state,1),(False,{'sample_stream_audit':True},state,1),(True,{'sample_stream_audit':True},None,1),(True,{'sample_stream_audit':True},state,2)]:
        with pytest.raises(ValueError):restore_stream_audit(enabled,config,saved,initial,step,3)
    malformed=dict(state,index_chain_sha256='00')
    with pytest.raises(ValueError):restore_stream_audit(True,{'sample_stream_audit':True},malformed,initial,1,3)
    with pytest.raises(ValueError):append_indices(initial,np.array([1.5]))


def test_actual_loop_audit_on_off_bit_equal_and_resume_chain(tmp_path,monkeypatch):
    monkeypatch.setattr('scripts.train_observed_geometry.check_multitask_model_gate',lambda *a,**kw:{'unit_test_only':True})
    values=args(tmp_path);values.update(sample_stream_audit=False)
    train(argparse.Namespace(**values));off=torch.load(tmp_path/'full/last.pt',weights_only=False)
    values.update(output=str(tmp_path/'audited'),sample_stream_audit=True);train(argparse.Namespace(**values))
    on=torch.load(tmp_path/'audited/last.pt',weights_only=False)
    for key in ('model','optimizer','scheduler','sampler_state','rng','trajectory_exposures','step','best'):same(off[key],on[key])
    values.update(output=str(tmp_path/'resumed'),stop_after=2);train(argparse.Namespace(**values))
    values.update(resume=True,stop_after=None);train(argparse.Namespace(**values))
    resumed=torch.load(tmp_path/'resumed/last.pt',weights_only=False)
    for key in ('model','optimizer','scheduler','sampler_state','rng','trajectory_exposures','step','best','sample_stream_audit'):same(on[key],resumed[key])
    assert on['sample_stream_audit']['observation_draws']==8
    summary=json.loads((tmp_path/'audited/summary.json').read_text())
    assert summary['sample_stream_audit']==on['sample_stream_audit']
    # Both ordinary and auxiliary have equal actual sampling despite different gradients.
    values.update(output=str(tmp_path/'ordinary'),resume=False,grounding_target='endpoint',grounding_weight=0.)
    train(argparse.Namespace(**values));ordinary=torch.load(tmp_path/'ordinary/last.pt',weights_only=False)
    assert ordinary['sample_stream_audit']==on['sample_stream_audit']
