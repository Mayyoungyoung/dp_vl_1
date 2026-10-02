"""Actual shared-loop CPU recovery with a tiny synthetic causal model.

No real-Qwen or RLBench result is inferred from these unit fixtures.
"""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from routeset.vlm_sft_data import fixed_development_plan,load_sft_data
from scripts.train_observed_lora import LoRALinear
from scripts.train_vlm_route_sft import digest_json,run_training
from test_vlm_sft_data import FakeProcessor,fixture_resample,make_fixture


class TorchProcessor(FakeProcessor):
    def __call__(self,*args,**kwargs):
        return {key:torch.from_numpy(value) for key,value in super().__call__(*args,**kwargs).items()}


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding=nn.Embedding(256,8).requires_grad_(False)
        self.first=LoRALinear(nn.Linear(8,8),rank=2,alpha=4.)
        self.second=LoRALinear(nn.Linear(8,8),rank=2,alpha=4.)

    def forward(self,input_ids,attention_mask,use_cache,return_dict):
        embedded=self.embedding(input_ids)
        positions=torch.arange(1,embedded.shape[1]+1,dtype=embedded.dtype)[None,:,None]
        prefix=embedded.cumsum(1)/positions
        hidden=torch.tanh(self.second(torch.tanh(self.first(prefix))))
        return SimpleNamespace(last_hidden_state=hidden)


class TinyCausal(nn.Module):
    def __init__(self):
        super().__init__();self.model=TinyDecoder();self.lm_head=nn.Linear(8,256,bias=False).requires_grad_(False)


def tensor_tree_equal(left,right):
    if isinstance(left,torch.Tensor):
        assert torch.equal(left,right)
    elif isinstance(left,np.ndarray):
        np.testing.assert_array_equal(left,right)
    elif isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:tensor_tree_equal(left[key],right[key])
    elif isinstance(left,(tuple,list)):
        assert len(left)==len(right)
        for a,b in zip(left,right):tensor_tree_equal(a,b)
    else:assert left==right


def test_actual_causal_sft_loop_four_steps_equals_two_then_resume_and_refuses_changes(tmp_path):
    torch.set_num_threads(1)
    obs,sup=make_fixture(tmp_path/'data');data=load_sft_data(obs,sup,4,fixture_resample)
    config=dict(steps=4,seed=13,lr=.001,weight_decay=0.,horizon=4,chunk_size=64,gradient_clip=1.,
                eval_every=4,checkpoint_every=2,log_every=4,dev_plan_seed=313,
                data_fingerprint=data['fingerprint'])
    config['dev_plan_sha256']=digest_json(fixed_development_plan(data['samples'],313))
    def model():torch.manual_seed(13);return TinyCausal()
    full=run_training(model(),TorchProcessor(),data,config,tmp_path/'full',device='cpu')
    half=run_training(model(),TorchProcessor(),data,config,tmp_path/'split',device='cpu',stop_after=2)
    assert half['status']=='stopped_early' and half['exposure']['TRAIN']['candidate_slots']==5
    for key,value in [('lr',.002),('data_fingerprint','bad'),('eval_every',2)]:
        changed=copy.deepcopy(config);changed[key]=value
        with pytest.raises(ValueError):
            run_training(model(),TorchProcessor(),data,changed,tmp_path/'split',device='cpu',resume=True)
    resumed=run_training(model(),TorchProcessor(),data,config,tmp_path/'split',device='cpu',resume=True)
    assert full['status']==resumed['status']=='completed'
    first=torch.load(tmp_path/'full/last.pt',weights_only=False)
    second=torch.load(tmp_path/'split/last.pt',weights_only=False)
    for key in ('adapters','optimizer','scheduler','rng','sampler_state','step','history','best_step','best_nll','exposure','gradient_audit'):
        tensor_tree_equal(first[key],second[key])
    assert first['exposure']['TRAIN']['candidate_slots']==10
    events=[json.loads(line) for line in (tmp_path/'split/requests.jsonl').read_text().splitlines()]
    assert [r['k'] for r in events if r['kind']=='train_request']==[1,4,1,4]
    assert set(first['adapters'])==set(second['adapters'])
    assert all('lora_' in name for name in first['adapters'])
    with pytest.raises(ValueError,match='planned limit'):
        run_training(model(),TorchProcessor(),data,config,tmp_path/'split',device='cpu',resume=True)
