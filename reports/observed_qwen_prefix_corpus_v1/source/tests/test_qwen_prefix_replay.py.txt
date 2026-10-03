import copy
import json
from pathlib import Path

import pytest

from routeset.qwen_prefix_replay import (BUDGET, IDS, PARENTS, validate_policy, validate_rows)
from scripts.audit_two_row_qwen_prefix_replay import BudgetLedger, exact_prefix, checked_train_path

ROOT = Path(__file__).resolve().parents[1]


def policy():
    return json.loads((ROOT/'configs/two_row_qwen_prefix_replay_probe_v1.json').read_text())


def rows():
    return [dict(id=i, parent_id=p, split='TRAIN', image='/safe/'+p+'/front.png', instruction='Reach the red target.')
            for i,p in zip(IDS, [p for p in PARENTS for _ in range(3)])]


def test_fixed_policy_and_all_budget_identities():
    assert validate_policy(policy())['budget'] == BUDGET
    assert BUDGET['full_features'] == BUDGET['replay_features'] == 6+2+2
    assert BUDGET['head_calls'] == BUDGET['optimizer_steps_total'] == 2+2
    assert BUDGET['candidate_path_states'] == BUDGET['head_calls']*4


@pytest.mark.parametrize('key,value', [('logical_steps',3),('cut_layer',25),('resume',True),
    ('retry',True),('allowed_split','DEV_MODEL'),('feature_comparison','close'),('micro_batch',2),('head_lr',.001)])
def test_policy_change_rejected(key,value):
    data=policy();data[key]=value
    with pytest.raises(ValueError): validate_policy(data)


def test_exact_rows_and_no_answer_fields():
    assert validate_rows(rows())
    for mutate in ('split','labels','order','parent'):
        data=rows()
        if mutate=='split':data[0]['split']='DEV_MODEL'
        elif mutate=='labels':data[0]['target_xyz']=[1,2,3]
        elif mutate=='parent':data[0]['parent_id']=PARENTS[1]
        else:data.reverse()
        with pytest.raises(ValueError):validate_rows(data)


def test_json_prefix_never_parses_later_dev(tmp_path):
    path=tmp_path/'rows.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows())+'INVALID DEV MUST NOT BE PARSED\n')
    assert exact_prefix(path)==rows()


def test_failure_consumes_budget_and_is_sealed(tmp_path):
    ledger=BudgetLedger(tmp_path)
    def bad():raise RuntimeError('injected')
    with pytest.raises(RuntimeError):ledger.call('head','first',bad)
    for i in range(3):assert ledger.call('head',str(i),lambda:7)==7
    with pytest.raises(ValueError):ledger.call('head','no fifth',lambda:7)
    assert ledger.counts()['candidate_path_states']==16
    saved=json.loads((tmp_path/'call_ledger.json').read_text())
    assert saved[0]['state']=='failed' and len(saved)==4


def test_train_path_rejects_escape_before_hash(tmp_path):
    from scripts.audit_two_row_qwen_prefix_replay import sha
    parent=tmp_path/'parents'/'TRAIN'/PARENTS[0];parent.mkdir(parents=True)
    item=parent/'front';item.write_bytes(b'pixel')
    manifest=dict(sources={'old':{'source_dataset':str(tmp_path)}},source_files_sha256={str(item):sha(item)})
    assert checked_train_path(item,PARENTS[0],manifest,{})==item
    outside=tmp_path/'reserved';outside.write_bytes(b'never hash')
    with pytest.raises(ValueError):checked_train_path(outside,PARENTS[0],manifest,{})


def cache_fixture(tmp_path):
    import hashlib
    import numpy as np
    from scripts.audit_two_row_qwen_prefix_replay import sha,EXPORT_SHA,REVISION
    data=rows();cache=tmp_path/'qwen_cache';cache.mkdir()
    image=tmp_path/'front.png';image.write_bytes(b'fixture image')
    for row in data:row['image']=str(image)
    (tmp_path/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in data)+'DO NOT PARSE DEV\n')
    config=dict(revision=REVISION,processor=REVISION,model_trainable_parameter_count=0,
                manifest_sha256=sha(tmp_path/'observations.jsonl'))
    (cache/'cache_config.json').write_text(json.dumps(config))
    records=[];prior=[];artifacts={}
    for i,row in enumerate(data):
        name=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]+'.npz'
        np.savez(cache/name,mean_hidden=np.full(2048,i,np.float32),last_hidden=np.full(2048,i+1,np.float32),
            id=row['id'],parent_id=row['parent_id'],split='TRAIN',image_sha256=sha(image),input_tokens=80+i)
        h=sha(cache/name);artifacts[name]=h
        records.append(dict(id=row['id'],file=name,sha256=h,image_sha256=sha(image),input_tokens=80+i))
        prior.append(dict(id=row['id'],file=name,sha256=h,split='TRAIN',npz_bytes_equal=True,feature_arrays_equal=True))
    (cache/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records)+'FORBIDDEN DEV ROW\n')
    artifacts.update({name:sha(cache/name) for name in ('cache_config.json','samples.jsonl')})
    receipt=dict(protocol='two_row_composite108_old225_cache_reuse_v1',source_export_manifest_sha256=EXPORT_SHA,
                 old_reused_count=225,old225_by_id=prior,artifact_sha256=artifacts)
    path=cache/'composite_cache_receipt.json';path.write_text(json.dumps(receipt))
    return data,config,sha(path),records


def test_sealed_cache_only_six_npz_and_exact_feature(tmp_path,monkeypatch):
    import numpy as np
    from scripts.audit_two_row_qwen_prefix_replay import sealed_training_cache,compare_historical_feature
    data,config,receipt_hash,records=cache_fixture(tmp_path)
    original=np.load;opened=[]
    def guarded(path,*args,**kwargs):
        opened.append(Path(path).name)
        assert Path(path).name in {r['file'] for r in records}
        return original(path,*args,**kwargs)
    monkeypatch.setattr(np,'load',guarded)
    features,audit=sealed_training_cache(tmp_path,data,receipt_hash,config)
    assert len(opened)==6 and audit['opened_npz_count']==6
    saved=features[IDS[0]]
    assert compare_historical_feature(saved['feature'].copy(),80,saved)['exact']
    changed=saved['feature'].copy();changed[0,0]=np.nextafter(changed[0,0],np.float32(1))
    assert not compare_historical_feature(changed,80,saved)['exact']
    with pytest.raises(ValueError):compare_historical_feature(saved['feature'],81,saved)
    assert not audit['original_prompt_and_token_ids_available']


@pytest.mark.parametrize('tamper',['npz','receipt','instruction','config'])
def test_historical_cache_guard_rejects_changed_identity_bytes(tmp_path,tamper):
    from scripts.audit_two_row_qwen_prefix_replay import sealed_training_cache
    data,config,receipt_hash,records=cache_fixture(tmp_path)
    cache=tmp_path/'qwen_cache'
    if tamper=='npz':
        with (cache/records[0]['file']).open('ab') as stream:stream.write(b'tampered')
    elif tamper=='receipt':receipt_hash='0'*64
    elif tamper=='instruction':data[0]['instruction']='Different target'
    else:config=dict(config,revision='wrong')
    with pytest.raises((ValueError,FileNotFoundError)):
        sealed_training_cache(tmp_path,data,receipt_hash,config)


def tiny_model():
    torch=pytest.importorskip('torch')
    pytest.importorskip('transformers')
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    torch.set_num_threads(1);torch.manual_seed(19)
    config=Qwen3VLConfig(text_config=dict(vocab_size=64,hidden_size=16,intermediate_size=32,
        num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=1,head_dim=8,
        rope_scaling={'rope_type':'default','mrope_section':[2,1,1],'mrope_interleaved':True}),
        vision_config=dict(depth=2,hidden_size=16,intermediate_size=32,num_heads=2,
            out_hidden_size=16,patch_size=2,spatial_merge_size=2,temporal_patch_size=2,
            num_position_embeddings=16,deepstack_visual_indexes=[0,1]),
        image_token_id=63,video_token_id=61,vision_start_token_id=62)
    model=Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    inputs=dict(input_ids=torch.tensor([[1,62,63,63,63,63,2,3]]),attention_mask=torch.ones(1,8,dtype=torch.long),
                pixel_values=torch.randn(16,24),image_grid_thw=torch.tensor([[1,4,4]]))
    return torch,model,inputs


def test_real_small_qwen_image_deepstack_capture_serial_roundtrip(tmp_path):
    from routeset.qwen_prefix_replay import capture_prefix,replay_feature,tensor_difference
    torch,model,inputs=tiny_model()
    payload,official=capture_prefix(model,inputs,2)
    torch.save(payload,tmp_path/'prefix.pt')
    payload=torch.load(tmp_path/'prefix.pt',weights_only=False)
    with torch.no_grad():replayed=replay_feature(model,payload)
    assert tensor_difference(official,replayed)['exact']
    assert payload['kwargs']['past_key_values'] is None
    assert payload['pooling_mask'].shape==(1,8)
    assert not model.model.language_model.layers[2]._forward_pre_hooks
    with torch.inference_mode(),pytest.raises(ValueError):capture_prefix(model,inputs,2)


def test_real_small_qwen_shared_tail_gradients_optimizer_isolation_and_resume():
    from routeset.qwen_prefix_replay import (capture_prefix,replay_feature,official_feature,PairedTailUpdater,
        parameter_hashes,adapter_parameters,cpu_copy,differences)
    torch,model,inputs=tiny_model()
    from scripts.train_observed_lora import install_lora
    payload,official=capture_prefix(model,inputs,2)
    install_lora(model,rank=2,alpha=4.);model.eval()
    frozen_before=parameter_hashes(model,exclude_adapters=True)
    original=cpu_copy(adapter_parameters(model))
    head=torch.nn.Linear(32,4)
    updater=PairedTailUpdater(model,head,copy.deepcopy(head),.001,.001)
    def feature_functions():return dict(full=lambda:official_feature(model,inputs),replay=lambda:replay_feature(model,payload))
    def loss(h,f,name):
        prediction=h(f)
        # Exercise branch RNG restoration and the saved-pair restart, not only
        # deterministic operators that could accidentally hide a missing RNG.
        desired=torch.rand_like(prediction)
        return (prediction-desired).square().mean(),dict(prediction=prediction)
    first=updater.step(feature_functions(),loss)
    assert first['exact'],first['differences']
    assert all(not v['nonzero'] for n,v in first['branches']['full']['gradient_audit'].items() if n.endswith('lora_A'))
    saved=copy.deepcopy(updater.state_dict())
    second=updater.step(feature_functions(),loss)
    assert second['exact'],second['differences']
    continuous=updater.state_dict()
    # A fresh pair of heads/optimizers restores each branch's independent state.
    restored=PairedTailUpdater(model,copy.deepcopy(head),copy.deepcopy(head),.001,.001)
    restored.load_state_dict(saved)
    again=restored.step(feature_functions(),loss)
    assert again['exact'],again['differences']
    assert not differences(continuous,restored.state_dict())
    assert parameter_hashes(model,exclude_adapters=True)==frozen_before
    assert all(not torch.equal(original[n],p.detach().cpu()) for n,p in adapter_parameters(model).items())
    assert all(v['nonzero'] for n,v in second['branches']['full']['gradient_audit'].items() if '.lora_' in n)
    assert not torch.equal(official,replay_feature(model,payload).detach().cpu())
    broken=copy.deepcopy(saved);del broken['optimizers']['replay']
    with pytest.raises(ValueError):restored.load_state_dict(broken)
    broken=copy.deepcopy(saved);del broken['torch_rng']
    with pytest.raises(ValueError):restored.load_state_dict(broken)


def test_state_comparison_missing_keys_and_scalar_hash():
    torch=pytest.importorskip('torch')
    from routeset.qwen_prefix_replay import differences,tensor_hash
    assert differences({'step':torch.tensor(1)}, {})
    assert len(tensor_hash(torch.tensor(3.)))==64
    assert differences(torch.tensor(1.),torch.tensor(2.))
