import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import run_observed_two_row_composite as pipeline

ROOT=Path(__file__).resolve().parents[1]


def policy():return json.loads((ROOT/'configs/observed_two_row_composite108_training_v1.json').read_text())


def test_fixed_policy_matches_original_model_and_constant_budget():
    p=policy();pipeline.validate_policy(p)
    original=json.loads((ROOT/'configs/observed_two_row_prefix76_selection_v1.json').read_text())['ordinary_training']
    assert all(original[k]==v for k,v in pipeline.MODEL_OPTIONS.items() if k!='steps')
    assert p['steps']*p['batch_size']*p['candidates']==1536000 and p['steps']//p['eval_every']==48


@pytest.mark.parametrize('field,value',[('steps',15000),('lr',.0001),('new_dev_raw_allowed',True),
    ('old_cache_inputs',224),('automatic_stage_chaining',True),('initial_state',{})])
def test_policy_rejects_budget_encoder_or_split_drift(field,value):
    p=policy();p[field]=value
    with pytest.raises(ValueError):pipeline.validate_policy(p)


def test_selected_train_keeps_empty_positive_inputs_but_rejects_new_dev_as_train(tmp_path):
    row=dict(id='two_row_reach_400000_target0',parent_id='two_row_reach_400000',split='TRAIN',image='x',instruction='touch')
    label=dict(row,routes=[])
    (tmp_path/'observations.jsonl').write_text(json.dumps(row)+'\n')
    (tmp_path/'supervision.jsonl').write_text(json.dumps(label)+'\n')
    _,labels=pipeline.selected_train(tmp_path);assert labels[0]['routes']==[]
    row.update(id='two_row_reach_400256_target0',parent_id='two_row_reach_400256')
    (tmp_path/'observations.jsonl').write_text(json.dumps(row)+'\n')
    (tmp_path/'supervision.jsonl').write_text(json.dumps(dict(row,routes=[]))+'\n')
    with pytest.raises(ValueError,match='TRAIN identities'):pipeline.selected_train(tmp_path)


def test_resume_rejects_changed_source_data_policy_cache_and_lr():
    cfg=dict(pipeline.MODEL_OPTIONS,composite_protocol=pipeline.PROTOCOL,composite_source_sha256={'base':'old'},
             composite_export_sha256='export',composite_quality_sha256='quality',composite_cache_sha256='cache',resume=False)
    pipeline.validate_resume(dict(cfg,resume=True),cfg)
    for key in cfg:
        if key=='resume':continue
        with pytest.raises(ValueError):pipeline.validate_resume(dict(cfg,**{key:'changed'}),cfg)


def test_actual_budget_requires_all48_selections_and_initial_state():
    audit=dict(pipeline.INITIAL,batches=12000,observation_draws=384000,index_chain_sha256='a'*64)
    ckpt=dict(step=12000,trajectory_exposures=1536000,history=[dict(step=i) for i in range(250,12001,250)],sample_stream_audit=audit)
    summary=dict(last_step=12000,trajectory_exposures=1536000,sample_stream_audit=audit)
    assert pipeline.validate_budget(summary,ckpt)['same_sample_stream_as_old'] is False
    ckpt['history'].pop()
    with pytest.raises(ValueError,match='budget differs'):pipeline.validate_budget(summary,ckpt)


def make_cache(root,manifest,rows):
    root.mkdir();records=[]
    for i,row in enumerate(rows):
        filename=pipeline.cache_key(row);path=root/filename
        np.savez_compressed(path,mean_hidden=np.full(2048,i,np.float32),last_hidden=np.full(2048,-i,np.float32),
            id=row['id'],parent_id=row['parent_id'],split=row['split'],image_sha256=pipeline.digest(row['image']),input_tokens=123)
        records.append(dict(id=row['id'],file=filename,sha256=pipeline.digest(path),input_tokens=123,
                            image_sha256=pipeline.digest(row['image']),single_request_seconds=.2))
    cfg=dict(model='Qwen/Qwen3-VL-2B-Instruct',revision=pipeline.qwen.REVISION,processor=pipeline.qwen.REVISION,
        manifest_sha256=pipeline.digest(manifest),max_pixels=262144,model_trainable_parameter_count=0,
        transformers='4.57.1',torch='fixture',dtype='torch.bfloat16',model_parameter_count=2127532032,
        input_contract=sorted(pipeline.qwen.ALLOWED),cache_scope='frozen only')
    pipeline.write(root/'cache_config.json',cfg)
    (root/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    pipeline.write(root/'status.json',dict(samples=len(rows),status='frozen_rgb_language_feature_extraction_complete',total_seconds=.5))
    return cfg,records


@pytest.fixture
def caches(tmp_path,monkeypatch):
    data=tmp_path/'data';old=tmp_path/'old';data.mkdir();old.mkdir();image=tmp_path/'rgb.bin';image.write_bytes(b'actual-image-bytes')
    rows=[]
    for parent in list(range(283200,283220))+list(range(283221,283276)):
        for target in range(3):
            rows.append(dict(id=f'two_row_reach_{parent}_target{target}',parent_id=f'two_row_reach_{parent}',
                split='TRAIN' if parent<283264 else 'DEV_MODEL',image=str(image),instruction=f'touch target{target}'))
    assert len(rows)==225
    added=[dict(id=f'two_row_reach_400000_target{t}',parent_id='two_row_reach_400000',split='TRAIN',image=str(image),instruction=f'touch new{t}') for t in range(3)]
    for path,records in [(old/'observations.jsonl',rows),(data/'observations.jsonl',rows+added),(tmp_path/'new.jsonl',added)]:
        path.write_text(''.join(json.dumps(r)+'\n' for r in records))
    pipeline.write(data/'export_manifest.json',{'test':'export'})
    m=dict(historical_export=str(old));monkeypatch.setattr(pipeline.exporter,'verify_export',lambda path:(m,{}))
    make_cache(old/'qwen_cache',old/'observations.jsonl',rows)
    make_cache(tmp_path/'new_cache',tmp_path/'new.jsonl',added)
    monkeypatch.setattr(pipeline,'OLD_CACHE_HASHES',{name.replace('.','_'):pipeline.digest(old/'qwen_cache'/name)
        for name in ('cache_config.json','samples.jsonl','status.json')})
    monkeypatch.setattr(pipeline,'EXTRACTOR_SHA',pipeline.digest(pipeline.qwen.__file__))
    return data,old,tmp_path/'new_cache',tmp_path/'new.jsonl',rows,added


def test_old225_feature_npz_bytes_and_dev_are_identical_after_merge(caches):
    data,old,new,manifest,rows,added=caches
    r=pipeline.merge_cache(data,old/'qwen_cache',new,manifest,data/'qwen_cache')
    assert r['new_encoding_count']==3 and r['old_reused_count']==225 and r['reused_dev_count']==36
    for row in rows:
        assert (data/'qwen_cache'/pipeline.cache_key(row)).read_bytes()==(old/'qwen_cache'/pipeline.cache_key(row)).read_bytes()
    pipeline.validate_cache(data)


@pytest.mark.parametrize('mutation',['role','image','feature','record_hash','config','partial'])
def test_cache_configuration_roles_and_arrays_are_strict(caches,mutation):
    data,old,new,manifest,rows,added=caches
    cfg=json.loads((new/'cache_config.json').read_text());records=pipeline.lines(new/'samples.jsonl')
    if mutation=='config':cfg['transformers']='different';pipeline.write(new/'cache_config.json',cfg)
    elif mutation=='partial':pipeline.write(new/'status.json',dict(samples=2,status='incomplete'))
    elif mutation=='record_hash':records[0]['sha256']='0'*64
    else:
        path=new/records[0]['file']
        with np.load(path) as a:arrays={k:a[k].copy() for k in a.files}
        if mutation=='role':arrays['split']=np.array('DEV_MODEL')
        elif mutation=='image':arrays['image_sha256']=np.array('0'*64)
        else:arrays['mean_hidden'][0]=np.nan
        np.savez_compressed(path,**arrays);records[0]['sha256']=pipeline.digest(path)
    (new/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    with pytest.raises(ValueError):pipeline.merge_cache(data,old/'qwen_cache',new,manifest,data/'qwen_cache')
    assert not (data/'qwen_cache').exists()


def test_new_cache_contract_cannot_change_dtype_or_torch(caches):
    _,old,new,_,_,_=caches
    a=json.loads((old/'qwen_cache/cache_config.json').read_text());b=json.loads((new/'cache_config.json').read_text())
    pipeline.same_cache_contract(a,b);b['torch']='different'
    with pytest.raises(ValueError,match='historical encoder'):pipeline.same_cache_contract(a,b)


def test_quality_gate_cannot_be_replaced_by_export_success(tmp_path,monkeypatch):
    pipeline.write(tmp_path/'export_manifest.json',{})
    monkeypatch.setattr(pipeline.exporter,'verify_export',lambda path:({},{}))
    monkeypatch.setattr(pipeline,'selected_train',lambda path:({},[]))
    pipeline.write(tmp_path/'failed_quality.json',dict(protocol=pipeline.QUALITY_PROTOCOL,capacity_gate_passed=False))
    with pytest.raises(ValueError,match='capacity audit'):pipeline.validate_quality(tmp_path,tmp_path/'failed_quality.json')


def test_cpu_stage_never_exposes_gpu(monkeypatch):
    if hasattr(pipeline.os,'sched_getaffinity'):monkeypatch.setattr(pipeline.os,'sched_getaffinity',lambda pid:{0})
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','1')
    with pytest.raises(ValueError,match='hide GPUs'):pipeline.resource_guard('quality')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','');pipeline.resource_guard('quality')


def test_unsealed_final_diagnostic_never_replays(tmp_path):
    output=tmp_path/'last_train';output.with_name('last_train.staging').mkdir()
    with pytest.raises(ValueError,match='no automatic forward'):pipeline.recover_diagnostic(output,'checkpoint','export')


def test_reused_cache_retains_new_actual_extraction_provenance(caches):
    data,old,new,manifest,_,_=caches
    pipeline.merge_cache(data,old/'qwen_cache',new,manifest,data/'qwen_cache')
    pipeline.write(new/'status.json',dict(samples=3,status='changed'))
    with pytest.raises(ValueError,match='provenance'):pipeline.validate_cache(data)
