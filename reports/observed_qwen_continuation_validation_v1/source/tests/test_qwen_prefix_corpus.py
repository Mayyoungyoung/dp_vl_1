import copy
import json
from pathlib import Path

import pytest

from routeset import qwen_prefix_corpus as core
from scripts.cache_observed_qwen_prefix import CaptureLedger

ROOT=Path(__file__).resolve().parents[1]


def policy():return json.loads((ROOT/'configs/two_row_qwen_prefix_corpus_v1.json').read_text())


def observations():
    return [dict(id=i,parent_id=p,split=s,image='/images/'+p+'/front.png',instruction='Reach the blue target.')
            for i,p,s in core.EXPECTED_ROWS]


def manifest_fixture():
    rows=[]
    for row in observations():
        rows.append(dict(id=row['id'],parent_id=row['parent_id'],split=row['split'],
            file=core.payload_filename(row['id']),sha256='3'*64,bytes=1,sequence_length=3,input_tokens=3,
            source=dict(observation_row=row,image_sha256='4'*64,historical_npz_sha256='5'*64),
            tensor_metadata={},comparisons=dict(historical_exact=True,replay_exact=True)))
    return dict(protocol=core.PROTOCOL,payload_protocol=core.PAYLOAD_PROTOCOL,export_sha256=core.EXPORT_SHA,
        historical_cache_receipt_sha256=core.CACHE_RECEIPT_SHA,model_revision=core.REVISION,
        processor_revision=core.REVISION,probe=copy.deepcopy(core.PROBE),counts=copy.deepcopy(core.COUNTS),
        serial_only=True,raw_supervision_opened=False,new_dev_raw_opened=False,optimizer_steps=0,head_calls=0,
        frozen_weights_unchanged=True,replay_source_sha256=core.REPLAY_SOURCE_SHA,
        model_provenance_sha256=core.MODEL_PROVENANCE_SHA,
        source_sha256={'routeset/qwen_prefix_replay.py':core.REPLAY_SOURCE_SHA},
        model_assets_sha256={'config.json':'6'*64},payload_bytes=321,rows=rows)


def seal(root,manifest):
    (root/'manifest.json').write_text(json.dumps(manifest))
    fingerprint=core.sha(root/'manifest.json')
    status=dict(protocol=core.PROTOCOL,status='completed',gate_passed=True,exit_code=0,
                manifest_sha256=fingerprint,actual_issued_budget=core.BUDGET,completed_rows=321)
    (root/'status.json').write_text(json.dumps(status))
    index={name:dict(sha256=core.sha(root/name)) for name in ('manifest.json','status.json')}
    (root/'artifact_index.json').write_text(json.dumps(index))
    return fingerprint


def test_fixed321_identity_roles_and_budget():
    assert core.validate_policy(policy())
    rows=core.validate_observations(observations())
    assert sum(r['split']=='TRAIN' for r in rows)==285
    assert sum(r['split']=='DEV_MODEL' for r in rows)==36
    assert len({r['parent_id'] for r in rows})==107
    assert not any('283220' in r['id'] for r in rows)
    assert core.BUDGET==dict(full_features=321,replay_features=321,head_calls=0,candidate_path_states=0,optimizer_steps=0)


@pytest.mark.parametrize('key,value',[('retry',True),('resume',True),('serial_only',False),
    ('cut_layer',25),('new_dev_allowed',True),('raw_supervision_allowed',True)])
def test_no_policy_expansion(key,value):
    p=policy();p[key]=value
    with pytest.raises(ValueError):core.validate_policy(p)


@pytest.mark.parametrize('kind',['new_dev','reorder','subset','answer','replacement'])
def test_observation_contract_rejects_subsets_newdev_labels(kind):
    data=observations()
    if kind=='new_dev':data[-1]['split']='DEV_MODEL'
    elif kind=='reorder':data[0],data[1]=data[1],data[0]
    elif kind=='subset':data.pop()
    elif kind=='answer':data[0]['labels']=[1,2,3]
    else:data[-1]['id']='two_row_reach_400032_target0'
    with pytest.raises(ValueError):core.validate_observations(data)


def test_failed_call_spends_slot_and_cannot_retry(tmp_path):
    ledger=CaptureLedger(tmp_path)
    with pytest.raises(ValueError):ledger.call('replay',core.IDS[0],lambda:1)
    def fail():raise RuntimeError('injected')
    with pytest.raises(RuntimeError):ledger.call('full',core.IDS[0],fail)
    assert ledger.counts()['full_features']==1
    with pytest.raises(ValueError):ledger.call('full',core.IDS[0],lambda:1)
    with pytest.raises(ValueError):ledger.call('replay',core.IDS[0],lambda:1)
    assert json.loads((tmp_path/'call_ledger.json').read_text())[0]['state']=='failed'


def test_ledger_exhaustion_no_hidden_head_or_optimizer(tmp_path):
    ledger=CaptureLedger(tmp_path)
    ledger.records=[dict(kind='full',id=i,state='completed') for i in core.IDS]
    with pytest.raises(ValueError):ledger.call('full',core.IDS[0],lambda:1)
    for kind in ('head','optimizer'):
        with pytest.raises(ValueError):ledger.call(kind,core.IDS[0],lambda:1)
    assert ledger.counts()['candidate_path_states']==0


@pytest.mark.parametrize('kind',['count','source','role','comparison','path','bytes','probe'])
def test_manifest_requires_full_sealed_identity(kind):
    value=manifest_fixture();assert core.validate_manifest(value)
    if kind=='count':value['counts']['TRAIN']=284
    elif kind=='source':value['replay_source_sha256']='0'*64
    elif kind=='role':value['rows'][0]['split']='TEST_LOCKED'
    elif kind=='comparison':value['rows'][0]['comparisons']['replay_exact']=False
    elif kind=='path':value['rows'][0]['file']='../escape.pt'
    elif kind=='bytes':value['payload_bytes']=320
    else:value['probe']['source_commit']='changed'
    with pytest.raises(ValueError):core.validate_manifest(value)


def test_reader_rejects_missing_seal_fingerprint_and_file(tmp_path):
    value=manifest_fixture();(tmp_path/'payloads').mkdir()
    for row in value['rows']:(tmp_path/row['file']).write_bytes(b'x')
    fingerprint=seal(tmp_path,value)
    reader=core.PrefixCorpus(tmp_path,expected_fingerprint=fingerprint)
    assert reader.ids==core.IDS and reader.fingerprint==fingerprint
    with pytest.raises(ValueError):core.PrefixCorpus(tmp_path,expected_fingerprint='0'*64)
    (tmp_path/value['rows'][-1]['file']).unlink()
    with pytest.raises(FileNotFoundError):core.PrefixCorpus(tmp_path)


def test_reader_rejects_incomplete_status(tmp_path):
    value=manifest_fixture();seal(tmp_path,value)
    status=json.loads((tmp_path/'status.json').read_text());status['completed_rows']=320
    (tmp_path/'status.json').write_text(json.dumps(status))
    with pytest.raises(ValueError):core.PrefixCorpus(tmp_path)


def test_payload_symlink_and_escape(tmp_path):
    row=manifest_fixture()['rows'][0];(tmp_path/'payloads').mkdir()
    outside=tmp_path/'outside';outside.write_bytes(b'x')
    try:(tmp_path/row['file']).symlink_to(outside)
    except OSError:pytest.skip('symlink privilege unavailable; server Linux covers this')
    with pytest.raises(ValueError):core.checked_payload_path(tmp_path,row)
    row['file']='../outside'
    with pytest.raises(ValueError):core.checked_payload_path(tmp_path,row)


def test_cache_validation_only_registered321_npzs_not_labels(tmp_path,monkeypatch):
    import hashlib
    import numpy as np
    from scripts import cache_observed_qwen_prefix as script
    cache=tmp_path/'qwen_cache';cache.mkdir();rows=observations()
    (tmp_path/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config=dict(model='Qwen/Qwen3-VL-2B-Instruct',revision=core.REVISION,processor=core.REVISION,
        manifest_sha256=core.sha(tmp_path/'observations.jsonl'),max_pixels=262144,dtype='torch.bfloat16',
        model_trainable_parameter_count=0,transformers='4.57.1',input_contract=list(core.OBSERVATION_KEYS))
    (cache/'cache_config.json').write_text(json.dumps(config))
    (cache/'status.json').write_text(json.dumps(dict(samples=321,status='frozen_rgb_language_feature_extraction_complete')))
    records=[];fake={};artifacts={}
    for row in rows:
        name=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]+'.npz'
        (cache/name).write_bytes(row['id'].encode());h=core.sha(cache/name);artifacts[name]=h
        records.append(dict(id=row['id'],file=name,sha256=h,image_sha256='1'*64,input_tokens=3))
        fake[name]=dict(mean_hidden=np.zeros(2048,np.float32),last_hidden=np.ones(2048,np.float32),
            id=np.array(row['id']),parent_id=np.array(row['parent_id']),split=np.array(row['split']),
            image_sha256=np.array('1'*64),input_tokens=np.array(3))
    (cache/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    artifacts.update({name:core.sha(cache/name) for name in ('cache_config.json','samples.jsonl','status.json')})
    receipt=dict(source_export_manifest_sha256=core.EXPORT_SHA,old_reused_count=225,new_encoding_count=96,artifact_sha256=artifacts)
    (cache/'composite_cache_receipt.json').write_text(json.dumps(receipt))
    monkeypatch.setattr(core,'CACHE_RECEIPT_SHA',core.sha(cache/'composite_cache_receipt.json'))
    monkeypatch.setattr(script,'checked_image',lambda row,manifest:'1'*64)
    opened=[]
    class Archive(dict):
        @property
        def files(self):return list(self)
        def __enter__(self):return self
        def __exit__(self,*args):pass
    def load(path,**kwargs):
        assert kwargs=={'allow_pickle':False}
        opened.append(Path(path).name);return Archive(fake[Path(path).name])
    monkeypatch.setattr(np,'load',load)
    selected,_=script.inspect_frozen_cache(tmp_path,dict(output_files_sha256={'observations.jsonl':core.sha(tmp_path/'observations.jsonl')}))
    assert len(selected)==len(opened)==321 and set(opened)==set(fake)
    assert not (tmp_path/'supervision.jsonl').exists()


def torch_payload():
    torch=pytest.importorskip('torch');length=3
    return dict(hidden=torch.zeros(1,length,2048,dtype=torch.bfloat16),pooling_mask=torch.ones(1,length,dtype=torch.int64),
        cut_layer=26,kwargs=dict(position_embeddings=(torch.ones(1,length,128,dtype=torch.bfloat16),torch.zeros(1,length,128,dtype=torch.bfloat16)),
        attention_mask=None,position_ids=torch.arange(length)[None],cache_position=torch.arange(length),past_key_values=None))


@pytest.mark.parametrize('kind',['shape','dtype','past','extra','gradient','cos'])
def test_cpu_payload_contract_rejects_changes(kind):
    torch=pytest.importorskip('torch');payload=torch_payload()
    assert core.validate_payload(payload,3,3)
    if kind=='shape':payload['hidden']=payload['hidden'][:,:,:12]
    elif kind=='dtype':payload['hidden']=payload['hidden'].float()
    elif kind=='past':payload['kwargs']['past_key_values']=[1]
    elif kind=='extra':payload['kwargs']['target_xyz']=[1,2,3]
    elif kind=='gradient':payload['hidden'].requires_grad_(True)
    else:payload['kwargs']['position_embeddings']=(torch.ones(1,3,64,dtype=torch.bfloat16),)*2
    with pytest.raises(ValueError):core.validate_payload(payload,3,3)


def test_get_safe_cpu_load_memoization_and_tamper(tmp_path,monkeypatch):
    import io
    torch=pytest.importorskip('torch');payload=torch_payload();buffer=io.BytesIO();torch.save(payload,buffer)
    contents=buffer.getvalue();(tmp_path/'payloads').mkdir();value=manifest_fixture()
    metadata=core.validate_payload(payload,3,3)
    for row in value['rows']:
        path=tmp_path/row['file'];path.write_bytes(contents)
        row.update(sha256=core.sha(path),bytes=len(contents),tensor_metadata=metadata)
    value['payload_bytes']=321*len(contents);fingerprint=seal(tmp_path,value)
    reader=core.PrefixCorpus(tmp_path,fingerprint);original=torch.load;calls=[]
    def load(*args,**kwargs):calls.append(kwargs);return original(*args,**kwargs)
    monkeypatch.setattr(torch,'load',load)
    a=reader.get(core.IDS[0]);b=reader.get(core.IDS[0])
    assert a is b and calls==[{'map_location':'cpu','weights_only':True}]
    assert a['hidden'].device.type=='cpu' and a['hidden'].dtype==torch.bfloat16
    with (tmp_path/value['rows'][1]['file']).open('ab') as stream:stream.write(b'tamper')
    with pytest.raises(ValueError):reader.get(core.IDS[1])
    with pytest.raises(KeyError):reader.get('two_row_reach_400256_target0')
