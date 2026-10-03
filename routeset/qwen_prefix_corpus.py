"""Sealed serial Qwen prefix corpus reader; no final features or labels.

Payloads are ordinary CPU tensors/primitives and are deserialized with
weights_only=True. First get(id) verifies bytes and tensor structure; subsequent
gets reuse that verified RAM snapshot. Callers must not mutate returned tensors.
"""
import hashlib
import json
from pathlib import Path

PROTOCOL = 'qwen3vl_prefix_corpus_v1'
PAYLOAD_PROTOCOL = 'qwen3vl_frozen_layer26_serial_v1'
EXPORT_SHA = '04ba27290e382787d5f1764fc2645d4219b09457ea972d15f7d1f3baeac74fdc'
CACHE_RECEIPT_SHA = '9f2ccf2179192da7d0a0478bc643d695b9c13495866cb714229d5f97f8615cff'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
COUNTS = {'TRAIN':285, 'DEV_MODEL':36, 'total':321}
BUDGET = {'full_features':321, 'replay_features':321, 'head_calls':0,
          'candidate_path_states':0, 'optimizer_steps':0}
PROBE = {
    'source_commit':'eddfacaddab2d12c67f5a56fd775de173c05f9b2',
    'status_sha256':'528d730af8a1a0c7910605883d56e856963a40aa2efae662c0553609fa2718fc',
    'preflight_sha256':'23d120d6770e32f69c0fd5932d04ec1f1b8bb4b93d0b9a88374e93e05cc100d1',
    'artifact_index_sha256':'776553d2b5a1d81cb2313c3c12ef88de5dab667961fdff7436a28d8fdae2b78a',
    'final_audit_sha256':'32ebdbdc62eb42e92877d82f0ee05f7004fcdac445d52f54c8e65b8139540673'}
REPLAY_SOURCE_SHA = '661a430d8e41de6be55845a52096f588621ff7e3202996e06f094e9b7a5fcf37'
MODEL_PROVENANCE_SHA = 'a001afb3ebdeceee3548ea9c8e58d5c9c4443b4d9a0ecdac004c85de3a691819'
EXPECTED_PARENTS = tuple(('two_row_reach_%d'%(283200+i), 'TRAIN' if i<64 else 'DEV_MODEL')
                         for i in range(76) if i!=20) + tuple(
                         ('two_row_reach_%d'%(400000+i),'TRAIN') for i in range(32))
EXPECTED_ROWS = tuple((p+'_target'+str(t),p,role) for p,role in EXPECTED_PARENTS for t in range(3))
IDS = tuple(r[0] for r in EXPECTED_ROWS)
OBSERVATION_KEYS = {'id','parent_id','split','image','instruction'}


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(2**20),b''):digest.update(value)
    return digest.hexdigest()


def payload_filename(identity):
    if identity not in IDS:raise ValueError('Unregistered prefix identity')
    return 'payloads/'+hashlib.sha256(identity.encode()).hexdigest()+'.pt'


def validate_observations(rows):
    if len(rows)!=321:raise ValueError('Exactly321 existing authorized observations required')
    for row,expected in zip(rows,EXPECTED_ROWS):
        if (set(row)!=OBSERVATION_KEYS or tuple(row.get(k) for k in ('id','parent_id','split'))!=expected
                or not isinstance(row['image'],str) or not row['image']
                or not isinstance(row['instruction'],str) or not row['instruction'].strip()):
            raise ValueError('Fixed composite observation identity/order/whitelist changed')
    return rows


def validate_policy(value):
    expected=dict(protocol=PROTOCOL,payload_protocol=PAYLOAD_PROTOCOL,
        export_sha256=EXPORT_SHA,historical_cache_receipt_sha256=CACHE_RECEIPT_SHA,
        model_revision=REVISION,processor_revision=REVISION,probe=PROBE,counts=COUNTS,
        budget=BUDGET,cut_layer=26,language_layers=28,hidden_size=2048,
        min_pixels=65536,max_pixels=262144,pooling='both',serial_only=True,
        dtype='torch.bfloat16',attention_backend='sdpa',use_cache=False,
        retry=False,resume=False,new_dev_allowed=False,raw_supervision_allowed=False,
        seed=0,head_checkpoint_sha256='ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3')
    if value!=expected:raise ValueError('Fixed prefix corpus policy changed')
    return value


def tensor_metadata(value):
    import torch
    from routeset.qwen_prefix_replay import tensor_hash
    if torch.is_tensor(value):
        return dict(kind='tensor',shape=list(value.shape),dtype=str(value.dtype),
                    device=str(value.device),requires_grad=value.requires_grad,sha256=tensor_hash(value))
    if isinstance(value,dict):return {k:tensor_metadata(v) for k,v in value.items()}
    if isinstance(value,tuple):return {'kind':'tuple','values':[tensor_metadata(v) for v in value]}
    if isinstance(value,list):return {'kind':'list','values':[tensor_metadata(v) for v in value]}
    if value is None or type(value) in (bool,int,float,str):return dict(kind='literal',value=value)
    raise ValueError('Unsupported serialized prefix value')


def validate_payload(payload,sequence_length,input_tokens,metadata=None):
    import torch
    if not isinstance(payload,dict) or set(payload)!={'hidden','kwargs','pooling_mask','cut_layer'} or payload['cut_layer']!=26:
        raise ValueError('Only the frozen layer26 payload is accepted')
    def cpu_tensor(value):
        return torch.is_tensor(value) and value.device.type=='cpu' and not value.requires_grad and not value.is_inference()
    def recurse(value):
        if torch.is_tensor(value):
            if not cpu_tensor(value):raise ValueError('Prefix must contain normal detached CPU tensors only')
        elif isinstance(value,dict):
            if any(not isinstance(k,str) for k in value):raise ValueError('String metadata keys required')
            for v in value.values():recurse(v)
        elif isinstance(value,(tuple,list)):
            for v in value:recurse(v)
        elif value is not None and type(value) not in (bool,int,float,str):
            raise ValueError('Unsupported prefix object')
    recurse(payload)
    hidden,mask,kwargs=payload['hidden'],payload['pooling_mask'],payload['kwargs']
    if (not isinstance(sequence_length,int) or not 1<=sequence_length<=4096 or not 1<=input_tokens<=sequence_length
            or not cpu_tensor(hidden) or hidden.shape!=(1,sequence_length,2048) or hidden.dtype!=torch.bfloat16
            or not bool(torch.isfinite(hidden).all()) or not cpu_tensor(mask) or mask.shape!=(1,sequence_length)
            or mask.dtype not in (torch.int64,torch.bool) or int(mask.bool().sum())!=input_tokens):
        raise ValueError('Frozen prefix hidden/pooling-mask shape or dtype changed')
    required={'position_embeddings','attention_mask','position_ids','cache_position','past_key_values'}
    allowed=required|{'use_cache','output_attentions','output_hidden_states','return_dict'}
    if (not isinstance(kwargs,dict) or not required.issubset(kwargs) or not set(kwargs).issubset(allowed)
            or kwargs['past_key_values'] is not None or kwargs.get('use_cache',False)):
        raise ValueError('Decoder kwargs changed or KV reuse requested')
    embeddings=kwargs['position_embeddings']
    if (not isinstance(embeddings,tuple) or len(embeddings)!=2 or any(not cpu_tensor(v) or
            v.shape!=(1,sequence_length,128) or v.dtype!=hidden.dtype or not bool(torch.isfinite(v).all()) for v in embeddings)):
        raise ValueError('Original multimodal rotary cos/sin required')
    if (not cpu_tensor(kwargs['position_ids']) or kwargs['position_ids'].shape!=(1,sequence_length)
            or kwargs['position_ids'].dtype!=torch.int64 or not cpu_tensor(kwargs['cache_position'])
            or kwargs['cache_position'].shape!=(sequence_length,) or kwargs['cache_position'].dtype!=torch.int64):
        raise ValueError('Original position/cache-position tensors required')
    attention=kwargs['attention_mask']
    if attention is not None and (not cpu_tensor(attention) or attention.shape!=(1,1,sequence_length,sequence_length)
                                  or attention.dtype not in (torch.bool,torch.float32,torch.bfloat16)):
        raise ValueError('Original serial causal mask required')
    actual=tensor_metadata(payload)
    if metadata is not None and actual!=metadata:raise ValueError('Payload metadata/tensor hashes changed')
    return actual


def checked_payload_path(root,row):
    root=Path(root).resolve(strict=True)
    if row.get('file')!=payload_filename(row['id']):raise ValueError('Unsafe/noncanonical payload path')
    path=root/row['file'];resolved=path.resolve(strict=True)
    try:resolved.relative_to(root)
    except ValueError as exc:raise ValueError('Payload path escapes corpus') from exc
    if resolved!=path.absolute() or not resolved.is_file() or resolved.stat().st_size!=row['bytes']:
        raise ValueError('Payload symlink/size mismatch')
    return path


def validate_manifest(manifest):
    required=dict(protocol=PROTOCOL,payload_protocol=PAYLOAD_PROTOCOL,export_sha256=EXPORT_SHA,
        historical_cache_receipt_sha256=CACHE_RECEIPT_SHA,model_revision=REVISION,
        processor_revision=REVISION,probe=PROBE,counts=COUNTS,serial_only=True,
        raw_supervision_opened=False,new_dev_raw_opened=False,optimizer_steps=0,head_calls=0,
        frozen_weights_unchanged=True,replay_source_sha256=REPLAY_SOURCE_SHA,
        model_provenance_sha256=MODEL_PROVENANCE_SHA)
    if any(manifest.get(k)!=v for k,v in required.items()):raise ValueError('Sealed corpus provenance/header mismatch')
    for field in ('model_assets_sha256','source_sha256'):
        values=manifest.get(field,{})
        if not isinstance(values,dict) or not values or any(not isinstance(v,str) or len(v)!=64 for v in values.values()):
            raise ValueError('Complete source/asset hashes required')
    if any(Path(name).name!=name for name in manifest['model_assets_sha256']):raise ValueError('Unsafe model asset filename')
    rows=manifest.get('rows',[])
    if len(rows)!=321 or tuple(r.get('id') for r in rows)!=IDS:raise ValueError('Incomplete or reordered corpus identities')
    sources=[]
    for row,expected in zip(rows,EXPECTED_ROWS):
        if tuple(row.get(k) for k in ('id','parent_id','split'))!=expected:raise ValueError('Corpus parent/role changed')
        if (row.get('file')!=payload_filename(row['id']) or not isinstance(row.get('bytes'),int) or row['bytes']<=0
                or len(row.get('sha256',''))!=64 or row.get('comparisons')!={'historical_exact':True,'replay_exact':True}):
            raise ValueError('Unsealed or invalid per-input prefix record')
        source=row['source'];sources.append(source['observation_row'])
        if any(len(source.get(k,''))!=64 for k in ('image_sha256','historical_npz_sha256')):
            raise ValueError('Missing input/cache source hash')
    validate_observations(sources)
    if manifest.get('payload_bytes')!=sum(row['bytes'] for row in rows):raise ValueError('Payload byte accounting differs')
    return manifest


class PrefixCorpus:
    def __init__(self,root,expected_fingerprint=None):
        self.root=Path(root).resolve(strict=True)
        self.fingerprint=sha(self.root/'manifest.json')
        if expected_fingerprint is not None and self.fingerprint!=expected_fingerprint:
            raise ValueError('Checkpoint expects a different prefix corpus fingerprint')
        self.manifest=validate_manifest(json.loads((self.root/'manifest.json').read_text()))
        status=json.loads((self.root/'status.json').read_text())
        if (status.get('protocol')!=PROTOCOL or status.get('status')!='completed' or not status.get('gate_passed')
                or status.get('exit_code')!=0 or status.get('manifest_sha256')!=self.fingerprint
                or status.get('actual_issued_budget')!=BUDGET or status.get('completed_rows')!=321):
            raise ValueError('A completely sealed successful corpus is required')
        index=json.loads((self.root/'artifact_index.json').read_text())
        for name in ('manifest.json','status.json'):
            if index.get(name,{}).get('sha256')!=sha(self.root/name):raise ValueError('Corpus final index is missing or changed')
        self.ids=IDS
        self.rows_by_id={row['id']:row for row in self.manifest['rows']}
        self._payloads={}
        for row in self.manifest['rows']:checked_payload_path(self.root,row)

    def get(self,identity):
        if identity not in self.rows_by_id:raise KeyError('Unregistered prefix id: '+str(identity))
        if identity not in self._payloads:
            import torch
            row=self.rows_by_id[identity];path=checked_payload_path(self.root,row)
            if sha(path)!=row['sha256']:raise ValueError('Prefix payload bytes changed: '+identity)
            payload=torch.load(path,map_location='cpu',weights_only=True)
            validate_payload(payload,row['sequence_length'],row['input_tokens'],row['tensor_metadata'])
            self._payloads[identity]=payload
        return self._payloads[identity]
