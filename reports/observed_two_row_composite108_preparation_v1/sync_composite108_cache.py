"""Completed composite cache metadata and NPZ hashes only; no training reads."""
import hashlib,io,json,subprocess,tarfile
from pathlib import Path,PurePosixPath
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
DEST=ROOT/'reports/observed_two_row_composite108_preparation_v1'
REMOTE=r'''
import datetime,hashlib,io,json,sys,tarfile
from pathlib import Path
p=Path('/home/wzy/dpvlm/route_set_v1');prep=p/'runs/observed_two_row_composite108_preparation_v1'
cache=p/'data/observation_two_row_composite108_v1/qwen_cache'
s=json.loads((prep/'cache.status.json').read_text());assert s['status']=='completed' and s['exit_code']==0 and s['code_commit']=='71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c'
files={n:prep/n for n in ('cache.status.json','cache.log')}
files['cache/registry_at_completion.jsonl']=prep/'registry.jsonl'
for path in sorted((prep/'cache').rglob('*')):
 if path.is_file() and path.suffix in ('.json','.jsonl','.log'):files['cache/'+str(path.relative_to(prep/'cache'))]=path
for name in ('cache_config.json','samples.jsonl','status.json','composite_cache_receipt.json'):
 files['merged_qwen_cache/'+name]=cache/name
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
receipt=json.loads((cache/'composite_cache_receipt.json').read_text())
samples=[json.loads(x) for x in (cache/'samples.jsonl').read_text().splitlines()]
assert len(samples)==321 and receipt['old_reused_count']==225 and receipt['new_encoding_count']==96 and receipt['reused_dev_count']==36
oldids={r['id'] for r in receipt['old225_by_id']};assert len(oldids)==225
assert {r['file'] for r in samples}=={x.name for x in cache.glob('*.npz')}
npz=[]
for row in samples:
 path=cache/row['file'];value=digest(path);origin=Path(receipt['old_cache'] if row['id'] in oldids else receipt['new_cache'])/row['file']
 assert value==row['sha256']==digest(origin)
 npz.append({'id':row['id'],'filename':row['file'],'path':str(path),'sha256':value,'bytes':path.stat().st_size,
   'origin':str(origin),'reused_old':row['id'] in oldids,'origin_bytes_sha256_equal':True,'downloaded':False})
generated={'CACHE_NPZ_HASH_INDEX.json':{'protocol':'composite108_321_cache_npz_hash_only_v1','npz_count':321,
 'old_reused':225,'newly_encoded':96,'all_source_bytes_equal':True,'files':npz,'new_forward_requests':0,'cache_array_values_decoded_by_archive':False}}
buffer=io.BytesIO();index={}
with tarfile.open(fileobj=buffer,mode='w') as archive:
 for name,path in sorted(files.items()):
  raw=path.read_bytes();index[name]={'source':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
 for name,value in generated.items():
  raw=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
 value={'protocol':'composite108_completed_cache_metadata_archive_v1','source_commit':s['code_commit'],
 'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':index,'NPZ_downloaded':False,
 'training_results_opened':False,'new_forward_requests':0,'new_searches':0}
 raw=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo('CACHE_REMOTE_ARTIFACT_INDEX.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
sys.stdout.buffer.write(buffer.getvalue())
'''
def main():
 r=subprocess.run(['ssh','wzy3090','taskset','-c','1','python3','-'],input=REMOTE.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace'))
 tar=ROOT/'.bootstrap/composite108_completed_cache_metadata.tar'
 if tar.exists():raise FileExistsError('Do not overwrite archived bytes')
 tar.write_bytes(r.stdout)
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as archive:
  for e in archive.getmembers():
   n=PurePosixPath(e.name)
   if not e.isfile() or n.is_absolute() or '..' in n.parts or '\\' in e.name:raise ValueError('Unsafe archive entry')
   data=archive.extractfile(e).read();path=DEST/e.name;path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists() and path.read_bytes()!=data:raise ValueError('Existing evidence differs: '+str(path))
   path.write_bytes(data)
 idx=json.loads((DEST/'CACHE_REMOTE_ARTIFACT_INDEX.json').read_text())
 for name,v in idx['files'].items():
  b=(DEST/name).read_bytes();assert len(b)==v['bytes'] and hashlib.sha256(b).hexdigest()==v['sha256']
 receipt=dict(archive_sha256=hashlib.sha256(r.stdout).hexdigest(),archive_bytes=len(r.stdout),verified_original_metadata_files=len(idx['files']),
  NPZ_hashes_verified=321,NPZ_downloaded=False,training_results_opened=False,new_forward_requests=0)
 (DEST/'CACHE_SYNC_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8');print(json.dumps(receipt))
if __name__=='__main__':main()
