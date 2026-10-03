"""Copy only completed validation/quality records; never inspect active cache."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
REMOTE=r'''
import datetime,hashlib,io,json,sys,tarfile
from pathlib import Path
p=Path('/home/wzy/dpvlm/route_set_v1');commit='71cf0c11d5b642672594bc28e8a69e5b1f4bdd0c'
source=p/'research_v2/releases'/commit;prep=p/'runs/observed_two_row_composite108_preparation_v1'
validation=p/'runs/observed_two_row_composite108_validation_v1'/commit
for status in (prep/'quality.status.json',validation/'tests.status.json'):
 value=json.loads(status.read_text());assert value['exit_code']==0 and value['status']=='completed' and value['code_commit']==commit
groups={'preparation':{name:prep/name for name in ('quality.status.json','quality.log','train_quality.json')},
        'validation':{name:validation/name for name in ('tests.status.json','tests.log','pytest.xml','registry.jsonl')}}
wrapper=p/'research_v2/incoming/composite108_pipeline_71cf0c1.sh'
for group in groups.values():group['wrapper/composite108_pipeline_71cf0c1.sh']=wrapper
for name in ('scripts/run_observed_two_row_composite.py','scripts/audit_two_row_train_reference_quality.py',
             'configs/observed_two_row_composite108_training_v1.json','tests/test_two_row_composite_pipeline.py',
             'tests/test_two_row_composite_training_torch.py','reports/TWO_ROW_COMPOSITE108_ORDINARY_PROTOCOL.md'):
 groups['validation']['source/'+name]=source/name
groups['preparation']['quality_source/scripts/run_observed_two_row_composite.py']=source/'scripts/run_observed_two_row_composite.py'
groups['preparation']['quality_source/scripts/audit_two_row_train_reference_quality.py']=source/'scripts/audit_two_row_train_reference_quality.py'
buffer=io.BytesIO()
with tarfile.open(fileobj=buffer,mode='w') as archive:
 for group,files in groups.items():
  index={}
  for name,path in sorted(files.items()):
   raw=path.read_bytes();index[name]={'source':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
   info=tarfile.TarInfo(group+'/'+name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
  receipt={'protocol':'composite108_completed_quality_validation_archive_v1','source_commit':commit,
    'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':index,'active_cache_opened':False,
    'raw_corpus_arrays_opened':False,'new_forward_requests':0,'new_searches':0}
  raw=(json.dumps(receipt,indent=2)+'\n').encode();name='QUALITY_REMOTE_ARTIFACT_INDEX.json' if group=='preparation' else 'REMOTE_ARTIFACT_INDEX.json'
  info=tarfile.TarInfo(group+'/'+name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
sys.stdout.buffer.write(buffer.getvalue())
'''

def main():
 result=subprocess.run(['ssh','wzy3090','taskset','-c','1','python3','-'],input=REMOTE.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
 tarpath=ROOT/'.bootstrap/composite108_quality_validation.tar'
 if tarpath.exists():raise FileExistsError('Do not replace original snapshot')
 tarpath.write_bytes(result.stdout)
 targets={'preparation':ROOT/'reports/observed_two_row_composite108_preparation_v1',
          'validation':ROOT/'reports/observed_two_row_composite108_validation_v1'}
 with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
  for entry in archive.getmembers():
   name=PurePosixPath(entry.name)
   if not entry.isfile() or name.is_absolute() or '..' in name.parts or '\\' in entry.name or name.parts[0] not in targets:raise ValueError('Invalid entry')
   path=targets[name.parts[0]].joinpath(*name.parts[1:]);raw=archive.extractfile(entry).read();path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing evidence differs: '+str(path))
   path.write_bytes(raw)
 for group,target in targets.items():
  name='QUALITY_REMOTE_ARTIFACT_INDEX.json' if group=='preparation' else 'REMOTE_ARTIFACT_INDEX.json'
  idx=json.loads((target/name).read_text())
  for filename,row in idx['files'].items():
   raw=(target/filename).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
  receipt=dict(archive_sha256=hashlib.sha256(result.stdout).hexdigest(),archive_bytes=len(result.stdout),verified_original_files=len(idx['files']),
    remote_index_sha256=hashlib.sha256((target/name).read_bytes()).hexdigest(),cache_results_read=False,new_forward_requests=0)
  (target/('QUALITY_SYNC_RECEIPT.json' if group=='preparation' else 'SYNC_RECEIPT.json')).write_text(json.dumps(receipt,indent=2)+'\n')
  print(group,json.dumps(receipt))

if __name__=='__main__':main()
