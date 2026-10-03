"""Archive only completed composite export metadata; no raw/JSONL corpus copies."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile

ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').exists())
DEST=ROOT/'reports/observed_two_row_composite108_preparation_v1'
REMOTE=r'''
import datetime,hashlib,io,json,sys,tarfile
from pathlib import Path
project=Path('/home/wzy/dpvlm/route_set_v1')
source=project/'research_v2/releases/1fccf4898bfa17233f92e20476adeb8db12c6b60'
run=project/'runs/observed_two_row_composite108_preparation_v1'
data=project/'data/observation_two_row_composite108_v1'
for stage in ('readiness','export'):
 s=json.loads((run/(stage+'.status.json')).read_text())
 assert s['status']=='completed' and s['exit_code']==0 and s['code_commit']==source.name
files={name:run/name for name in ('export.log','export.status.json','readiness.log','readiness.status.json','registry.jsonl')}
files['export_manifest.json']=data/'export_manifest.json'
files['wrapper/composite108_export_1fccf48.sh']=project/'research_v2/incoming/composite108_export_1fccf48.sh'
for name in ('scripts/export_two_row_composite_observations.py','configs/observed_two_row_composite108_v1.json'):
 files['source/'+name]=source/name
buffer=io.BytesIO();index={}
with tarfile.open(fileobj=buffer,mode='w') as archive:
 for name,path in sorted(files.items()):
  raw=path.read_bytes();index[name]={'source':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
 value={'protocol':'composite108_metadata_only_archive_v1','source_commit':source.name,'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':index,
        'raw_corpus_arrays_opened':False,'supervision_rows_copied':False,'observation_rows_copied':False,'new_forward_requests':0,'new_searches':0}
 raw=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo('REMOTE_ARTIFACT_INDEX.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
sys.stdout.buffer.write(buffer.getvalue())
'''

def main():
 result=subprocess.run(['ssh','wzy3090','taskset','-c','1','python3','-'],input=REMOTE.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 if result.returncode:raise RuntimeError(result.stderr.decode(errors='replace'))
 tarpath=ROOT/'.bootstrap/composite108_preparation_metadata.tar'
 if tarpath.exists():raise FileExistsError('Preserve previous archive')
 tarpath.write_bytes(result.stdout)
 with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
  for member in archive.getmembers():
   name=PurePosixPath(member.name)
   if not member.isfile() or name.is_absolute() or '..' in name.parts or '\\' in member.name:raise ValueError('Unsafe archive entry')
   path=DEST/member.name;raw=archive.extractfile(member).read();path.parent.mkdir(parents=True,exist_ok=True)
   if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing evidence differs')
   path.write_bytes(raw)
 index=json.loads((DEST/'REMOTE_ARTIFACT_INDEX.json').read_text())
 for name,row in index['files'].items():
  raw=(DEST/name).read_bytes();assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 receipt=dict(archive_sha256=hashlib.sha256(result.stdout).hexdigest(),archive_bytes=len(result.stdout),verified_files=len(index['files']),
              remote_artifact_index_sha256=hashlib.sha256((DEST/'REMOTE_ARTIFACT_INDEX.json').read_bytes()).hexdigest(),
              read_only=True,raw_corpus_arrays_opened=False,supervision_rows_copied=False,observation_rows_copied=False,new_forward_requests=0)
 (DEST/'SYNC_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(receipt))

if __name__=='__main__':main()
