"""Read-only archive of the completed bounded TRAIN6 probe; no PT download."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'AGENTS.md').is_file())
OUTPUT = ROOT / 'reports/observed_qwen_prefix_replay_probe_v1'
ARCHIVE = ROOT / '.bootstrap/qwen_prefix_probe_eddfaca_metadata.tar'
REMOTE = r'''
from pathlib import Path
import hashlib, io, json, sys, tarfile
root=Path('/home/wzy/dpvlm/route_set_v1')
commit='eddfacaddab2d12c67f5a56fd775de173c05f9b2'
release=root/'research_v2/releases'/commit
groups={
 '': root/'runs/observed_qwen_prefix_replay_probe_v1',
 'validation': root/'runs/observed_qwen_prefix_replay_validation_v1'/commit,
 'environment': root/'runs/observed_qwen_prefix_probe_environment_v1',
}
for f in (groups['']/'probe.status.json',groups['']/'probe/status.json',groups['validation']/'tests.status.json',groups['environment']/'pytest_install.status.json'):
 d=json.loads(f.read_text())
 if d.get('status')!='completed' or d.get('exit_code')!=0: raise RuntimeError('Not a completed success: '+str(f))
pre=json.loads((groups['']/'probe/preflight.json').read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''): h.update(b)
 return h.hexdigest()
files=[]
for prefix,base in groups.items():
 for f in sorted(base.rglob('*')):
  if f.is_file(): files.append((str(Path(prefix)/f.relative_to(base)),f))
for name,expected in pre['source_sha256'].items():
 f=release/name
 if sha(f)!=expected: raise ValueError('Immutable source mismatch '+name)
 files.append(('source/'+name,f))
for name in ('configs/two_row_qwen_prefix_replay_probe_v1.json','tests/test_qwen_prefix_replay.py','tests/test_observed_geometry.py','tests/test_two_row_observation_training.py'):
 files.append(('source/'+name,release/name))
files.append(('wrapper/qwen_prefix_probe_eddfaca.sh',root/'research_v2/incoming/qwen_prefix_probe_eddfaca.sh'))
index=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for local,f in files:
  downloaded=f.suffix not in ('.pt','.npz','.bin','.safetensors')
  item=dict(local_path=local,remote_path=str(f),sha256=sha(f),bytes=f.stat().st_size,downloaded=downloaded)
  index.append(item)
  if downloaded: tar.add(f,arcname=local,recursive=False)
 receipt=dict(protocol='readonly_qwen_prefix_probe_archive_v1',commit=commit,files=index,
   downloaded_pt_files=0,new_model_forwards=0,new_searches=0,raw_dataset_files_opened=0)
 data=(json.dumps(receipt,indent=2)+'\n').encode()
 item=tarfile.TarInfo('REMOTE_ARTIFACT_INDEX.json');item.size=len(data);tar.addfile(item,io.BytesIO(data))
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open('wb') as handle:
        run = subprocess.run(['ssh', 'wzy3090', 'taskset', '-c', '1', 'python3', '-'],
                             input=REMOTE.encode(), stdout=handle, stderr=subprocess.PIPE)
    if run.returncode:
        raise RuntimeError(run.stderr.decode())
    with tarfile.open(ARCHIVE) as tar:
        index = json.load(tar.extractfile('REMOTE_ARTIFACT_INDEX.json'))
        for item in tar.getmembers():
            if not item.isfile() or Path(item.name).is_absolute() or '..' in Path(item.name).parts:
                raise ValueError('Unsafe archive member')
            target = OUTPUT / item.name
            data = tar.extractfile(item).read()
            if target.exists() and target.read_bytes() != data:
                raise ValueError('Preserve different pre-existing archive bytes: '+str(target))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    for item in index['files']:
        if item['downloaded']:
            target=OUTPUT/item['local_path']
            assert sha(target)==item['sha256'] and target.stat().st_size==item['bytes']
    receipt=dict(archive_sha256=sha(ARCHIVE),archive_bytes=ARCHIVE.stat().st_size,
        remote_files=len(index['files']),downloaded_files=sum(x['downloaded'] for x in index['files']),
        remote_only_files=sum(not x['downloaded'] for x in index['files']),
        all_downloaded_bytes_verified=True,new_forward_requests=0,new_searches=0)
    (OUTPUT/'SYNC_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(receipt))


if __name__=='__main__':main()
