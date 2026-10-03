from pathlib import Path
import json,hashlib,subprocess,tarfile,io,ast
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file());OUT=ROOT/'reports/observed_qwen_continuation_validation_v1';COMMIT='f41be1ff1a35b0eab3d36edf4ca23197359f4731'
REMOTE=r'''
from pathlib import Path
import json,hashlib,tarfile,io,sys,ast
r=Path('/home/wzy/dpvlm/route_set_v1');c='f41be1ff1a35b0eab3d36edf4ca23197359f4731';src=r/'research_v2/releases'/c
base=r/'runs/observed_qwen_continuation_validation_v1'/c
status=json.loads((base/'tests.status.json').read_text());assert status['status']=='completed' and status['exit_code']==0 and status['code_commit']==c
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names={'routeset/observed_qwen_continuation.py','scripts/train_observed_two_row_lora_continuation.py','configs/observed_two_row_lora_continuation_v1.json','tests/test_observed_qwen_continuation.py','reports/TWO_ROW_COMMON_HEAD_LORA_CONTINUATION_PROTOCOL.md','tests/test_qwen_prefix_corpus.py','tests/test_qwen_prefix_replay.py','scripts/record_job.py'}
module=ast.parse((src/'scripts/train_observed_two_row_lora_continuation.py').read_text())
for node in module.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SOURCE_FILES' for t in node.targets):names.update(ast.literal_eval(node.value))
files=[('validation/'+str(f.relative_to(base)),f) for f in sorted(base.rglob('*')) if f.is_file()]
files += [('source/'+n,src/n) for n in sorted(names)]
wrapper=r/'research_v2/incoming/qwen_continuation_f41be1f.sh';files += [('wrapper/'+wrapper.name,wrapper)]
archive=r/'research_v2/incoming/code_f41be1f.tar';freeze=dict(code_commit=c,release=str(src),code_archive=dict(remote_path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)),wrapper=dict(remote_path=str(wrapper),bytes=wrapper.stat().st_size,sha256=sha(wrapper)),validation_status_sha256=sha(base/'tests.status.json'),no_training_output_read=True,no_model_or_checkpoint_download=True)
index=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for local,p in files:
  assert p.suffix not in ('.pt','.bin','.npz','.safetensors')
  index.append(dict(local_path=local,remote_path=str(p),sha256=sha(p),bytes=p.stat().st_size));tar.add(p,arcname=local,recursive=False)
 for name,value in [('FREEZE_METADATA.json',freeze),('REMOTE_ARTIFACT_INDEX.json',dict(source_commit=c,files=index,model_calls=0,training_files_opened=0))]:
  data=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(data);tar.addfile(info,io.BytesIO(data))
'''
OUT.mkdir(parents=True,exist_ok=False);archive=ROOT/'.bootstrap/qwen_continuation_validation_f41be1f.tar'
with archive.open('wb') as stream:
 r=subprocess.run(['ssh','wzy3090','taskset','-c','1','python3','-'],input=REMOTE.encode(),stdout=stream,stderr=subprocess.PIPE)
assert r.returncode==0,r.stderr.decode()
with tarfile.open(archive) as tar:
 for item in tar.getmembers():
  assert item.isfile() and not Path(item.name).is_absolute() and '..' not in Path(item.name).parts
  path=OUT/item.name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(tar.extractfile(item).read())
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
idx=json.loads((OUT/'REMOTE_ARTIFACT_INDEX.json').read_text());freeze=json.loads((OUT/'FREEZE_METADATA.json').read_text())
for row in idx['files']:assert H(OUT/row['local_path'])==row['sha256']
assert H(ROOT/'.bootstrap/code_f41be1f.tar')==freeze['code_archive']['sha256']
assert H(ROOT/'.bootstrap/qwen_continuation_f41be1f.sh')==freeze['wrapper']['sha256']
with tarfile.open(ROOT/'.bootstrap/code_f41be1f.tar') as tar:
 members={m.name:m for m in tar.getmembers() if m.isfile()}
 checks=[]
 for row in idx['files']:
  if not row['local_path'].startswith('source/'):continue
  name=row['local_path'][7:];m=members.get(name)
  assert m is not None,name
  assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==row['sha256'],name
  checks.append(name)
(OUT/'SYNC_RECEIPT.json').write_text(json.dumps(dict(archive_bytes=archive.stat().st_size,archive_sha256=H(archive),downloaded_files=len(idx['files']),all_remote_bytes_verified=True,source_files_compared_to_actual_deploy_archive=checks,code_archive_local_remote_equal=True,wrapper_local_remote_equal=True,model_calls=0,weights_downloaded=0,training_output_read=False),indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(files=len(idx['files']),sources=len(checks),archive_sha256=H(archive))))
