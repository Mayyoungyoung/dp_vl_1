from pathlib import Path
import json,hashlib,subprocess,base64
R=next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file());P=R/'reports/observed_two_row_extension64_quality_v1';O=R/'reports/observed_two_row_extension288_quality_v1'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
old=json.loads((O/'PARENT_JOB_ARCHIVE_INDEX.json').read_text())['files']; rows=[]
for x in old:
 src=O/x['path']; dst=P/x['path']; assert H(src)==x['sha256']; dst.parent.mkdir(exist_ok=True); dst.write_bytes(src.read_bytes()); rows.append(dict(x,archive_method='copied_verified_old32_archive'))
code="""from pathlib import Path
import json,hashlib,base64
root=Path('/home/wzy/dpvlm/route_set_v1/runs/observed_two_row_extension288_v1/parents')
rows=[]
for i in range(32,64):
 for suffix in ('.status.json','.log'):
  p=root/('shard'+str(i%2))/('parent_%03d'%i+suffix);b=p.read_bytes()
  if suffix=='.status.json':
   v=json.loads(b);assert v['exit_code']==0,(p,v)
  rows.append(dict(path='parent_jobs/'+p.name,server_path=str(p),sha256=hashlib.sha256(b).hexdigest(),bytes=len(b),base64=base64.b64encode(b).decode()))
print(json.dumps(rows))
"""
r=subprocess.run(['ssh','wzy3090','taskset','-c','1','python3','-'],input=code.encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True)
for x in json.loads(r.stdout):
 b=base64.b64decode(x.pop('base64'));assert hashlib.sha256(b).hexdigest()==x['sha256'];(P/x['path']).write_bytes(b);rows.append(dict(x,archive_method='readonly_completed_new32_server'))
assert len(rows)==128
(P/'PARENT_JOB_ARCHIVE_INDEX.json').write_text(json.dumps(dict(files=rows,files_count=len(rows),bytes=sum(x['bytes'] for x in rows),old32_copied=64,new32_downloaded=64,scope='TRAIN indices 0..63 only; missing local logs were an archive omission, not missing original data'),indent=2)+'\n',encoding='utf-8')
idx=json.loads((P/'SERVER_ARCHIVE_INDEX.json').read_text()); print('serverindexkeys',list(idx));print('parent files',len(rows),'bytes',sum(x['bytes'] for x in rows));print('status',json.loads((P/'parent_jobs/parent_063.status.json').read_text()))
