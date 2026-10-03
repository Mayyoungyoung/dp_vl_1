"""Prepared archive only. Run after root confirms actual TRAIN128 quality exit0.

No simulation, planner, Qwen/model call, or subsequent collection is launched.
The remote guard precedes all quality-content reads. All fetched bytes are
indexed, including the large JSON retained locally under an exact Git ignore.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
DEST = ROOT/'reports/observed_two_row_extension128_quality_v1'
OLD = ROOT/'reports/observed_two_row_extension64_quality_v1'
SOURCE = '1a3eef1fb12d55e98d4d188a091ea40ea62c0a02'
COLLECTOR = '5c8f8e4f5cd478c793a0e0d9640005deaf700973'
WRAPPER_SHA = '24c8af0f0e950c16c9ce3105b0ff50a18e25d9f59218d15e8bb888afa7921ad0'

REMOTE = r'''
from pathlib import Path
import datetime,hashlib,io,json,os,sys,tarfile
P=Path('/home/wzy/dpvlm/route_set_v1')
run=P/'runs/observed_two_row_extension288_quality_v1'
status=json.loads((run/'train128.status.json').read_text())
assert status['status']=='completed' and status['exit_code']==0,'quality must complete exit0 before raw results are read'
session=P/'runs/observed_two_row_extension288_v1/sessions/20261003T011626Z_train128_675690'
assert (session/'status').read_text().strip()=='completed' and (session/'exit_code').read_text().strip()=='0'
assert sorted(os.sched_getaffinity(0))==[0]
source=P/'research_v2/releases/1a3eef1fb12d55e98d4d188a091ea40ea62c0a02'
collector=P/'research_v2/releases/5c8f8e4f5cd478c793a0e0d9640005deaf700973'
wrapper=P/'research_v2/incoming/extension128_quality_1a3eef1.sh'
assert hashlib.sha256(wrapper.read_bytes()).hexdigest()=='24c8af0f0e950c16c9ce3105b0ff50a18e25d9f59218d15e8bb888afa7921ad0'
analysis=json.loads((run/'train128/analysis.json').read_text())
assert analysis['requested_parents']==128 and analysis['requested_slots']==3456 and analysis['requested_conditions']==384
assert analysis['selected_train_indices']==list(range(128)) and analysis['new_dev_raw_sealed'] is True
assert analysis['dev_or_higher_raw_opened'] is False
assert analysis['analysis_source_sha256']==hashlib.sha256((source/'scripts/analyze_two_row_extension_train.py').read_bytes()).hexdigest()
files={}
def add(path,name):
 if path.is_symlink():raise ValueError('no symlink archive inputs: '+str(path))
 if path.is_file():
  if name in files:raise ValueError('duplicate archive name: '+name)
  files[name]=path
 elif path.is_dir():
  for f in sorted(path.rglob('*')):
   if f.is_symlink():raise ValueError('no symlink archive inputs')
   if f.is_file():add(f,name+'/'+f.relative_to(path).as_posix())
 else:raise FileNotFoundError(str(path))
add(run/'train128','analysis_run/train128')
for name in ('train128.status.json','train128.log','registry.jsonl'):add(run/name,'analysis_run/'+name)
add(session,'collection_session')
corpus=P/'data/observed_two_row_extension288_v1'
for name in ('corpus_manifest.json','registration.json'):add(corpus/name,'corpus/'+name)
manifest=json.loads((corpus/'corpus_manifest.json').read_text())
assert manifest['registration_sha256']==hashlib.sha256((corpus/'registration.json').read_bytes()).hexdigest()
parent_metadata=('mechanical_receipt.json','slot_ledger.jsonl','artifact_hashes.json','observations.jsonl',
 'supervision.jsonl','summary.json','manifest.json','layout_audits.jsonl','model_assets.json','selected_snapshot_return.json')
for i in range(128):
 identity='two_row_reach_'+str(400000+i)
 add(corpus/'closures'/('%03d.json'%i),'corpus/closures/%03d.json'%i)
 for name in parent_metadata:add(corpus/'parents/TRAIN'/identity/name,'corpus/parents/'+identity+'/'+name)
 jobs=P/'runs/observed_two_row_extension288_v1/parents'/('shard'+str(i%2))
 job_status=json.loads((jobs/('parent_%03d.status.json'%i)).read_text())
 assert job_status['status']=='completed' and job_status['exit_code']==0
 for suffix in ('.status.json','.log'):add(jobs/('parent_%03d'%i+suffix),'parent_jobs/parent_%03d'%i+suffix)
for name in ('analyze_two_row_extension_train.py','analyze_two_row_formal_train.py',
 'analyze_observed_two_row_pilot.py','collect_two_row_extension.py','record_job.py'):
 add(source/'scripts'/name,'source/analysis/'+name+'.txt')
for name,expected in manifest['source_sha256'].items():
 if name=='registration_bytes':continue
 assert Path(name).name==name,'collector source names must be basename-only'
 path=collector/('configs' if name.endswith('.json') else 'scripts')/name
 assert hashlib.sha256(path.read_bytes()).hexdigest()==expected,'frozen collector source differs: '+name
 add(path,'source/collector/'+name+'.txt')
add(wrapper,'source/extension128_quality_1a3eef1.sh')
index=[]
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as archive:
 for name,path in sorted(files.items()):
  data=path.read_bytes()
  index.append(dict(path=name,server_path=str(path),bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
  info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
 data=(json.dumps(dict(protocol='extension128_readonly_archive_v1',captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
  analysis_source=source.name,collector_source=collector.name,selected_train_indices=list(range(128)),
  cpu_affinity=sorted(os.sched_getaffinity(0)),new_dev_raw_opened=False,old_reserved_raw_opened=False,
  trajectory_npz_downloaded=False,checkpoint_downloaded=False,files=index),indent=2)+'\n').encode()
 info=tarfile.TarInfo('SERVER_ARCHIVE_INDEX.json');info.size=len(data);archive.addfile(info,io.BytesIO(data))
'''


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root-confirmed-quality-exit0',action='store_true',required=True)
    args=parser.parse_args()
    if DEST.exists():raise FileExistsError('fresh archive required; do not overwrite partial evidence')
    large=DEST/'analysis_run/train128/all_requested_slots.json'
    ignored=subprocess.run(['git','check-ignore','--no-index','-q',str(large)],cwd=ROOT)
    if ignored.returncode!=0:raise ValueError('root must add the exact large-JSON ignore rule before archive')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    tar=ROOT/'.bootstrap'/('extension128_quality_archive_'+stamp+'.tar.gz')
    if tar.exists():raise FileExistsError(str(tar))
    command=['ssh','wzy3090',"CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 0 python3 -"]
    with tar.open('xb') as handle:
        result=subprocess.run(command,input=REMOTE.encode(),stdout=handle,stderr=subprocess.PIPE)
    if result.returncode:
        failure=tar.with_suffix('.failure.json')
        failure.write_text(json.dumps(dict(exit_code=result.returncode,stderr=result.stderr.decode(errors='replace'),
            preserved_archive=str(tar),raw_result_archive_completed=False),indent=2)+'\n',encoding='utf-8')
        raise RuntimeError('archive failed; receipt '+str(failure))
    DEST.mkdir(parents=True)
    seen=set()
    with tarfile.open(tar,'r:gz') as archive:
        for member in archive:
            if not member.isfile() or member.name in seen:raise ValueError('non-file or duplicate archive member')
            target=DEST/member.name;target.resolve().relative_to(DEST.resolve());seen.add(member.name)
            target.parent.mkdir(parents=True,exist_ok=True)
            with archive.extractfile(member) as source,target.open('xb') as out:
                for chunk in iter(lambda:source.read(1024*1024),b''):out.write(chunk)
    index=json.loads((DEST/'SERVER_ARCHIVE_INDEX.json').read_text())
    for row in index['files']:
        path=DEST/row['path']
        if path.stat().st_size!=row['bytes'] or sha(path)!=row['sha256']:raise ValueError('archive hash mismatch: '+row['path'])
    old_checks=[]
    for i in range(64):
        parent='two_row_reach_'+str(400000+i)
        for name in ('front.png','target0_all9.png','target1_all9.png','target2_all9.png'):
            prior=OLD/'analysis_run/train64'/parent/name;current=DEST/'analysis_run/train128'/parent/name
            old_checks.append(dict(parent=parent,file=name,old_sha256=sha(prior),new_sha256=sha(current),
                byte_equal=prior.read_bytes()==current.read_bytes()))
    if len(old_checks)!=256 or not all(x['byte_equal'] for x in old_checks):
        (DEST/'OLD64_IMAGE_MISMATCH.json').write_text(json.dumps(old_checks,indent=2)+'\n')
        raise ValueError('old64 image bytes differ; do not automatically carry prior QA forward')
    large_artifacts=[]
    for row in index['files']:
        if row['bytes']>10*1024*1024:
            ignored=subprocess.run(['git','check-ignore','--no-index','-q',str(DEST/row['path'])],cwd=ROOT).returncode==0
            large_artifacts.append(dict(row,git_storage='ignored_large_json' if ignored else 'large_pending_root_ignore_rule'))
    receipt=dict(protocol='extension128_readonly_archive_v1',files=len(index['files']),
        bytes=sum(x['bytes'] for x in index['files']),archive_sha256=sha(tar),archive_path=str(tar),
        all_server_sha256_verified=True,parent_job_files=256,old64_images_byte_equal=old_checks,
        large_artifacts=large_artifacts,
        helper_sha256=sha(__file__),new_forward=0,new_search=0,new_simulation=0)
    (DEST/'SYNC_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:receipt[k] for k in ('files','bytes','archive_sha256','all_server_sha256_verified','parent_job_files')},indent=2))


if __name__=='__main__':main()
