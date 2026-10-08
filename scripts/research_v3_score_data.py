"""New paired-layout score-role data, registered before any observations exist."""
import argparse
import copy
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from scripts import paired_modes_data as base
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha,lines
from scripts.research_v3_frequency import RUN

DATA=ROOT/'data/research_v3_paired_score_v1'
POLICY=SOURCE/'configs/research_v3_paired_score_data.json'


def registration():
    original=base.POLICY
    try:
        base.POLICY=POLICY;value=base.registration()
    finally:base.POLICY=original
    policy=read(POLICY);roles=[r for r,n in policy['role_families'].items() for _ in range(n)]
    assert len(roles)==64
    for p in value['parent_plan']:
        family=p['index']//3;variant=p['index']%3
        p['family_id']=f"paired_family_{policy['family_base']+family:06d}"
        p['parent_id']=p['family_id']+'_'+p['variant']
        p['role']=p['split']=p['config']['split']=roles[family]
        p['seed']=p['config']['seed']=policy['family_base']+3*family+variant
    value['protocol']='research_v3_paired_score_data_v1'
    return value


def prepare():
    value=registration();old=read(base.DATA/'registration.json')
    hashes={p['registered_geometry_1mm_sha256'] for p in value['parent_plan']}
    assert len(hashes)==192 and not hashes&{p['registered_geometry_1mm_sha256'] for p in old['parent_plan']}
    DATA.mkdir(parents=True,exist_ok=False)
    write(DATA/'registration.json',value)
    write(DATA/'source_sha256.json',{str(p.relative_to(SOURCE)):sha(p)
        for directory in ('scripts','routeset','configs') for p in (SOURCE/directory).rglob('*')
        if p.is_file() and p.suffix in ('.py','.json','.sh')})
    write(DATA/'parent_isolation.json',dict(existing_paired_registration_sha256=sha(base.DATA/'registration.json'),
        new_family_count=64,new_parent_count=192,geometry_hash_intersection=0,TEST_LOCKED_access=False))


def worker(index):
    base.DATA=DATA;base.RUN=RUN/'paired_score_collection_v1';base.worker(index)


def collect(start,stop):
    assert 0<=start<stop<=192
    value=read(DATA/'registration.json');work=RUN/f'paired_score_collection_{start:03d}_{stop:03d}'
    work.mkdir(parents=True,exist_ok=False)
    def one(plan):
        i=plan['index'];out=DATA/'parents'/plan['role']/plan['parent_id']
        if out.exists():raise FileExistsError(str(out))
        if sum(p.stat().st_size for p in DATA.rglob('*') if p.is_file())>read(POLICY)['maximum_new_bytes']:
            raise RuntimeError('New data disk ceiling reached')
        cmd=[str(ROOT/'.venv-sim/bin/python'),'-m','scripts.research_v3_score_data','worker','--index',str(i)]
        tic=time.monotonic();receipt=dict(command=cmd,status='running',index=i,start_unix=time.time())
        with (work/f'{i:03d}.log').open('w') as log:
            child=subprocess.Popen(cmd,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT)
            receipt['pid']=child.pid;write(work/f'{i:03d}.json',receipt);code=child.wait()
        receipt.update(exit_code=code,elapsed_seconds=time.monotonic()-tic,status='completed' if code==0 else 'failed')
        if (out/'summary.json').exists():
            summary=read(out/'summary.json');receipt['initialization']=summary['initialization']['passed']
        write(work/f'{i:03d}.json',receipt);print(receipt,flush=True)
        return receipt
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(one,value['parent_plan'][start:stop]))
    passed=all(r['exit_code']==0 and r.get('initialization') for r in results)
    write(work/'summary.json',dict(results=results,passed=passed))
    if not passed:raise RuntimeError('Collection has unresolved failures; no export')


def export():
    value=read(DATA/'registration.json')
    for role in read(POLICY)['role_families']:
        out=DATA/'exports'/role;out.mkdir(parents=True,exist_ok=False);obs=[];labels=[]
        for p in value['parent_plan']:
            if p['role']!=role:continue
            f=DATA/'parents'/role/p['parent_id'];s=read(f/'summary.json')
            assert s['status']!='error' and s['initialization']['passed']
            for r in lines(f/'observations.jsonl'):
                assert r['split']==role;r['image']=str(f/r['image']);obs.append(r)
            for r in lines(f/'supervision.jsonl'):
                assert r['split']==role
                for k in ('observation','verification_only','route_config'):r[k]=str(f/r[k])
                r['routes']=[str(f/name) for name in r['routes']];labels.append(r)
        assert len(obs)==read(POLICY)['role_families'][role]*9==len(labels)
        import json
        for name,rows in [('observations',obs),('supervision',labels)]:
            (out/(name+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
        write(out/'manifest.json',dict(role=role,registration_sha256=sha(DATA/'registration.json'),
            requests=len(obs),source_commit=os.environ.get('CODE_COMMIT'),
            output_sha256={n:sha(out/n) for n in ('observations.jsonl','supervision.jsonl')}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','worker','collect','export'])
    p.add_argument('--index',type=int);p.add_argument('--start',type=int);p.add_argument('--stop',type=int);a=p.parse_args()
    if a.stage=='prepare':prepare()
    elif a.stage=='worker':worker(a.index)
    elif a.stage=='collect':collect(a.start,a.stop)
    else:export()
