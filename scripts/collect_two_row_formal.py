"""Formal narrow-ID collection with two fixed shards and mechanical closures.

Each parent reuses the verified canonical physical worker. Resume continues
corpus progress; it never replays an interrupted parent's already-issued slots.
Locked raw observations/outcomes are not read by this coordinator.
"""
import argparse
from contextlib import contextmanager
import copy
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from scripts import collect_two_row_canonical_repair as physical
from scripts import register_two_row_formal as registration
batch=physical.batch;HERE=Path(__file__).resolve().parent


def atomic_write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    batch.write(temporary,value);os.replace(str(temporary),str(path))


def load_registration(path,expected_sha=None):
    if expected_sha and batch.digest(path)!=expected_sha:raise ValueError('registration SHA changed')
    value=json.loads(path.read_text())
    if value['protocol']!=registration.PROTOCOL or value['requested_parents']!=116 or value['requested_routes']!=3132:
        raise ValueError('wrong formal registration')
    if len(value['parent_plan'])!=116:raise ValueError('parent registration incomplete')
    for index,p in enumerate(value['parent_plan']):
        if p['index']!=index or p['seed']!=283200+index or p['parent_id']!='two_row_reach_%d'%(283200+index):raise ValueError('parent identity mismatch')
        if p['role']!=registration.role_for_index(index) or p['config']['split']!=p['role']:raise ValueError('parent split changed')
    return value


def source_hashes(path):
    paths=[Path(__file__),Path(physical.__file__),Path(physical.pilot.__file__),Path(physical.legacy.__file__),
           Path(physical.legacy.native_snapshot.__code__.co_filename),Path(physical.endpoint.__file__),
           Path(physical.static.__file__),Path(batch.__file__),Path(registration.__file__),HERE/'record_job.py']
    return dict({p.name:batch.digest(p) for p in paths},registration_bytes=batch.digest(path))


def prepare(path,output):
    value=load_registration(path)
    base=json.loads((HERE.parent/'configs/observed_two_row_pilot_v4.json').read_text())
    regenerated=registration.build_registration(value['specification'],base,value['excluded_hashes'])
    for key in ('parent_plan','execution_indices','execution_order','shards','canonical_init','duplicate_gate'):
        if value[key]!=regenerated[key]:raise ValueError('registration does not match its one-shot sampler: '+key)
    frozen=dict(protocol='two_row_formal116_corpus_v1',registration_sha256=batch.digest(path),source_sha256=source_hashes(path),
        registration_file='registration.json',requested_parents=116,requested_routes=3132,
        locked_policy='Only mechanical receipts, slot ledger, hashes and PID/exit are inspected by coordinator; raw outcomes stay sealed.')
    output.mkdir(parents=True,exist_ok=False)
    (output/'registration.json').write_bytes(path.read_bytes())
    atomic_write(output/'corpus_manifest.json',frozen)
    return frozen


def verify_corpus(output):
    manifest=json.loads((output/'corpus_manifest.json').read_text());path=output/'registration.json'
    value=load_registration(path,manifest['registration_sha256'])
    if source_hashes(path)!=manifest['source_sha256']:raise ValueError('corpus resume source differs from frozen collector')
    return value,manifest


def worker(output,index,data):
    value,manifest=verify_corpus(output);plan=copy.deepcopy(value['parent_plan'][index])
    if not plan['collection_allowed']:raise ValueError('registered parent is closed to collection')
    init=copy.deepcopy(value['canonical_init'])
    config=dict(init,protocol='two_row_formal116_parent_v1',role=plan['role'],requested_layout_audits=1,
        route_parent_seed=plan['seed'],requested_route_proposals=27,registered_parent=plan,
        corpus_registration_sha256=manifest['registration_sha256'],requested_setup_actions=0)
    config_path=output/'parent_configs'/('%03d.json'%index)
    if not config_path.exists():raise ValueError('coordinator must freeze parent configuration before worker')
    if json.loads(config_path.read_text())!=config:raise ValueError('parent configuration differs from registration')
    def sources(cfg):
        from rlbench import const
        if batch.digest(const.__file__)!=value['specification']['color_source_sha256']:raise ValueError('actual official color table source changed')
        anchor=physical.checked_json(Path(init['source_v4'])/init['anchor_file'],init['anchor_sha256'])
        q=np.asarray(anchor['state']['_robot']['arm_joints']);g=np.asarray(anchor['state']['_robot']['gripper_joints'])
        if not np.array_equal(q,init['canonical_arm_joints']) or not np.array_equal(g,init['canonical_gripper_joints']):raise ValueError('canonical source joints changed')
        # New registered fixtures, not a fabricated old world snapshot.
        plan['saved_world']=dict(state={name:dict(pose=list(xyz)+[0.,0.,0.,1.],color=color['rgb'])
            for name,xyz,color in zip(('target','distractor0','distractor1'),plan['config']['goal_xyz'],plan['target_colors'])})
        return [plan],q,g
    return physical.run_collection(config,config_path,data,sources)


def parent_config(value,plan,manifest):
    return dict(copy.deepcopy(value['canonical_init']),protocol='two_row_formal116_parent_v1',role=plan['role'],
        requested_layout_audits=1,route_parent_seed=plan['seed'],requested_route_proposals=27,
        registered_parent=plan,corpus_registration_sha256=manifest['registration_sha256'],requested_setup_actions=0)


def pid_alive(pid):
    if not pid:return False
    if os.name=='nt':
        # Windows os.kill(pid, 0) is not a portable existence probe.
        import ctypes
        kernel=ctypes.windll.kernel32
        kernel.OpenProcess.restype=ctypes.c_void_p
        handle=kernel.OpenProcess(0x1000,False,int(pid))
        if not handle:return False
        try:
            code=ctypes.c_ulong()
            if not kernel.GetExitCodeProcess(ctypes.c_void_p(handle),ctypes.byref(code)):raise OSError('process status unreadable')
            return code.value==259
        finally:kernel.CloseHandle(ctypes.c_void_p(handle))
    try:os.kill(int(pid),0)
    except ProcessLookupError:return False
    return True


@contextmanager
def shard_lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        old=json.loads(path.read_text())
        if pid_alive(old.get('pid')):raise RuntimeError('shard coordinator still alive')
        path.rename(path.with_name(path.name+'.stale.'+str(time.time_ns())))
    fd=os.open(str(path),os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(fd,'w') as stream:json.dump(dict(pid=os.getpid()),stream)
    try:yield
    finally:
        if path.exists() and json.loads(path.read_text()).get('pid')==os.getpid():path.unlink()


def slot_counts(path,parent_id):
    started=set();completed=set()
    if not path.exists():return 0,0,False
    text=path.read_text();partial=bool(text and not text.endswith('\n'))
    lines=text.splitlines()
    if partial:lines=lines[:-1]
    for line in lines:
        r=json.loads(line)
        if set(r)!={'event','parent_id','input_id','attempt'} or r['parent_id']!=parent_id:raise ValueError('invalid mechanical slot ledger')
        if r['input_id'] not in [parent_id+'_target%d'%i for i in range(3)] or not 0<=r['attempt']<9:raise ValueError('invalid target/slot')
        key=(r['input_id'],r['attempt'])
        if r['event']=='started':
            if key in started:raise ValueError('duplicate started route slot')
            started.add(key)
        elif r['event']=='completed':
            if key not in started or key in completed:raise ValueError('invalid completed route slot')
            completed.add(key)
        else:raise ValueError('invalid ledger event')
    if len(started)-len(completed)>1:raise ValueError('more than one unclosed sequential slot')
    return len(started),len(completed),partial


def mechanical_closure(data,plan,status,elapsed,worker_code):
    started,completed,partial=slot_counts(data/'slot_ledger.jsonl',plan['parent_id'])
    receipt_path=data/'mechanical_receipt.json';receipt=json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    if receipt:
        if receipt['role']!=plan['role'] or receipt['started_slots']!=started or receipt['completed_slots']!=completed:
            raise ValueError('mechanical receipt and ledger differ')
    known_complete=receipt is not None and not partial
    upper=started if known_complete else min(27,started+int(partial))
    # A slot's started record is emitted before its restore/planning action.
    closed_status='worker_completed' if known_complete else 'interrupted_without_replay'
    if worker_code not in (None,0):closed_status='worker_runtime_failed'
    if not plan['collection_allowed']:closed_status='registration_gate_closed';upper=0
    layouts=(receipt or {}).get('layouts',[])
    actual=layouts[0].get('actual_geometry_1mm_sha256') if layouts else None
    files={name:batch.digest(data/name) for name in ('mechanical_receipt.json','slot_ledger.jsonl','artifact_hashes.json') if (data/name).exists()}
    initial_saved=any(r.get('initial_observation_saved') for r in layouts)
    actual_match=actual==plan['registered_geometry_1mm_sha256'] if actual is not None else None
    return dict(parent_id=plan['parent_id'],index=plan['index'],role=plan['role'],status=closed_status,
        requested_routes=27,attempted_lower=started,attempted_upper=upper,completed_slots=completed,
        unattempted_lower=27-upper,unattempted_upper=27-started,worker_exit_code=worker_code,
        worker_elapsed_seconds=elapsed,worker_elapsed_unknown=elapsed is None,
        worker_elapsed_upper_at_closure_seconds=(time.time()-datetime.datetime.fromisoformat(status['start_utc']).timestamp())
            if elapsed is None and status and status.get('start_utc') else None,
        initial_observation_saved=initial_saved,actual_matches_registered=actual_match,
        model_eligible=bool(plan['registration_eligible'] and initial_saved and actual_match),
        registered_geometry_1mm_sha256=plan['registered_geometry_1mm_sha256'],actual_geometry_1mm_sha256=actual,
        registration_eligible=plan['registration_eligible'],mechanical_files_sha256=files,
        no_route_success_metrics=True)


def validate_closure(output,row,plans):
    if row['parent_id'] not in plans:raise ValueError('unregistered closure parent')
    plan=plans[row['parent_id']]
    if row['index']!=plan['index'] or row['role']!=plan['role']:raise ValueError('closure index/role mismatch')
    data=output/'parents'/plan['role']/plan['parent_id']
    allowed={'mechanical_receipt.json','slot_ledger.jsonl','artifact_hashes.json'}
    if not set(row['mechanical_files_sha256'])<=allowed:raise ValueError('closure attempts to read nonmechanical files')
    for filename,digest in row['mechanical_files_sha256'].items():
        if batch.digest(data/filename)!=digest:raise ValueError('closed mechanical evidence changed')
    receipt=data/'mechanical_receipt.json'
    if receipt.exists():
        value=json.loads(receipt.read_text())
        if value['role']!=row['role']:raise ValueError('receipt role differs from registered parent')
        layouts=value.get('layouts',[])
        actual=layouts[0].get('actual_geometry_1mm_sha256') if layouts else None
        if actual!=row.get('actual_geometry_1mm_sha256'):raise ValueError('closure layout differs from hashed receipt')


def live_layout_gate(output):
    """Only registered/mechanical hashes; safe across roles including locked."""
    value,_=verify_corpus(output);plans={p['parent_id']:p for p in value['parent_plan']}
    rows=[json.loads(p.read_text()) for p in sorted((output/'closures').glob('*.json'))]
    groups={};blocked=set(value['duplicate_gate']['blocked_parent_ids'])
    old={r['geometry_1mm_sha256'] for r in value['excluded_hashes']}
    for row in rows:
        validate_closure(output,row,plans)
        h=row.get('actual_geometry_1mm_sha256')
        if h:
            groups.setdefault(h,[]).append(row['parent_id'])
            if h in old or h!=plans[row['parent_id']]['registered_geometry_1mm_sha256']:blocked.add(row['parent_id'])
    duplicates=[]
    for h,ids in groups.items():
        if len(ids)>1:
            duplicates.append(dict(sha256=h,parent_ids=ids,roles=sorted({plans[p]['role'] for p in ids})))
            blocked.update(ids)
    return dict(blocked_parent_ids=sorted(blocked),duplicate_groups=duplicates,
        closed_parents=len(rows),unavailable_initial_parent_ids=[r['parent_id'] for r in rows if not r['initial_observation_saved']],
        mechanical_only=True,raw_locked_opened=False)


def prior_phase_indices(index):
    if index<16:return []
    if 64<=index<76:return list(range(16))
    if index<64:return list(range(16))+list(range(64,76))
    return list(range(76))


def wait_phase(output,index,shard):
    required=prior_phase_indices(index);started=time.monotonic()
    while any(not (output/'closures'/('%03d.json'%i)).exists() for i in required):
        atomic_write(output/'shards'/str(shard)/'barrier.json',dict(waiting_for_phase_before_index=index,
            missing_closed_indices=[i for i in required if not (output/'closures'/('%03d.json'%i)).exists()],
            mechanical_only=True))
        other=output/'shards'/str(1-shard)/'coordinator.lock'
        if time.monotonic()-started>30 and (not other.exists() or not pid_alive(json.loads(other.read_text()).get('pid'))):
            raise RuntimeError('phase barrier cannot advance: peer shard is not running; resume after resolving its status')
        time.sleep(1)


def run_shard(output,run_root,shard,sim_python,resume=False,max_new_parents=None):
    value,manifest=verify_corpus(output)
    if shard not in (0,1):raise ValueError('exactly two registered shards')
    state_dir=output/'shards'/str(shard)
    if state_dir.exists() and not resume:raise ValueError('existing shard requires --resume')
    run_root.mkdir(parents=True,exist_ok=True);new_count=0
    plans={p['parent_id']:p for p in value['parent_plan']}
    with shard_lock(state_dir/'coordinator.lock'):
        for parent_id in value['shards'][shard]:
            plan=plans[parent_id];index=plan['index'];data=output/'parents'/plan['role']/parent_id
            closure_path=output/'closures'/('%03d.json'%index);status_path=run_root/('parent_%03d.status.json'%index)
            if closure_path.exists():
                closed=json.loads(closure_path.read_text())
                validate_closure(output,closed,plans)
                continue
            if max_new_parents is not None and new_count>=max_new_parents:break
            wait_phase(output,index,shard)
            status=json.loads(status_path.read_text()) if status_path.exists() else {}
            if status.get('status') in ('starting','running') and any(pid_alive(status.get(k)) for k in ('pid','child_pid')):
                raise RuntimeError('parent worker still alive; resume refused')
            elapsed=None;code=status.get('exit_code')
            if status.get('start_utc') and status.get('end_utc'):
                elapsed=(datetime.datetime.fromisoformat(status['end_utc'])-datetime.datetime.fromisoformat(status['start_utc'])).total_seconds()
            cfg=parent_config(value,plan,manifest);cfgpath=output/'parent_configs'/('%03d.json'%index)
            if cfgpath.exists():
                if json.loads(cfgpath.read_text())!=cfg:raise ValueError('frozen parent config changed')
            else:atomic_write(cfgpath,cfg)
            if plan['collection_allowed'] and not data.exists() and not status_path.exists():
                data.parent.mkdir(parents=True,exist_ok=True);clock=time.perf_counter()
                command=[sys.executable,str(HERE/'record_job.py'),'--output',str(run_root),'--run-id','parent_%03d'%index,
                    '--resume-strategy','none','--',str(sim_python),str(Path(__file__)),'worker',
                    '--corpus',str(output),'--parent-index',str(index),'--output',str(data)]
                code=subprocess.run(command,check=False).returncode;elapsed=time.perf_counter()-clock
                status=json.loads(status_path.read_text()) if status_path.exists() else {}
            closed=mechanical_closure(data,plan,status,elapsed,code);atomic_write(closure_path,closed);new_count+=1
            closures=[json.loads((output/'closures'/('%03d.json'%plans[p]['index'])).read_text())
                for p in value['shards'][shard] if (output/'closures'/('%03d.json'%plans[p]['index'])).exists()]
            atomic_write(state_dir/'progress.json',dict(shard=shard,closed_parents=len(closures),requested_parents=58,
                closed_by_role={role:sum(r['role']==role for r in closures) for role,_,_ in registration.ROLE_RANGES},
                finalized_worker_seconds=sum(r['worker_elapsed_seconds'] or 0 for r in closures),
                unknown_cost_parents=sum(r['worker_elapsed_unknown'] for r in closures),
                status='complete' if len(closures)==58 else 'running',only_mechanical_fields=True))
            print(json.dumps(dict(parent_id=parent_id,role=plan['role'],closure='written',shard=shard)),flush=True)
            if code not in (None,0):raise RuntimeError('worker runtime failed; this shard stops after preserving its closure')
    return new_count


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--registration',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('worker');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--parent-index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('shard');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--shard',type=int,choices=(0,1),required=True)
    p.add_argument('--sim-python',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--max-new-parents',type=int)
    args=parser.parse_args(argv)
    if args.action=='prepare':prepare(args.registration,args.output)
    elif args.action=='worker':
        if not 0<=args.parent_index<116:raise ValueError('parent index outside registration')
        result=worker(args.corpus,args.parent_index,args.output)
        return 1 if result.get('fatal_error') or result.get('shutdown_error') else 0
    else:run_shard(args.corpus,args.run_root,args.shard,args.sim_python,args.resume,args.max_new_parents)
    return 0


if __name__=='__main__':sys.exit(main())
