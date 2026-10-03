"""Versioned numeric-identity repair; only original TRAIN parents 5 and 7.

No old source/data is rewritten. A scoped project-local validator delegates
physical collection to the unchanged variable-layout worker. This CLI has no
certification, extra-layout, model, or successful-parent replay stage.
"""
import argparse
import copy
from contextlib import contextmanager
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from scripts import collect_observed_layout_variation as legacy

registration=legacy.registration
PROTOCOL='observed_layout_variation_hash_recovery_v2'
CONFIG=ROOT/'configs/observed_layout_hash_recovery_v2.json'
INDICES=(5,7)
BASE=Path('/home/wzy/dpvlm/route_set_v1')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):legacy.old.atomic_write(Path(path),value)

def quantized_hash_v2(centers,halves,goals):
    """Only identity encoding changes; raw coordinates/physical gates do not."""
    c,h,g=(np.asarray(x,dtype=np.float64) for x in (centers,halves,goals))
    if c.ndim!=2 or c.shape[1:]!=(3,) or h.shape!=c.shape or g.shape!=(3,3):
        raise ValueError('Complete variable-N geometry required')
    order=np.lexsort((c[:,2],c[:,1],c[:,0]))
    a=np.r_[c[order].ravel(),h[order].ravel(),g.ravel()]
    if not np.isfinite(a).all() or np.any(np.abs(a)>1000):raise ValueError('Finite bounded physical coordinates required')
    if len(c)==4:a=np.r_[a,[0.,0.,0.]]
    um=np.rint(a*1_000_000).astype(np.int64)
    encoded=np.rint(um/1000).astype('<i8').tobytes()
    if len(c)!=4:encoded=registration.canonical(dict(schema='variable_axis_aligned_cuboids_v1',n=len(c)))+encoded
    return hashlib.sha256(encoded).hexdigest()

def validate_initial_geometry(plan,centers,halves,goals):
    cs,hs=registration.geometry(plan['config'])
    if (np.shape(centers)!=cs.shape or np.shape(halves)!=hs.shape or np.shape(goals)!=(3,3)
        or not np.allclose(centers,cs,atol=1e-6,rtol=0)
        or not np.allclose(halves,hs,atol=1e-6,rtol=0)
        or not np.allclose(goals,plan['config']['goal_xyz'],atol=1e-6,rtol=0)):
        raise ValueError('Actual complete variable-N geometry differs')
    digest=quantized_hash_v2(centers,halves,goals)
    if digest!=plan['registered_geometry_1mm_sha256']:raise ValueError('Versioned geometry identity mismatch')
    return digest

def policy():
    p=read(CONFIG)
    if (p['protocol']!=PROTOCOL or p['indices']!=list(INDICES) or p['role']!='TRAIN'
        or p['new_requested_slots']!=54 or p['total_worker_soft_cap_seconds']!=2700
        or p['original_source_commit']!='eba09945ecade4bdb9a5b4ab123a92da898283ed'
        or p['automatic_certify'] is not False or p['replay_successful_parents'] is not False):
        raise ValueError('Fixed recovery policy required')
    return p

def source_hashes():
    result=legacy.source_hashes()
    result.update({Path(__file__).relative_to(ROOT).as_posix():sha(__file__),CONFIG.relative_to(ROOT).as_posix():sha(CONFIG)})
    return result

def validate_original_metadata(value,closures,summaries,attempts,p):
    if sorted(x['index'] for x in closures)!=list(range(12)):
        raise ValueError('All original twelve mechanical closures required')
    if any(x['worker_elapsed_unknown'] or not isinstance(x['worker_elapsed_seconds'],(int,float))
           or not math.isfinite(x['worker_elapsed_seconds']) or x['worker_elapsed_seconds']<0 for x in closures):
        raise ValueError('Exact inherited worker cost required')
    spent=sum(x['worker_elapsed_seconds'] for x in closures)
    if spent!=p['inherited_worker_seconds']:raise ValueError('Inherited cumulative cost changed')
    for i in INDICES:
        plan=value['parent_plan'][i];s=summaries[i];rows=attempts[i]
        if plan['index']!=i or plan['role']!='TRAIN' or not plan['collection_allowed']:
            raise ValueError('Recovery parent identity/role differs')
        if (s['initialization']['passed'] or s['initialization'].get('error')!="ValueError('Actual geometry 1mm registration mismatch')"
            or s['route_attempts']!=0 or s['unattempted_route_proposals']!=27):
            raise ValueError('Only the two proven pre-attempt hash failures may recover')
        expected={(plan['parent_id']+'_target%d'%t,k) for t in range(3) for k in range(9)}
        if len(rows)!=27 or {(x['input_id'],x['attempt']) for x in rows}!=expected:
            raise ValueError('All original 27 slots required')
        if any(x['attempted'] or x['success'] or x['reason']!='initialization_gate_closed' for x in rows):
            raise ValueError('An already attempted slot cannot be replayed')
    return spent

def verify_original(p):
    path=Path(p['original_corpus'])
    if path.is_symlink():raise ValueError('Original corpus symlink forbidden')
    for rel,digest in p['original_files_sha256'].items():
        f=path/rel
        if f.is_symlink() or sha(f)!=digest:raise ValueError('Original sealed evidence changed: '+rel)
    value,manifest=legacy.verify_corpus(path)
    closures=legacy.checked_closures(path,value)
    summaries={};attempts={}
    for i in INDICES:
        parent=path/'parents'/'TRAIN'/value['parent_plan'][i]['parent_id']
        summaries[i]=read(parent/'summary.json')
        attempts[i]=[json.loads(line) for line in (parent/'attempts.jsonl').read_text().splitlines() if line]
    spent=validate_original_metadata(value,closures,summaries,attempts,p)
    return value,spent

def recovered_registration(original,p,spent):
    value=copy.deepcopy(original)
    value.update(protocol=PROTOCOL,parent_plan=[copy.deepcopy(original['parent_plan'][i]) for i in INDICES],
        execution_order=list(INDICES),role_counts={'TRAIN':2},requested_parents=2,requested_routes=54,
        original_requested_routes=324,combined_historical_proposal_slots=378,
        recovery_lineage=dict(original_corpus=p['original_corpus'],original_source_commit=p['original_source_commit'],
            original_files_sha256=p['original_files_sha256'],inherited_worker_seconds=spent,
            total_worker_soft_cap_seconds=2700,remaining_worker_soft_seconds_at_start=2700-spent,
            physical_readback_tolerance_m=1e-6,identity_protocol='integer_micrometer_then_round_even_millimeter_v2',
            same_original_parent_ids=True,same_role='TRAIN',new_attempt_namespace=PROTOCOL))
    return value

def lineage_rows(plan):
    return [dict(parent_id=plan['parent_id'],role='TRAIN',input_id=plan['parent_id']+'_target%d'%t,slot=k,
                 original_unattempted_slot='v1:%s:target%d:slot%d'%(plan['parent_id'],t,k),
                 new_attempt_id=PROTOCOL+':%s:target%d:slot%d'%(plan['parent_id'],t,k)) for t in range(3) for k in range(9)]

def no_symlink_components(path):
    if any(x.is_symlink() for x in (path,*path.parents)):raise ValueError('Symlink path forbidden')

def prepare(output):
    p=policy();output=Path(output);no_symlink_components(output)
    if output!=Path(p['new_corpus']) or output==Path(p['original_corpus']):raise ValueError('Fixed distinct recovery root required')
    original,spent=verify_original(p);value=recovered_registration(original,p,spent)
    output.mkdir(parents=True,exist_ok=False)
    write(output/'registration.json',value)
    write(output/'corpus_manifest.json',dict(protocol=PROTOCOL,source_sha256=source_hashes(),policy_sha256=sha(CONFIG),
        registration_sha256=sha(output/'registration.json'),original_files_sha256=p['original_files_sha256'],
        requested_parents=2,requested_routes=54,all_roles='TRAIN',automatic_certify=False))
    for plan in value['parent_plan']:
        write(output/'lineage'/('%03d.json'%plan['index']),dict(parent_id=plan['parent_id'],role='TRAIN',slots=lineage_rows(plan)))
    return value

def verify_corpus(output):
    p=policy();output=Path(output);no_symlink_components(output)
    if output!=Path(p['new_corpus']):raise ValueError('Fixed recovery corpus required')
    manifest=read(output/'corpus_manifest.json')
    if manifest['protocol']!=PROTOCOL or manifest['source_sha256']!=source_hashes() or manifest['policy_sha256']!=sha(CONFIG):
        raise ValueError('Frozen recovery source/policy changed')
    if sha(output/'registration.json')!=manifest['registration_sha256']:raise ValueError('Recovery registration changed')
    original,spent=verify_original(p);value=read(output/'registration.json')
    if value!=recovered_registration(original,p,spent):raise ValueError('Recovery registration differs')
    for plan in value['parent_plan']:
        if read(output/'lineage'/('%03d.json'%plan['index']))!={'parent_id':plan['parent_id'],'role':'TRAIN','slots':lineage_rows(plan)}:
            raise ValueError('Distinct attempt lineage changed')
    return value,manifest

def checked_closures(output,value):
    rows=[];plans={x['parent_id']:x for x in value['parent_plan']}
    for path in sorted((output/'closures').glob('*.json')):
        row=read(path)
        if row['index'] not in INDICES or path.name!='%03d.json'%row['index']:raise ValueError('Only selected recovery indices may close')
        legacy.old.validate_closure(output,row,plans);rows.append(row)
    if len({x['index'] for x in rows})!=len(rows):raise ValueError('Duplicate closure')
    return rows

def cumulative_worker_seconds(inherited,rows):
    if any(x['worker_elapsed_unknown'] or x['worker_elapsed_seconds'] is None for x in rows):
        raise RuntimeError('Unknown interrupted cost; manual review required')
    values=[inherited]+[x['worker_elapsed_seconds'] for x in rows]
    if any(not math.isfinite(x) or x<0 for x in values):raise ValueError('Invalid cumulative cost')
    return sum(values)

@contextmanager
def scoped_validator(data):
    old_validator,old_protocol=legacy.validate_initial_geometry,legacy.PROTOCOL
    def validate(plan,c,h,g):
        # Original float arrays are used for both physical checks and collection.
        digest=validate_initial_geometry(plan,c,h,g)
        cs,hs=registration.geometry(plan['config'])
        write(data/'geometry_identity_audit.json',dict(protocol=PROTOCOL,
            original_registered_geometry_1mm_sha256=plan['registered_geometry_1mm_sha256'],
            legacy_actual_geometry_1mm_sha256=registration.physical_hash(c,h,g,.001),
            actual_geometry_1mm_v2_sha256=digest,actual_geometry_exact_sha256=registration.physical_hash(c,h,g),
            maximum_physical_readback_difference_m=float(max(np.abs(np.asarray(c)-cs).max(),np.abs(np.asarray(h)-hs).max(),np.abs(np.asarray(g)-plan['config']['goal_xyz']).max())),
            unchanged_physical_readback_atol_m=1e-6,actual_arrays_modified=False))
        return digest
    legacy.validate_initial_geometry,legacy.PROTOCOL=validate,PROTOCOL
    try:yield
    finally:legacy.validate_initial_geometry,legacy.PROTOCOL=old_validator,old_protocol

def worker(output,index,data):
    value,manifest=verify_corpus(output);plans={x['index']:x for x in value['parent_plan']}
    if index not in plans:raise ValueError('Only failed original parents5/7 are allowed')
    plan=plans[index]
    if data!=output/'parents'/'TRAIN'/plan['parent_id'] or data.exists():raise ValueError('Fresh selected parent path required')
    rows=checked_closures(output,value)
    if any(x['index']==index for x in rows):raise ValueError('Closed recovery parent cannot replay')
    if cumulative_worker_seconds(value['recovery_lineage']['inherited_worker_seconds'],rows)>=2700:
        raise legacy.bounded.InternalBudgetPause('Inherited cumulative45min soft cap exhausted')
    cfg=output/'parent_configs'/('%03d.json'%index)
    if read(cfg)!=plan:raise ValueError('Unchanged original parent configuration required')
    with scoped_validator(data):return legacy.physical_worker(value,manifest,plan,cfg,data)

def run_recovery(output,run_root,sim_python,resume=False,env=None):
    value,_=verify_corpus(output);p=policy();run_root=Path(run_root);no_symlink_components(run_root)
    if run_root!=Path(p['new_run_root']):raise ValueError('Fixed new run root required')
    stage=run_root/'recovery'
    if stage.exists() and not resume:raise ValueError('Existing stage requires explicit resume')
    stage.mkdir(parents=True,exist_ok=resume);jobs=run_root/'parents';jobs.mkdir(exist_ok=True)
    inherited=value['recovery_lineage']['inherited_worker_seconds']
    with legacy.old.shard_lock(run_root/'coordinator.lock'):
        for plan in value['parent_plan']:
            rows=checked_closures(output,value)
            if any(x['index']==plan['index'] for x in rows):continue
            i=plan['index'];data=output/'parents'/'TRAIN'/plan['parent_id'];sp=jobs/('parent_%03d.status.json'%i)
            status=read(sp) if sp.exists() else {};elapsed=None;code=status.get('exit_code')
            if status.get('status') in ('starting','running') and any(legacy.old.pid_alive(status.get(k)) for k in ('pid','child_pid')):
                raise RuntimeError('Original recovery worker still alive')
            if status.get('start_utc') and status.get('end_utc'):
                elapsed=(datetime.datetime.fromisoformat(status['end_utc'])-datetime.datetime.fromisoformat(status['start_utc'])).total_seconds()
            cfg=output/'parent_configs'/('%03d.json'%i)
            if cfg.exists():
                if read(cfg)!=plan:raise ValueError('Parent configuration changed')
            else:write(cfg,plan)
            if not data.exists() and not sp.exists():
                spent=cumulative_worker_seconds(inherited,rows)
                write(stage/'budget_before_parent.json',dict(next_index=i,inherited_seconds=inherited,cumulative_worker_seconds=spent,remaining_soft_seconds=2700-spent,total_soft_seconds=2700))
                if spent>=2700:raise legacy.bounded.InternalBudgetPause('Original cumulative45min soft cap exhausted; no new parent')
                legacy.bounded.budget_status(output,run_root,legacy.bounded.NEXT_PARENT_RESERVE_BYTES)
                data.parent.mkdir(parents=True,exist_ok=True);clock=time.perf_counter()
                command=[sys.executable,str(ROOT/'scripts/record_job.py'),'--output',str(jobs),'--run-id','parent_%03d'%i,'--resume-strategy','none','--',str(sim_python),str(Path(__file__)),'worker','--corpus',str(output),'--parent-index',str(i),'--output',str(data)]
                code=subprocess.run(command,check=False,env=env).returncode;elapsed=time.perf_counter()-clock
                status=read(sp) if sp.exists() else {}
            # Existing or interrupted parent paths are closed, never launched again.
            closed=legacy.old.mechanical_closure(data,plan,status,elapsed,code)
            write(output/'closures'/('%03d.json'%i),closed)
            rows=checked_closures(output,value)
            write(stage/'progress.json',dict(closed_indices=[r['index'] for r in rows],new_requested_slots=54,original_requested_slots=324,
                cumulative_worker_seconds=None if any(r['worker_elapsed_unknown'] for r in rows) else cumulative_worker_seconds(inherited,rows),automatic_certify=False))
            legacy.bounded.budget_status(output,run_root,0)
            if code not in (None,0):raise RuntimeError('Recovery runtime failure closed; no automatic continuation')
        rows=checked_closures(output,value)
        write(stage/'complete.json',dict(protocol=PROTOCOL,closed_indices=[x['index'] for x in rows],new_requested_slots=54,
            inherited_worker_seconds=inherited,cumulative_worker_seconds=cumulative_worker_seconds(inherited,rows),
            attempted_lower=sum(x['attempted_lower'] for x in rows),attempted_upper=sum(x['attempted_upper'] for x in rows),
            route_quality_not_inferred=True,automatic_certify=False))

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('prepare');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('recover');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True);p.add_argument('--sim-python',type=Path,required=True);p.add_argument('--resume',action='store_true')
    p=sub.add_parser('worker');p.add_argument('--corpus',type=Path,required=True);p.add_argument('--parent-index',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(argv)
    if a.action=='prepare':prepare(a.output);return 0
    legacy.bounded.runtime_guard()
    if any(os.environ.get(k)!='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')):raise ValueError('One-thread environment required')
    if a.action=='worker':
        r=worker(a.corpus,a.parent_index,a.output);return int(bool(r['fatal_error'] or r['shutdown_error']))
    try:run_recovery(a.corpus,a.run_root,a.sim_python,a.resume)
    except legacy.bounded.InternalBudgetPause as error:print(str(error),file=sys.stderr);return 3
    return 0

if __name__=='__main__':raise SystemExit(main())
