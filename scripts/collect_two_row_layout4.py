"""Four preregistered DEV_COLLECTION layouts, sequential fresh workers.

No sampling until success, interrupted-parent rerun, role assignment, or model
training. Resume skips closed parents and closes interrupted partial parents
with explicit attempt bounds; it never duplicates their proposals.
"""
import argparse
import copy
import datetime
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import collect_observed_two_row_pilot as collector


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []


def physical_layout_hash(centers,halves,goals,entry,quantum=None):
    """Only observed/audited physical geometry; no color, seed or robot joints."""
    values=np.concatenate([np.asarray(x,dtype=float).ravel() for x in (centers,halves,goals,entry)])
    if len(values)!=36 or not np.isfinite(values).all():raise ValueError('invalid physical layout arrays')
    array=np.rint(values/quantum).astype('<i8') if quantum else values.astype('<f8')
    return hashlib.sha256(array.tobytes()).hexdigest()


def scene_geometry_hash(centers,halves,goals,quantum=None):
    # Preparation/FK drift must not make the same physical scene look unique.
    return physical_layout_hash(centers,halves,goals,[0.,0.,0.],quantum)


def geometry_precheck(config):
    centers,halves=collector.validate_config(config)
    checked=0
    for target in collector.registered_target_indices(config):
        for sequence in itertools.product(collector.PASSAGES,repeat=2):
            raw=np.vstack([config['entry_xyz'],collector.proposed_waypoints(config['goal_xyz'][target],sequence,config)])
            for xyz in (raw,collector.legacy.resample(raw,24)):
                if not collector.legacy.tip_polyline_clear(xyz,centers,halves,config['tip_clearance_m']):
                    return dict(passed=False,checked=checked,error='registered tip guide path collides',target=target,sequence=sequence)
                if collector.crossing_signature(xyz,config)!=sequence:
                    return dict(passed=False,checked=checked,error='registered guide/H24 type mismatch',target=target,sequence=sequence)
            checked+=1
    return dict(passed=True,checked=checked,scope='Ideal tip polylines only; no IK or robot feasibility claim')


def build_plan(registration,base):
    if (registration['protocol']!='two_row_layout4_coordinator_v5' or registration['role']!='DEV_COLLECTION' or
            registration['parent_seeds']!=[283100,283101,283102,283103] or registration['requested_route_proposals']!=108 or
            registration['requested_parents']!=4 or registration['targets_per_parent']!=3 or
            registration['proposals_per_target']!=9 or registration['post_height_m']!=.14 or
            registration['geometry_resampling_attempts']!=0 or registration['failed_parent_replacements']!=0):
        raise ValueError('four-parent pilot registration changed')
    rng=np.random.RandomState(registration['layout_rng_seed'])
    sample=lambda bounds:float(rng.uniform(*bounds))
    result=[]
    for seed in registration['parent_seeds']:
        config=copy.deepcopy(base)
        config.update(protocol=collector.LAYOUT_PROTOCOL,seed=seed,selected_target_indices=[0,1,2],requested_route_proposals=27)
        config['post_size_xyz'][2]=registration['post_height_m']
        config['row_x']=[sample(bounds) for bounds in registration['row_x_ranges']]
        config['post_y']=[[sample(bounds) for bounds in row] for row in registration['post_y_ranges']]
        config['goal_xyz']=[[sample(registration['goal_x_range']),sample(bounds),registration['goal_z']] for bounds in registration['goal_y_ranges']]
        config['entry_xyz']=[sample(registration['entry_x_range']),sample(registration['entry_y_range']),registration['entry_z']]
        config['preparation_xyz']=[config['entry_xyz'][:2]+[base['preparation_xyz'][0][2]],config['entry_xyz'].copy()]
        check=geometry_precheck(config)
        centers,halves=collector.geometry(config)
        fingerprint=physical_layout_hash(centers,halves,config['goal_xyz'],config['entry_xyz'],registration['physical_layout_hash_quantization_m'])
        result.append(dict(parent_id='two_row_reach_%06d'%seed,config=config,geometry_precheck=check,registered_layout_1mm_sha256=fingerprint,
            registered_geometry_1mm_sha256=scene_geometry_hash(centers,halves,config['goal_xyz'],registration['physical_layout_hash_quantization_m'])))
    return result


def duplicate_groups(rows,key):
    groups={}
    for row in rows:
        if row.get(key):groups.setdefault(row[key],[]).append(row['parent_id'])
    return [members for members in groups.values() if len(members)>1]


def parent_evidence(data,plan,worker_seconds,returncode,quantum):
    row=dict(parent_id=plan['parent_id'],role='DEV_COLLECTION',requested_routes=27,worker_seconds=worker_seconds,
        worker_returncode=returncode,geometry_precheck=plan['geometry_precheck'],registered_layout_1mm_sha256=plan['registered_layout_1mm_sha256'],
        registered_geometry_1mm_sha256=plan['registered_geometry_1mm_sha256'])
    if not plan['geometry_precheck']['passed']:
        row.update(status='geometry_precheck_failed',attempted_lower=0,attempted_upper=0,unattempted_lower=27,unattempted_upper=27)
        return row
    records=read_rows(data/'attempts.jsonl')
    row['strict_restore_passes']=sum(r.get('restore',{}).get('global_inventory_equal',False) and
        r.get('restore',{}).get('max_abs')==0 and r.get('restore',{}).get('rgb_max_difference')==0 and
        bool(r.get('observation_restore')) and all(r['observation_restore'].values()) for r in records)
    summary=json.loads((data/'summary.json').read_text()) if (data/'summary.json').exists() else None
    if len(records)>27:raise ValueError('route budget exceeded')
    attempts=[(r['input_id'],r['attempt']) for r in records]
    if len(attempts)!=len(set(attempts)):raise ValueError('duplicate route attempt in parent')
    if summary:
        if summary['requested_route_proposals']!=27 or summary['route_attempts']!=len(records):raise ValueError('summary/ledger mismatch')
        if 'restore_passes' in summary and summary['restore_passes']!=row['strict_restore_passes']:raise ValueError('restore count mismatch')
        row.update(status=summary['status'],attempted_lower=len(records),attempted_upper=len(records),
            unattempted_lower=27-len(records),unattempted_upper=27-len(records),summary=summary)
    else:
        row.update(status='interrupted_no_retry',attempted_lower=len(records),attempted_upper=min(len(records)+1,27),
            unattempted_lower=max(26-len(records),0),unattempted_upper=27-len(records),
            interruption_note='One unlogged in-flight proposal may exist; failed parent is retained and never replaced.')
    if (data/'artifact_hashes.json').exists():
        expected=json.loads((data/'artifact_hashes.json').read_text())
        if not all(digest(data/name)==sha for name,sha in expected.items()):raise ValueError('parent artifact SHA mismatch')
        row['artifact_hashes_verified']=len(expected)
    parent=data/plan['parent_id'];verification=parent/'verification_only.npz'
    if verification.exists():
        audited=json.loads((parent/'restore_reference.json').read_text())
        posts=[audited['state']['_extra_derived_two_row_post_%d'%index] for index in range(4)]
        actual_centers=np.asarray([post['pose'][:3] for post in posts])
        actual_halves=np.asarray([np.diff(np.asarray(post['bounding_box']).reshape(3,2),axis=1).ravel()/2 for post in posts])
        with np.load(verification,allow_pickle=False) as v, np.load(parent/'observation.npz',allow_pickle=False) as o:
            values=[actual_centers,actual_halves,v['target_centers'],o['gripper_pose'][:3]]
            expected_centers,expected_halves=collector.geometry(plan['config'])
            if not (np.allclose(values[0],expected_centers,atol=1e-6,rtol=0) and
                    np.allclose(values[1],expected_halves,atol=1e-6,rtol=0) and
                    np.allclose(values[2],plan['config']['goal_xyz'],atol=1e-6,rtol=0)):
                raise ValueError('audited actual geometry differs from registration')
            row['actual_layout_exact_sha256']=physical_layout_hash(*values)
            row['actual_layout_1mm_sha256']=physical_layout_hash(*values,quantum=quantum)
            row['actual_geometry_exact_sha256']=scene_geometry_hash(*values[:3])
            row['actual_geometry_1mm_sha256']=scene_geometry_hash(*values[:3],quantum=quantum)
        row['initial_rgb_sha256']=digest(parent/'front.png')
        row['world_audit_sha256']=digest(parent/'restore_reference.json')
    return row


def publish_progress(output,plan,closures,source_sha):
    duplicates=duplicate_groups(closures,'actual_geometry_1mm_sha256')
    registered_duplicates=duplicate_groups(plan,'registered_geometry_1mm_sha256')
    observations=[];supervision=[];attempts=[]
    for row in closures:
        seed=row['parent_id'].rsplit('_',1)[1];prefix='raw/'+seed+'/'
        data=output/'raw'/seed
        for item in read_rows(data/'observations.jsonl'):
            item['image']=prefix+item['image'];observations.append(item)
        for item in read_rows(data/'supervision.jsonl'):
            for field in ('observation','verification_only'):item[field]=prefix+item[field]
            item['routes']=[prefix+p for p in item['routes']];supervision.append(item)
        for item in read_rows(data/'attempts.jsonl'):
            item['source_parent_directory']=prefix;attempts.append(item)
    for name,items in [('observations',observations),('supervision',supervision),('attempts',attempts)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(x,allow_nan=False)+'\n' for x in items),encoding='utf-8')
    summary=dict(protocol='two_row_layout4_coordinator_v5',role='DEV_COLLECTION',requested_parents=4,closed_parents=len(closures),
        requested_routes=108,completed_ledger_routes=len(attempts),accepted_references=sum(r['success'] for r in attempts),
        route_attempts_lower=sum(r['attempted_lower'] for r in closures),
        route_attempts_upper=sum(r['attempted_upper'] for r in closures)+27*(4-len(closures)),
        unattempted_routes_lower=sum(r['unattempted_lower'] for r in closures),
        unattempted_routes_upper=sum(r['unattempted_upper'] for r in closures)+27*(4-len(closures)),
        live_unclosed_parent_attempts_not_read=True,
        summarized_get_path_calls_lower=sum(r.get('summary',{}).get('get_path_calls',0) for r in closures),
        get_path_call_count_complete=len(closures)==4 and all('summary' in r for r in closures),
        finalized_worker_seconds=sum(r.get('worker_seconds') or 0 for r in closures),
        interrupted_worker_cost_unknown=any(r.get('worker_seconds') is None for r in closures),
        parent_closures=closures,actual_geometry_duplicate_groups=duplicates,
        registered_geometry_duplicate_groups=registered_duplicates,
        future_cross_split_use_blocked=bool(duplicates or registered_duplicates),training_authorized=False,source_sha256=source_sha,
        layout_hash_scope='Duplicate gate: 4 actual audited post centers/halves + 3 actual goal centers, exact and 1mm; excludes entry drift, color, seed, robot joints, success dummies and velocity. Layout-plus-entry/RGB/world hashes supplementary.',
        status='complete' if len(closures)==4 else 'running')
    write(output/'summary.json',summary)


def run(registration_path,base_path,output,run_root,resume=False):
    registration=json.loads(registration_path.read_text());base=json.loads(base_path.read_text())
    plan=build_plan(registration,base)
    source_sha={p.name:digest(p) for p in (Path(__file__),Path(collector.__file__),Path(collector.legacy.__file__),
        Path(collector.legacy.native_snapshot.__code__.co_filename),Path(collector.two_row_anchor.__file__),registration_path,base_path)}
    frozen=dict(registration=registration,parent_plan=plan,source_sha256=source_sha)
    if output.exists():
        if not resume or json.loads((output/'registration.json').read_text())!=frozen:raise ValueError('resume requires identical registration/source')
    else:
        output.mkdir(parents=True);write(output/'registration.json',frozen)
    run_root.mkdir(parents=True,exist_ok=True);closures=[]
    if not (output/'summary.json').exists():publish_progress(output,plan,closures,source_sha)
    for parent in plan:
        seed='%06d'%parent['config']['seed'];data=output/'raw'/seed;closure_path=output/'closures'/(seed+'.json')
        config_path=output/'configs'/(seed+'.json')
        if closure_path.exists():
            closed=json.loads(closure_path.read_text())
            current=parent_evidence(data,parent,closed.get('worker_seconds'),closed.get('worker_returncode'),
                registration['physical_layout_hash_quantization_m'])
            if current!=closed:raise ValueError('closed parent evidence changed; refuse resume')
            closures.append(closed);continue
        write(config_path,parent['config'])
        elapsed=None;returncode=None
        status_path=run_root/('parent_'+seed+'.status.json')
        if status_path.exists():
            status=json.loads(status_path.read_text())
            if status['status'] in ('starting','running'):
                for pid in (status.get('pid'),status.get('child_pid')):
                    if pid:
                        try:os.kill(pid,0)
                        except ProcessLookupError:continue
                        else:raise RuntimeError('parent worker still running; refuse overlapping resume')
            if status.get('start_utc') and status.get('end_utc'):
                elapsed=(datetime.datetime.fromisoformat(status['end_utc'])-datetime.datetime.fromisoformat(status['start_utc'])).total_seconds()
            returncode=status.get('exit_code')
        if parent['geometry_precheck']['passed'] and not data.exists() and not status_path.exists():
            tic=time.perf_counter()
            command=[sys.executable,str(HERE/'record_job.py'),'--output',str(run_root),'--run-id','parent_'+seed,
                '--resume-strategy','none','--',sys.executable,str(Path(collector.__file__)),
                '--config',str(config_path),'--output',str(data)]
            returncode=subprocess.run(command,check=False).returncode;elapsed=time.perf_counter()-tic
        evidence=parent_evidence(data,parent,elapsed,returncode,registration['physical_layout_hash_quantization_m'])
        write(closure_path,evidence);closures.append(evidence)
        publish_progress(output,plan,closures,source_sha)
        print(json.dumps(dict(parent_id=evidence['parent_id'],status=evidence['status'],closed=len(closures))),flush=True)
    publish_progress(output,plan,closures,source_sha)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-root',type=Path,required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--base-config',type=Path,default=HERE.parent/'configs/observed_two_row_pilot_v4.json')
    args=parser.parse_args();run(args.registration,args.base_config,args.output,args.run_root,args.resume)


if __name__=='__main__':main()
