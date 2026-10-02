"""Atomically export only explicit TRAIN/DEV_MODEL reservation roles.

Unselected JSONL rows are filtered by their parent_id scalar BEFORE decoding
the row. No unselected image, trajectory, current observation or target payload
is opened. Failed parents retain their requested identities and accounting.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile


INPUT_KEYS={'id','parent_id','split','image','instruction'}
ALLOWED_ROLES={'TRAIN','DEV_MODEL'}
PARENT_PATTERN=re.compile(rb'(?<!\\)"parent_id"\s*:\s*"([A-Za-z0-9_-]+)"')


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def row_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def selected_rows(path,selected):
    """Read only parent scalars for gating; never decode unselected payloads."""
    result=[]
    with Path(path).open('rb') as stream:
        for raw in stream:
            if not raw.endswith(b'\n'):
                # A live writer may be appending an uncommitted final row.
                break
            if not raw.strip():continue
            matches=PARENT_PATTERN.findall(raw)
            if len(matches)!=1:raise ValueError('one unescaped parent_id scalar required in JSONL')
            parent=matches[0].decode('ascii')
            if parent not in selected:continue
            row=json.loads(raw.decode('utf-8-sig'))
            if row.get('parent_id')!=parent:raise ValueError('parent gate differs from decoded selected row')
            result.append(row)
    return result


def inside(path,directory):
    try:Path(path).resolve().relative_to(Path(directory).resolve())
    except ValueError:raise ValueError('selected parent reference escapes its own directory: '+str(path))
    return Path(path).resolve()


def parent_directory(source,parent):
    """Reject parent aliases before opening any selected reference."""
    folder=Path(source)/parent
    if folder.resolve()!=Path(source).resolve()/parent:
        raise ValueError('selected parent directory is an alias or escapes its reserved identity: '+str(folder))
    return folder


def selected_file(source,parent,filename):
    folder=parent_directory(source,parent)
    path=inside(source/filename,folder)
    if not path.is_file():raise ValueError('missing selected parent file: '+str(path))
    return path


def write_rows(path,rows):
    Path(path).write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')


def selected_collection_rows(source,selected,targets,attempts_per_target,collection_format):
    if collection_format=='natural_v1':
        return {key:selected_rows(source/(key+'.jsonl'),selected)
                for key in ('parents','observations','supervision','attempts')}
    if collection_format!='obstacle_v3' or targets!=3 or attempts_per_target!=4:
        raise ValueError('obstacle_v3 requires its original three targets and four proposals')
    # The original obstacle collector has no parent ledger. A parent closes
    # after all twelve recorded outcomes, or one explicit setup failure.
    rows={key:selected_rows(source/(key+'.jsonl'),selected)
          for key in ('observations','supervision','attempts')}
    raw_attempts=rows['attempts']; rows['attempts']=[]; rows['parents']=[]
    grouped=defaultdict(list)
    for row in raw_attempts:grouped[row['parent_id']].append(row)
    for parent in sorted(selected):
        attempts=grouped[parent]
        failures=[row for row in attempts if row.get('phase')=='parent_setup']
        if failures:
            if len(failures)!=1 or len(attempts)!=1 or failures[0].get('success',False):
                raise ValueError('inconsistent original setup failure: '+parent)
            failure=failures[0]
            rows['parents'].append(dict(parent_id=parent,split=failure['split'],setup_success=False,
                original_setup_failure=failure,closure='original setup failure; proposal slots unattempted'))
            for target in range(targets):
                for attempt in range(attempts_per_target):
                    rows['attempts'].append(dict(parent_id=parent,split=failure['split'],
                        input_id=parent+'_target'+str(target),attempt=attempt,attempted=False,
                        success=False,phase='unattempted_after_parent_setup_failure',
                        original_setup_failure_sha256=row_hash(failure)))
        else:
            expected={(parent+'_target'+str(target),attempt)
                      for target in range(targets) for attempt in range(attempts_per_target)}
            if len(attempts)!=len(expected) or {(r.get('input_id'),r.get('attempt')) for r in attempts}!=expected:
                raise ValueError('all requested proposal outcomes must be recorded: '+parent)
            splits={row['split'] for row in attempts}
            if len(splits)!=1:raise ValueError('original parent split inconsistent: '+parent)
            rows['parents'].append(dict(parent_id=parent,split=next(iter(splits)),setup_success=True,
                closure='all original twelve proposal outcomes recorded'))
            rows['attempts'].extend(attempts)
    return rows


def build(root,reservation,collection,roles,output,targets=3,attempts_per_target=3,collection_format='natural_v1'):
    root,reservation,output=Path(root).resolve(),Path(reservation).resolve(),Path(output).resolve()
    if not roles or len(set(roles))!=len(roles) or not set(roles)<=ALLOWED_ROLES:
        raise ValueError('only explicit unique TRAIN/DEV_MODEL roles are permitted; no locked/score/calibration export')
    if min(targets,attempts_per_target)<1:raise ValueError('positive target/proposal counts required')
    if output.exists():raise ValueError('export exists; never replace a versioned dataset')
    registry=json.loads(reservation.read_text())
    entries=[item for item in registry['collections'] if item['source']==collection]
    if len(entries)!=1:raise ValueError('collection must exactly match one reservation source')
    entry=entries[0]
    source=inside(root/collection,root)
    all_parents={}
    for role,(first,last) in entry['roles_inclusive'].items():
        for number in range(first,last+1):
            parent=entry['parent_prefix']+('%06d'%number)
            if parent in all_parents:raise ValueError('overlapping reservation ranges')
            all_parents[parent]=role
    selected={parent:role for parent,role in all_parents.items() if role in roles}
    if any(role not in entry['roles_inclusive'] for role in roles) or not selected:
        raise ValueError('requested role absent from reservation')
    streams=('parents','observations','supervision','attempts')
    rows=selected_collection_rows(source,selected,targets,attempts_per_target,collection_format)
    source_row_hashes={key:row_hash(value) for key,value in rows.items()}
    grouped={key:defaultdict(list) for key in streams}
    for key,values in rows.items():
        for row in values:grouped[key][row['parent_id']].append(row)
    exported={key:[] for key in streams}
    file_hashes,provenance,unavailable={},{},[]
    for parent,role in sorted(selected.items()):
        folder=parent_directory(source,parent)
        records=grouped['parents'][parent]
        if len(records)!=1:raise ValueError('requested parent not closed or duplicate: '+parent)
        parent_record=records[0]
        observations,labels,attempts=[grouped[key][parent] for key in ('observations','supervision','attempts')]
        expected={parent+'_target'+str(target) for target in range(targets)}
        expected_attempts={(identifier,attempt) for identifier in expected for attempt in range(attempts_per_target)}
        if len(attempts)!=len(expected_attempts) or {(r.get('input_id'),r.get('attempt')) for r in attempts}!=expected_attempts:
            raise ValueError('all requested proposal outcomes must be recorded: '+parent)
        if not parent_record.get('setup_success',False):
            if observations or labels or any(r.get('attempted',True) or r.get('success',False) for r in attempts):
                raise ValueError('failed setup has inconsistent observation or proposal records')
            unavailable.extend(dict(id=identifier,parent_id=parent,split=role,reason='parent_setup_failed_no_observation') for identifier in sorted(expected))
        else:
            if len(observations)!=targets or {r['id'] for r in observations}!=expected:
                raise ValueError('requested initialized parent observations incomplete: '+parent)
            if len(labels)!=targets or {r['id'] for r in labels}!=expected:
                raise ValueError('requested initialized parent references incomplete: '+parent)
            if any(set(row)!=INPUT_KEYS for row in observations):raise ValueError('strict five-key observation input violated')
            if any(not row.get('attempted',True) for row in attempts):raise ValueError('initialized parent contains unattempted proposals')
            for row in observations:
                exported['observations'].append(dict(row,split=role,image=str(selected_file(source,parent,row['image']))))
            for row in labels:
                successful={r['route_file'] for r in attempts if r['input_id']==row['id'] and r.get('success',False)}
                if successful!=set(row.get('routes',[])):raise ValueError('successful attempts and positive reference list differ')
                changed=dict(row,split=role,observation=str(selected_file(source,parent,row['observation'])),
                             routes=[str(selected_file(source,parent,name)) for name in row.get('routes',[])])
                if row.get('verification_only'):changed['verification_only']=str(selected_file(source,parent,row['verification_only']))
                exported['supervision'].append(changed)
        # Traverse/hash only the already gated selected parent folder. Refuse
        # symlinks that would read another parent, even for ancillary files.
        parent_files={}
        if folder.exists():
            inside(folder,source)
            for path in sorted(folder.rglob('*')):
                if path.is_file():
                    path=inside(path,folder)
                    parent_files[str(path)]=digest(path)
        file_hashes.update(parent_files)
        exported['parents'].append(dict(parent_record,split=role,original_split=parent_record.get('split')))
        for row in attempts:
            changed=dict(row,split=role,source_dataset=str(source))
            for key in ('route_file','failed_partial_route','failed_route_file'):
                if row.get(key):changed[key]=str(selected_file(source,parent,row[key]))
            exported['attempts'].append(changed)
        provenance[parent]=dict(reserved_role=role,source_split=parent_record.get('split'),setup_success=bool(parent_record.get('setup_success')),
                                source_row_hashes={key:row_hash(grouped[key][parent]) for key in streams},source_files_sha256=parent_files)
    # A live collector may append UNSELECTED parents. Selected rows and files
    # must be identical across the export; source files are never modified.
    reread=selected_collection_rows(source,selected,targets,attempts_per_target,collection_format)
    if any(row_hash(reread[key])!=value for key,value in source_row_hashes.items()):
        raise RuntimeError('selected source records changed during export')
    if any(digest(filename)!=expected for filename,expected in file_hashes.items()):
        raise RuntimeError('selected source files changed during export')
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=output.with_name(output.name+'.staging')
    if staging.exists():raise ValueError('staging exists; inspect without overwriting')
    staging.mkdir()
    for key,value in exported.items():write_rows(staging/(key+'.jsonl'),value)
    write_rows(staging/'unavailable_observations.jsonl',unavailable)
    manifest=dict(protocol='observation_parent_reservation_v1',created_utc=datetime.now(timezone.utc).isoformat(),
        reservation_path=str(reservation),reservation_sha256=digest(reservation),registered_utc=registry['registered_utc'],
        source_dataset=str(source),collection_format=collection_format,requested_roles=roles,requested_parent_ids=sorted(selected),
        requested_parent_counts={role:sum(value==role for value in selected.values()) for role in roles},
        initialized_parent_counts={role:sum(r['split']==role and r['setup_success'] for r in exported['parents']) for role in roles},
        setup_failed_parents=[r['parent_id'] for r in exported['parents'] if not r['setup_success']],
        unavailable_observation_inputs=len(unavailable),observations=len(exported['observations']),
        supervised_examples_by_role={role:sum(r['split']==role and bool(r.get('routes')) for r in exported['supervision']) for role in roles},
        semantic_inputs_by_role={role:sum(r['split']==role for r in exported['supervision']) for role in roles},
        attempts=len(exported['attempts']),successful_attempts=sum(bool(r.get('success')) for r in exported['attempts']),
        unattempted_proposals=sum(not r.get('attempted',True) for r in exported['attempts']),
        input_contract=sorted(INPUT_KEYS),input_manifest_sha256=digest(staging/'observations.jsonl'),
        supervision_manifest_sha256=digest(staging/'supervision.jsonl'),source_selected_row_hashes=source_row_hashes,
        source_parents=provenance,script_sha256=digest(__file__),source_unchanged=True,
        selection='exact complete reserved parent ranges; setup failures kept and never replaced; zero-reference instructions retained',
        unselected_access='only parent_id scalar scanned before row decoding; no unselected targets or referenced files decoded/opened',
        model_cache_policy='no hidden cache copied; new true frozen-Qwen cache requires this exact five-key manifest and authorized GPU queue')
    (staging/'export_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    staging.rename(output)
    return {key:manifest[key] for key in ('requested_parent_counts','initialized_parent_counts','setup_failed_parents','observations',
                                          'supervised_examples_by_role','semantic_inputs_by_role','input_manifest_sha256')}


def self_test():
    symlink_check='unavailable on host'
    with tempfile.TemporaryDirectory() as directory:
        root=Path(directory);source=root/'data/fixture';source.mkdir(parents=True)
        registry=dict(registered_utc='fixture',collections=[dict(source='data/fixture',parent_prefix='parent_',
            roles_inclusive=dict(TRAIN=[1,2],DEV_MODEL=[3,3],TEST_LOCKED=[4,4]))])
        reservation=root/'reservation.json';reservation.write_text(json.dumps(registry))
        rows={key:[] for key in ('parents','observations','supervision','attempts')}
        for number in (1,2,3):
            parent='parent_%06d'%number;folder=source/parent;folder.mkdir()
            success=number!=2
            rows['parents'].append(dict(parent_id=parent,split='RAW',setup_success=success))
            if success:
                (folder/'front.png').write_bytes(b'fixture RGB')
                (folder/'observation.npz').write_bytes(b'fixture current')
            for target in range(3):
                identifier=parent+'_target'+str(target)
                refs=[]
                for attempt in range(3):
                    accepted=success and target!=2
                    row=dict(parent_id=parent,input_id=identifier,attempt=attempt,attempted=success,success=accepted)
                    if accepted:
                        filename=parent+'/route%d_%d.npz'%(target,attempt)
                        (source/filename).write_bytes(b'fixture trajectory');refs.append(filename);row['route_file']=filename
                    rows['attempts'].append(row)
                if success:
                    rows['observations'].append(dict(id=identifier,parent_id=parent,split='RAW',image=parent+'/front.png',instruction='fixture'))
                    rows['supervision'].append(dict(id=identifier,parent_id=parent,split='RAW',observation=parent+'/observation.npz',routes=refs))
        for key,value in rows.items():
            write_rows(source/(key+'.jsonl'),value)
            with (source/(key+'.jsonl')).open('ab') as stream:
                stream.write(b'{"parent_id":"parent_000004","forbidden_target":THIS_PAYLOAD_MUST_NOT_BE_DECODED}\n')
        before={key:digest(source/(key+'.jsonl')) for key in rows}
        result=build(root,reservation,'data/fixture',['TRAIN','DEV_MODEL'],root/'out')
        assert result['requested_parent_counts']=={'TRAIN':2,'DEV_MODEL':1}
        assert result['setup_failed_parents']==['parent_000002'] and result['observations']==6
        assert result['supervised_examples_by_role']=={'TRAIN':2,'DEV_MODEL':2}
        assert before=={key:digest(source/(key+'.jsonl')) for key in rows}
        assert all(set(json.loads(line))==INPUT_KEYS for line in (root/'out/observations.jsonl').read_text().splitlines())
        for forbidden in ('TEST_LOCKED','DEV_SCORE','CALIBRATION'):
            try:build(root,reservation,'data/fixture',[forbidden],root/forbidden)
            except ValueError:pass
            else:raise AssertionError('forbidden role accepted')
        # A selected record cannot smuggle a locked file through its path.
        poisoned=dict(rows['observations'][0],image='parent_000004/front.png')
        write_rows(source/'observations.jsonl',[poisoned]+rows['observations'][1:])
        try:build(root,reservation,'data/fixture',['TRAIN'],root/'escape')
        except ValueError as exc:assert 'escapes' in str(exc)
        else:raise AssertionError('cross-parent path accepted')
        write_rows(source/'observations.jsonl',rows['observations'])
        # The selected parent folder itself must not alias a locked parent.
        locked=source/'parent_000004';locked.mkdir()
        alias=source/'parent_000099'
        try:alias.symlink_to(locked,target_is_directory=True)
        except (OSError,NotImplementedError):pass
        else:
            try:parent_directory(source,alias.name)
            except ValueError as exc:assert 'alias' in str(exc)
            else:raise AssertionError('parent-directory alias accepted')
            symlink_check='passed'
        write_rows(source/'attempts.jsonl',rows['attempts'][:-1])
        try:build(root,reservation,'data/fixture',['TRAIN','DEV_MODEL'],root/'partial')
        except ValueError as exc:assert 'all requested proposal' in str(exc)
        else:raise AssertionError('unfinished requested parent silently skipped')
        assert not (root/'partial').exists()
    print(json.dumps(dict(status='passed',parent_directory_symlink_check=symlink_check,checks=['filter before forbidden payload decoding','exact reserved parents including setup failure',
        'zero refs retained','strict five-key input','source unchanged','forbidden roles rejected','cross-parent path rejected','incomplete parent aborts atomic export'])))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--root',type=Path)
    parser.add_argument('--reservation',type=Path)
    parser.add_argument('--collection')
    parser.add_argument('--roles',nargs='+')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--targets',type=int,default=3)
    parser.add_argument('--attempts-per-target',type=int,default=3)
    parser.add_argument('--collection-format',choices=['natural_v1','obstacle_v3'],default='natural_v1')
    args=parser.parse_args()
    if args.self_test:self_test();return
    if not all((args.root,args.reservation,args.collection,args.roles,args.output)):
        parser.error('root/reservation/collection/explicit roles/output required')
    print(json.dumps(build(args.root,args.reservation,args.collection,args.roles,args.output,args.targets,args.attempts_per_target,args.collection_format)),flush=True)


if __name__=='__main__':main()
