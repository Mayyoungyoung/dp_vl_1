"""Read-only archive/analysis of COMPLETED 12000 online runs; no model imports.

sync requires an explicit remote family, immutable commit and job receipts.
analyze uses only archived request records/sealed predictions and original summaries.
Neither subcommand can launch inference, open raw corpus data or change a result.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p/'AGENTS.md').is_file())
IDS = [f'two_row_reach_{parent}_target{target}' for parent in range(283264, 283276) for target in range(3)]
ARMS = ('constant', 'no_direct')
TRAIN = {
    'constant': 'reports/observed_two_row_prefix76_convergence_v1',
    'no_direct': 'reports/observed_two_row_prefix76_no_direct_v1',
}
METRICS = ('TipValidAtK', 'AnyTipValidAtK', 'UniqueClassifiedTipValidAtK',
           'KnownReferenceTypeCoverageAtK', 'UnknownTypeTipValidCount',
           'DuplicateClassifiedTipValidCount', 'TipClearAtK', 'semantic_goal_accuracy')
REMOTE = r'''
import datetime,hashlib,io,json,sys,tarfile
from pathlib import Path
args=json.loads(sys.stdin.readline());project=Path('/home/wzy/dpvlm/route_set_v1')
family=project/'runs'/args['family'];source=project/'research_v2/releases'/args['source']
family.resolve().relative_to((project/'runs').resolve());assert source.is_dir()
jobs={}
for name in args['jobs']:
 p=family/name;v=json.loads(p.read_text());assert v['status'] in ('completed','failed') and v.get('exit_code') is not None
 assert v['code_commit']==args['source'];jobs[name]=v
files={str(f.relative_to(family)):f for f in sorted(family.rglob('*')) if f.is_file() and f.suffix in ('.json','.jsonl','.log','.npz','.xml','.txt')}
for name in args['sources']:
 p=source/name;p.resolve().relative_to(source.resolve());assert p.is_file();files['source/'+name]=p
for name in args['wrappers']:
 p=project/'research_v2/incoming'/name;p.resolve().relative_to((project/'research_v2/incoming').resolve());assert p.is_file();files['wrapper/'+name]=p
buffer=io.BytesIO();index={}
with tarfile.open(fileobj=buffer,mode='w') as archive:
 for name,path in sorted(files.items()):
  raw=path.read_bytes();index[name]={'source':str(path),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
  info=tarfile.TarInfo(name);info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
 value={'protocol':'completed_online12000_readonly_archive_v1','source_root':str(family),'source_commit':args['source'],'retrieved_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_receipts':args['jobs'],'files':index,'new_forward_requests':0,'new_qwen_encodings':0,'new_searches':0,'raw_corpus_opened':False}
 raw=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo('REMOTE_ARTIFACT_INDEX.json');info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
sys.stdout.buffer.write(buffer.getvalue())
'''


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
    if path.exists() and path.read_bytes() != data:
        raise ValueError('Do not overwrite different evidence: '+str(path))
    path.write_bytes(data)


def checked_relative(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name:
        raise ValueError('Only safe relative POSIX paths')
    return p


def sync(args):
    checked_relative(args.family)
    for name in args.job + args.wrapper + args.source_file:
        checked_relative(name)
    if not re.fullmatch('[0-9a-f]{40}', args.source):
        raise ValueError('Actual immutable source required')
    payload = dict(family=args.family, source=args.source, jobs=args.job,
                   wrappers=args.wrapper, sources=args.source_file)
    # stdin first line is data; the Python program is supplied as shell-quoted code.
    import shlex
    command = 'taskset -c 1 python3 -c '+shlex.quote(REMOTE)
    result = subprocess.run(['ssh', 'wzy3090', command], input=(json.dumps(payload)+'\n').encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace'))
    tar_path = ROOT/'.bootstrap'/(args.dest.name+'_completed.tar')
    if tar_path.exists():
        raise FileExistsError('Preserve existing archive: '+str(tar_path))
    tar_path.write_bytes(result.stdout)
    with tarfile.open(fileobj=io.BytesIO(result.stdout)) as archive:
        for entry in archive.getmembers():
            checked_relative(entry.name)
            if not entry.isfile():
                raise ValueError('Only ordinary evidence files permitted')
            path = args.dest/entry.name; raw = archive.extractfile(entry).read()
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists() and path.read_bytes() != raw:
                raise ValueError('Existing evidence changed: '+str(path))
            path.write_bytes(raw)
    index = verify_archive(args.dest)
    receipt = dict(archive_sha256=sha(tar_path), archive_bytes=tar_path.stat().st_size,
                   verified_files=len(index['files']), read_only=True, raw_corpus_opened=False,
                   new_forward_requests=0, new_qwen_encodings=0,
                   remote_artifact_index_sha256=sha(args.dest/'REMOTE_ARTIFACT_INDEX.json'))
    write_new(args.dest/'SYNC_RECEIPT.json', receipt)
    print(json.dumps(receipt))


def verify_archive(root):
    index = load(root/'REMOTE_ARTIFACT_INDEX.json')
    for name, entry in index['files'].items():
        checked_relative(name); path = root/name
        if sha(path) != entry['sha256'] or path.stat().st_size != entry['bytes']:
            raise ValueError('Archive mismatch: '+name)
    return index


def stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return dict(count=0, total=None, mean=None, median=None, p95=None, min=None, max=None)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite timing statistic')
    return dict(count=len(values), total=float(values.sum()), mean=float(values.mean()),
                median=float(np.median(values)), p95=float(np.percentile(values,95)),
                min=float(values.min()), max=float(values.max()))


def rows_by_id(rows):
    result = {r['id']:r for r in rows}
    if len(result) != len(rows) or set(result) != set(IDS):
        raise ValueError('Every registered request exactly once; no ordering-based pairing')
    return result


def analyze_arm(family, relative, arm):
    root = family/relative
    summary, receipt, comparison = (load(root/name) for name in
        ('summary.json','paired12000_online_receipt.json','exact_saved_comparison.json'))
    status = load(root/'status.json'); preflight = load(root/'preflight.json')
    if (receipt['arm'] != arm or receipt['summary_sha256'] != sha(root/'summary.json')
            or receipt['exact_saved_comparison_sha256'] != sha(root/'exact_saved_comparison.json')
            or summary['prediction_sha256'] != sha(root/'predictions.npz')
            or summary['cache_comparison_sha256'] != sha(root/'cache_comparison.json')):
        raise ValueError('Actual online artifact receipt mismatch')
    if status['status'] not in ('completed','completed_with_failures'):
        raise ValueError('Archive preserved, but completed pools required for this pair analysis')
    rows = rows_by_id([json.loads(line) for line in (root/'requests.jsonl').read_text().splitlines()])
    cmp_rows = rows_by_id(comparison['per_scene'])
    if summary['requested_requests'] != 36 or summary['requested_candidate_slots'] != 144:
        raise ValueError('Original request/candidate denominator changed')
    training = ROOT/TRAIN[arm]/'peak_seed0'; original = load(training/'summary.json')
    if preflight['checkpoint_sha256'] != original['best_checkpoint_sha256'] or preflight['checkpoint_step'] != original['best_step']:
        raise ValueError('Original completed best checkpoint differs')
    decisions = ('TipValid','semantic_goal_correct','starts_at_current_state','tip_segments_clear',
                 'event_state_sequence_correct','declared_passage_type')
    old_rows = {r['scene_id']:r for r in load(training/'dev_model/per_scene.json')}
    mismatch = []; counts = dict(finite_candidates=0, tip_valid=0, semantic_correct=0, tip_clear=0)
    with np.load(root/'predictions.npz', allow_pickle=False) as archive:
        ids = list(map(str, archive['scene_ids'])); paths = archive['paths']; opened = archive['gripper_open']
        if len(ids)!=36 or len(set(ids))!=36 or set(ids)!=set(IDS) or paths.shape!=(36,4,24,3) or opened.shape!=(36,4,24):
            raise ValueError('Aggregated saved pool dimensions differ')
        for identifier in IDS:
            r = rows[identifier]; sealed = root/'requests'/r['prediction_file']
            seal = load(sealed.with_suffix('.seal.json'))
            if sha(sealed)!=r['prediction_sha256'] or sha(sealed)!=seal['prediction_sha256'] or not seal['sealed_before_label_access']:
                raise ValueError('Per-request seal changed')
            if (r['requested_candidates']!=4 or len(r['candidates'])!=4 or any(r[x]!=0 for x in
                    ('retries','repairs','filtered_candidates','additional_complete_paths'))):
                raise ValueError('Unexpected candidate additions/filters/retries')
            with np.load(sealed,allow_pickle=False) as a:
                i=ids.index(identifier)
                if not np.array_equal(a['paths'],paths[i],equal_nan=True) or not np.array_equal(a['gripper_open'],opened[i],equal_nan=True):
                    raise ValueError('Aggregated and sealed request arrays differ')
            counts['finite_candidates'] += r['finite_path_candidates']
            for slot,(new,old) in enumerate(zip(r['candidates'],old_rows[identifier]['tip_candidates'])):
                for field,key in [('TipValid','tip_valid'),('semantic_goal_correct','semantic_correct'),('tip_segments_clear','tip_clear')]:
                    counts[key] += int(bool(new[field]))
                difference={key:dict(saved=old[key],online=new[key]) for key in decisions if old[key]!=new[key]}
                if difference:mismatch.append(dict(id=identifier,slot=slot,differences=difference))
    for key in METRICS:
        values=[r['metrics'][key] for r in rows.values() if r['metrics'].get(key) is not None]
        actual=float(np.mean(values)) if values else None
        expected=summary[key]
        if actual != expected and not (actual is not None and expected is not None and math.isclose(actual,expected,abs_tol=1e-12)):
            raise ValueError('Saved metric aggregation differs: '+key)
    attempted=[rows[i] for i in IDS if rows[i]['generation_attempted']]
    stages={key:stats([r['stages'][key] for r in attempted if key in r['stages']])
            for key in sorted({k for r in attempted for k in r['stages'] if k.endswith('_ms')})}
    return dict(arm=arm,original_best_step=original['best_step'],original_checkpoint_sha256=preflight['checkpoint_sha256'],
        status=status,requested_requests=36,charged_candidate_slots=144,actual_attempted_requests=len(attempted),
        generation_errors=[dict(id=i,error=rows[i]['generation_error'],attempted=rows[i]['generation_attempted'])
                           for i in IDS if rows[i]['generation_error'] is not None],
        counts=counts,metrics={key:summary[key] for key in METRICS},
        total_continuous_requested_ms=stats([rows[i]['continuous_request_wall_ms'] for i in IDS]),
        attempted_request_ms=stats([r['continuous_request_wall_ms'] for r in attempted]),
        first_request_ms=rows[IDS[0]]['continuous_request_wall_ms'],
        after_first_descriptive_ms=stats([rows[i]['continuous_request_wall_ms'] for i in IDS[1:]]),
        stages_ms=stages,stage_note='Geometry encoder is nested in geometry/head; do not add nested stages. First request remains in primary timing.',
        model_loading_seconds=summary['model_loading_seconds'],metadata_startup_seconds=summary['metadata_startup_seconds'],
        total_pipeline_wall_seconds=summary['total_pipeline_wall_seconds'],gpu_hours_reserved=summary['gpu_hours_reserved'],
        peak_cuda_allocated_bytes=summary['peak_cuda_allocated_bytes'],
        all_feature_bytes_exact=comparison['all_feature_bytes_exact'],
        feature_mismatch_ids=[i for i in IDS if not all(cmp_rows[i]['feature_exact_bytes'].values())],
        finite_xyz_comparison_requests=sum(cmp_rows[i]['xyz_max_abs_difference_m'] is not None for i in IDS),
        maximum_xyz_difference_from_original_m=max((cmp_rows[i]['xyz_max_abs_difference_m'] for i in IDS
            if cmp_rows[i]['xyz_max_abs_difference_m'] is not None),default=None),
        all_candidate_decisions_identical=not mismatch,decision_differences=mismatch,
        request_order=IDS,source_training_cost_counted_again=False), rows


def analyze(args):
    index=verify_archive(args.dest)
    output={}; rows={}
    for arm,relative in [('constant',args.constant),('no_direct',args.no_direct)]:
        checked_relative(relative)
        output[arm],rows[arm]=analyze_arm(args.dest,relative,arm)
    pairs=[]
    for identifier in IDS:
        a,b=rows['constant'][identifier],rows['no_direct'][identifier]
        pairs.append(dict(id=identifier,parent_id=a['parent_id'],
            continuous_request_ms=dict(constant=a['continuous_request_wall_ms'],no_direct=b['continuous_request_wall_ms'],
                                       difference_no_direct_minus_constant=b['continuous_request_wall_ms']-a['continuous_request_wall_ms']),
            metrics={key:dict(constant=a['metrics'].get(key),no_direct=b['metrics'].get(key)) for key in METRICS}))
    parents=[]
    for parent in sorted({r['parent_id'] for r in pairs}):
        group=[r for r in pairs if r['parent_id']==parent]
        parents.append(dict(parent_id=parent,requests=len(group),
            latency_difference_ms=stats([r['continuous_request_ms']['difference_no_direct_minus_constant'] for r in group])))
    result=dict(protocol='saved_online12000_paired_analysis_v1',source_commit=index['source_commit'],
        arms=output,paired_requests=72,charged_candidate_slots=288,
        new_forward_requests=0,new_qwen_encodings=0,new_searches=0,raw_corpus_opened=False,
        original_training_cost_counted_again=False,per_request=pairs,per_parent=parents,
        interpretation='Single fixed sequential measurement on reused DEV. Different originally selected best steps at equal12000 training budget. Latency is measured, not a causal speedup test; OS/cache/startup/noise remain. Capacity reduction is explicit. No new quality experiment or checkpoint choice.')
    write_new(args.dest/'PAIRED_ONLINE_ANALYSIS.json',result)
    print(json.dumps({arm:{k:v for k,v in output[arm].items() if k in ('original_best_step','counts','attempted_request_ms','all_feature_bytes_exact','all_candidate_decisions_identical')} for arm in ARMS},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('sync');p.add_argument('--family',required=True);p.add_argument('--source',required=True)
    p.add_argument('--dest',type=Path,required=True);p.add_argument('--job',action='append',required=True)
    p.add_argument('--wrapper',action='append',default=[]);p.add_argument('--source-file',action='append',default=[])
    p=sub.add_parser('analyze');p.add_argument('--dest',type=Path,required=True)
    p.add_argument('--constant',default='constant');p.add_argument('--no-direct',default='no_direct')
    args=parser.parse_args();globals()[args.action](args)


if __name__=='__main__':main()
