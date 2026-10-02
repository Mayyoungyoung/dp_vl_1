"""Hash-verified local analysis of the twelve completed frozen transfer outputs."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import numpy as np

FIELDS=('candidate_matched_ADE_m','candidate_endpoint_error_m','event_state_accuracy','event_sequence_accuracy')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def stats(values):
    return dict(values=values,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)))


def analyze(root,archive,report):
    root,archive,report=map(Path,(root,archive,report))
    source=root/'runs/observed_multitask_legacy12_transfer_v1'
    if report.exists():raise FileExistsError('Fresh report directory required')
    result=read(source/'evaluation/summary.json');registration=read(source/'preregistered_config.json')
    export=read(archive/'export_manifest.json')
    if result['status']!='completed' or result['additional_training_steps']!=0 or result['checkpoint_reselection']:
        raise ValueError('Expected completed frozen transfer')
    if result['collection_denominators']!={k:export[k] for k in result['collection_denominators']}:
        raise ValueError('Export denominator differs')
    rows={(r['arm'],r['seed'],r['checkpoint_kind']):r for r in result['entries']}
    planned={(r['arm'],r['seed'],r['checkpoint_kind']):r for r in registration['requested_checkpoints']}
    if rows.keys()!=planned.keys() or len(rows)!=12:raise ValueError('All twelve checkpoints required')
    index={};remote_checkpoints={}
    for key,entry in rows.items():
        arm,seed,kind=key;folder=source/'evaluation'/('seed'+str(seed))/arm/kind
        for field in ('run','checkpoint','checkpoint_sha256','config_sha256','summary_sha256','step','training_commit'):
            if entry[field]!=planned[key][field]:raise ValueError('Frozen checkpoint identity differs')
        prov=read(folder/'provenance.json');metrics=read(folder/'metrics.json')
        if prov!=entry['provenance'] or metrics!=entry['metrics']:raise ValueError('Actual output metadata differs')
        if sha(folder/'predictions.npz')!=prov['prediction_sha256'] or not prov['source_checkpoint_unchanged']:
            raise ValueError('Actual prediction/checkpoint SHA differs')
        with np.load(folder/'predictions.npz',allow_pickle=False) as data:
            if data['paths'].shape!=(48,4,24,3) or data['gripper_open'].shape!=(48,4,24):raise ValueError('Full prediction budget required')
            if len(set(data['scene_ids']))!=48 or len(set(data['parent_ids']))!=12:raise ValueError('Complete identities required')
            if not np.isfinite(data['paths']).all() or not np.isfinite(data['gripper_open']).all():raise ValueError('Nonfinite saved predictions')
        for metric in ('semantic_goal_accuracy','ValidAtK'):
            if metrics[metric] is not None:raise ValueError('Unsupported task success claim')
        original=root/Path(entry['run'].split('/route_set_v1/',1)[1])/entry['checkpoint']
        local_hash=sha(original) if original.exists() else None
        if local_hash is not None and local_hash!=entry['checkpoint_sha256']:raise ValueError('Local original weight differs')
        remote_checkpoints[str(PurePosixPath(entry['run'])/entry['checkpoint'])]=dict(sha256=entry['checkpoint_sha256'],
            unchanged_server_proof=True,local_path=str(original) if original.exists() else None,local_sha256=local_hash)
        index[str(folder/'predictions.npz')]=sha(folder/'predictions.npz')
    for filename,expected in export['output_files_sha256'].items():
        if sha(archive/filename)!=expected:raise ValueError('Snapshot byte differs')
    samples=[json.loads(x) for x in (archive/'qwen_cache/samples.jsonl').read_text().splitlines()]
    if len(samples)!=48 or len({r['id'] for r in samples})!=48:raise ValueError('Full cache request count required')
    for row in samples:
        path=archive/'qwen_cache'/row['file']
        if sha(path)!=row['sha256']:raise ValueError('Cache feature byte differs')
        index[str(path)]=sha(path)
    deltas=[];grouped={}
    for kind in ('best','last'):
        for level,identity in (('per_task','task'),('per_parent','parent_id')):
            all_seed=[]
            for seed in (0,1,2):
                tables={arm:{r[identity]:r for r in rows[(arm,seed,kind)]['metrics']['task_parent_reference_metrics'][level]}
                        for arm in ('ordinary','event_supported')}
                if tables['ordinary'].keys()!=tables['event_supported'].keys():raise ValueError('Paired group identity differs')
                for name in sorted(tables['ordinary']):
                    left,right=[tables[arm][name] for arm in ('ordinary','event_supported')]
                    row=dict(checkpoint_kind=kind,seed=seed,level=level,identity=name,task=left['task'],
                        ordinary={k:left[k] for k in FIELDS},event_supported={k:right[k] for k in FIELDS},
                        aux_minus_ordinary={k:right[k]-left[k] for k in FIELDS})
                    deltas.append(row);all_seed.append(row)
            summaries=[]
            for name in sorted({r['identity'] for r in all_seed}):
                selected=sorted((r for r in all_seed if r['identity']==name),key=lambda r:r['seed'])
                summaries.append(dict(identity=name,task=selected[0]['task'],seeds=[r['seed'] for r in selected],
                    ordinary={k:stats([r['ordinary'][k] for r in selected]) for k in FIELDS},
                    event_supported={k:stats([r['event_supported'][k] for r in selected]) for k in FIELDS},
                    aux_minus_ordinary={k:stats([r['aux_minus_ordinary'][k] for r in selected]) for k in FIELDS},
                    seeds_with_lower_ADE=sum(r['aux_minus_ordinary']['candidate_matched_ADE_m']<0 for r in selected)))
            grouped[kind+'_'+level]=summaries
    jobs={name:read(source/(name+'.status.json')) for name in ('source_tests','export','qwen_cache','frozen_transfer')}
    if any(v['status']!='completed' or v['exit_code']!=0 for v in jobs.values()):raise ValueError('Incomplete pipeline stage')
    durations={name:(datetime.fromisoformat(row['end_utc'])-datetime.fromisoformat(row['start_utc'])).total_seconds() for name,row in jobs.items()}
    cache_status=read(archive/'qwen_cache/status.json');times=np.asarray([r['single_request_seconds'] for r in samples])
    cost=dict(job_wall_seconds=durations,gpu_hours_whole_cache_process=durations['qwen_cache']/3600,
        cache_internal_load_seconds=cache_status['load_seconds'],cache_internal_total_seconds=cache_status['total_seconds'],
        real_new_qwen_requests=48,qwen_request_median_seconds=float(np.median(times)),qwen_request_p95_seconds=float(np.percentile(times,95)),
        cache_peak_cuda_allocated_bytes=cache_status['cuda_peak_allocated_bytes'],
        shared_geometry_load_seconds=result['entries'][0]['provenance']['shared_preprocess_seconds'],
        cpu_checkpoint_evaluation_seconds=[dict(arm=r['arm'],seed=r['seed'],kind=r['checkpoint_kind'],seconds=r['provenance']['cpu_seconds']) for r in result['entries']],
        total_evaluated_candidate_path_states=12*48*4,per_request_candidates=4,
        warning='Shared Qwen cache and CPU head costs are separate; these are not end-to-end online request latency measurements.')
    analysis=dict(protocol='legacy12_saved_artifact_analysis_v1',source_script_sha256=sha(__file__),
        collection_denominators=result['collection_denominators'],aggregate=result['aggregate'],grouped=grouped,
        all_paired_task_parent_deltas=deltas,cost=cost,checkpoint_index=remote_checkpoints,
        actual_binary_artifacts_sha256=index,additional_training_steps=0,checkpoint_reselection=False,
        scope='All12 same-task unseen-parent DEV transfer; best and fixed1500 separately. No semantic/collision/execution validity measurement.')
    for path in source.rglob('*'):
        if path.is_file() and path.suffix in ('.json','.jsonl','.log'):
            dst=report/path.relative_to(source);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dst)
    for name in ('export_manifest.json','observations.jsonl','supervision.jsonl','attempts.jsonl','parent_inventory.json'):
        dst=report/'snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive/name,dst)
    for name in ('cache_config.json','status.json','samples.jsonl'):
        dst=report/'qwen_cache'/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(archive/'qwen_cache'/name,dst)
    write(report/'artifact_analysis.json',analysis)
    write(report/'artifact_index.json',dict(checkpoints=remote_checkpoints,ignored_local_binary_sha256=index,
        committed_metadata_sha256={str(p.relative_to(report)):sha(p) for p in report.rglob('*') if p.is_file()}))
    print(json.dumps(dict(status='passed',paired_groups=len(deltas),actual_predictions=12,new_cache_files=48,cost=cost)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','archive','report'):p.add_argument('--'+key,required=True)
    a=p.parse_args();analyze(a.root,a.archive,a.report)
