"""Audit synchronized prefix24/prefix60 artifacts without re-evaluating labels."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    values=[json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines()]
    result={value['id']:value for value in values}
    if len(result)!=len(values):raise ValueError('Duplicate observation id')
    return result


def run(root, output, expanded_prefix=60):
    if expanded_prefix not in (60,108):raise ValueError("Only registered prefix60 or prefix108 comparisons supported")
    root,output=Path(root),Path(output)
    if output.exists():raise FileExistsError('Fresh analysis output required')
    summaries={};configs={};artifacts={};indices={};observations={};supervision={};sources={}
    for prefix in (24,expanded_prefix):
        reports=root/'reports'/f'observed_multitask_prefix{prefix}_v1'
        run_path=root/'runs'/f'observed_multitask_prefix{prefix}_v1/free_offset_seed0'
        snapshot=root/'reports'/f'observation_multitask_prefix{prefix}_v1/snapshot'
        cfg=read(reports/'free_offset_seed0/config.json');summary=read(reports/'free_offset_seed0/summary.json')
        index=read(reports/'free_offset_seed0/source_hashes.json')
        obs=rows(snapshot/'observations.jsonl');sup=rows(snapshot/'supervision.jsonl')
        if set(obs)!=set(sup):raise ValueError('Observation/supervision identity mismatch')
        for filename in ('observations.jsonl','supervision.jsonl'):
            key=str(cfg['observations'] if filename=='observations.jsonl' else cfg['supervision'])
            if sha(snapshot/filename)!=index[key]:raise ValueError('Snapshot hash differs from actual model source index')
        binaries={'best.pt':'best_checkpoint_sha256','last.pt':'last_checkpoint_sha256',
                  'dev_model/predictions.npz':'prediction_sha256','last_dev_model/predictions.npz':'last_prediction_sha256'}
        hashes={name:sha(run_path/name) for name in binaries}
        if any(hashes[name]!=summary[key] for name,key in binaries.items()):raise ValueError('Actual checkpoint/prediction hash mismatch')
        samples=rows(reports/'qwen_cache_artifacts/samples.jsonl')
        for key,row in obs.items():
            cache_name=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]+'.npz'
            cache_path=cfg['cache_dir']+'/'+cache_name
            if samples[key]['file']!=cache_name or samples[key]['sha256']!=index[cache_path]:
                raise ValueError('Actual Qwen producer/consumer feature mismatch')
        sources[prefix]={name.split('/releases/')[1].split('/',1)[1]:value for name,value in read(reports/'source_hashes.json').items() if '/releases/' in name}
        summaries[prefix]=summary;configs[prefix]=cfg;indices[prefix]=index
        observations[prefix]=obs;supervision[prefix]=sup;artifacts[prefix]=hashes
    dev={prefix:{key:row for key,row in obs.items() if row['split']=='DEV_MODEL'} for prefix,obs in observations.items()}
    if dev[24]!=dev[expanded_prefix] or len(dev[24])!=48:raise ValueError('DEV observations changed')
    if {key:supervision[24][key] for key in dev[24]}!={key:supervision[expanded_prefix][key] for key in dev[expanded_prefix]}:
        raise ValueError('DEV reference/event metadata changed')
    dev_hashes={};cache_hashes={}
    for key,row in dev[24].items():
        label=supervision[24][key]
        for path in [row['image'],label['observation']]+label['routes']:
            old={**indices[24],**configs[24]['geometry_preprocessing']['source_hashes']}[path]
            new={**indices[expanded_prefix],**configs[expanded_prefix]['geometry_preprocessing']['source_hashes']}[path]
            if old!=new:raise ValueError('DEV source file changed')
            dev_hashes[path]=old
        cache_name=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]+'.npz'
        values=[indices[p][configs[p]['cache_dir']+'/'+cache_name] for p in (24,expanded_prefix)]
        if values[0]!=values[1]:raise ValueError('Shared DEV cached Qwen bytes differ')
        cache_hashes[key]=values[0]
    source_shared=set(sources[24])&set(sources[expanded_prefix])
    if any(sources[24][name]!=sources[expanded_prefix][name] for name in source_shared):raise ValueError('Shared source hashes differ')
    keys=('steps','batch_size','candidates','horizon','width','depth','pooling','geometry_pooling','anchor_mode',
          'endpoint_mode','sampling_mode','metric_aggregation','refinement_mode','checkpoint_selection','point_width',
          'pixel_stride','endpoint_residual_bound','grounding_weight','grounding_sigma','event_scale','lr','seed',
          'eval_every','feature_dim','source_script_sha256','checkpoint_selection_protocol','evaluation_protocol')
    matched={key:configs[24][key] for key in keys}
    if matched!={key:configs[expanded_prefix][key] for key in keys}:raise ValueError('Architecture/training/evaluation settings differ')
    metrics={};parents={};tasks={};retrieval_artifacts={}
    for prefix in (24,expanded_prefix):
        summary=summaries[prefix]
        retrieval=read(root/'reports'/f'observation_retrieval_prefix{prefix}_v1/dev_model/summary.json')
        pred=root/'runs'/f'observation_retrieval_prefix{prefix}_v1/dev_model/predictions.npz'
        if sha(pred)!=retrieval['prediction_sha256']:raise ValueError('Retrieval prediction hash mismatch')
        retrieval_artifacts[prefix]={'prediction_sha256':sha(pred),'summary':retrieval}
        for tag,metric in [('best',summary['metrics']),('last',summary['last_metrics']),('train_best',summary['train_metrics']),('retrieval',retrieval['metrics'])]:
            name=f'prefix{prefix}_{tag}'
            metrics[name]={key:value for key,value in metric.items() if key not in ('task_parent_reference_metrics','instruction_weighted_reference_metrics')}
            parents[name]={row['parent_id']:row for row in metric['task_parent_reference_metrics']['per_parent']}
            tasks[name]={row['task']:row for row in metric['task_parent_reference_metrics']['per_task']}
    comparisons=[]
    for parent in sorted(parents['prefix24_best']):
        a,b=parents['prefix24_best'][parent],parents[f'prefix{expanded_prefix}_best'][parent]
        comparisons.append(dict(parent_id=parent,task=a['task'],
            prefix24_best_ADE_m=a['candidate_matched_ADE_m'],expanded_best_ADE_m=b['candidate_matched_ADE_m'],
            delta_ADE_m=b['candidate_matched_ADE_m']-a['candidate_matched_ADE_m'],
            prefix24_endpoint_error_m=a['candidate_endpoint_error_m'],expanded_endpoint_error_m=b['candidate_endpoint_error_m'],
            expanded_retrieval_ADE_m=parents[f'prefix{expanded_prefix}_retrieval'][parent]['candidate_matched_ADE_m']))
    result=dict(protocol='same_dev_multitask_training_parent_learning_curve_v1',status='passed',
        source_script_sha256=sha(__file__),expanded_prefix=expanded_prefix,matched_training_options=matched,
        same_dev=dict(instructions=len(dev[24]),parents=len({x['parent_id'] for x in dev[24].values()}),
            observations_exactly_equal=True,supervision_exactly_equal=True,source_files_sha256=dev_hashes,
            qwen_feature_files_sha256=cache_hashes),shared_actual_source_sha256={name:sources[24][name] for name in sorted(source_shared)},
        actual_artifacts=artifacts,retrieval_artifacts=retrieval_artifacts,metrics=metrics,per_task=tasks,
        per_parent=parents,paired_dev_parents=comparisons,
        better_ADE_parents=sum(x['delta_ADE_m']<0 for x in comparisons),worse_ADE_parents=sum(x['delta_ADE_m']>0 for x in comparisons),
        cost={str(prefix):{key:summaries[prefix][key] for key in ('elapsed_s','data_load_preprocess_s','gpu_hours_reserved','parameters','trajectory_exposures','peak_cuda_memory_mb','best_step','last_step')} for prefix in (24,expanded_prefix)},
        scope='Single-seed same-developed-DEV data-size control, same 1500x32 exposure and K4. No semantic, collision or robot execution certification; task IDs only sampler/report metadata. Initialization reconstruction is a separate CPU proof, not a saved historical checkpoint.')
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(status='passed',parents_better=result['better_ADE_parents'],parents_worse=result['worse_ADE_parents'],dev_source_files=len(dev_hashes),same_dev_cache_files=len(cache_hashes))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--expanded-prefix',type=int,choices=(60,108),default=60)
    args=parser.parse_args();run(args.root,args.output,args.expanded_prefix)
