"""Fixed original best36 per arm after12000: real Qwen, no new selection."""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
from types import SimpleNamespace

from scripts import evaluate_observed_two_row_online as online

PROTOCOL='two_row_constant_no_direct12000_best_online_v1'
ROOT=Path(__file__).resolve().parents[1]
REVISION=online.REVISION
INPUT_KEYS=online.INPUT_KEYS
DEV_IDS=online.DEV_IDS
DEPENDENCIES=online.DEPENDENCIES
(digest,head_options,select_observations,input_paths,verify_training_artifacts)=(
    online.digest,online.head_options,online.select_observations,online.input_paths,online.verify_training_artifacts)
ARMS={
 'constant':dict(run='runs/observed_two_row_prefix76_convergence_v1/peak_seed0',
     source='1417cdc802674a00e70d4ced3181c6a6ede2a99e',protocol='ordinary_two_row_prefix76_constant_lr_12000_v1',
     config_key='convergence_protocol',receipt='convergence_result_receipt.json',parameters=1231965),
 'no_direct':dict(run='runs/observed_two_row_prefix76_no_direct_v1/peak_seed0',
     source='1a3eef1fb12d55e98d4d188a091ea40ea62c0a02',protocol='ordinary_two_row_prefix76_no_direct_constant12000_v1',
     config_key='no_direct_protocol',receipt='no_direct_result_receipt.json',parameters=682845)}
EXPORT_SHA='309966e192a3c582cde503151e607b5ede679eaec596bd8ab2bc1fd06ffdbab1'


def validate_training_identity(arm,run,config,summary,status):
    if arm not in ARMS:raise ValueError('Only two predeclared ordinary arms')
    spec=ARMS[arm]
    if (status.get('status')!='completed' or status.get('exit_code')!=0 or status.get('step')!=12000
            or config.get(spec['config_key'])!=spec['protocol'] or config.get('code_commit')!=spec['source']
            or config.get('steps')!=12000 or config.get('eval_every')!=250 or config.get('lr')!=.0003
            or config.get('batch_size')!=32 or config.get('seed')!=0 or config.get('train_examples')!=189
            or config.get('dev_examples')!=36 or config.get('two_row_export_sha256')!=EXPORT_SHA
            or summary.get('last_step')!=12000 or summary.get('parameters')!=spec['parameters']
            or summary.get('trajectory_exposures')!=1536000):
        raise ValueError('Fixed completed12000 arm/config/exposure required')
    receipt_path=Path(run).parent/spec['receipt'];value=json.loads(receipt_path.read_text())
    if (value.get('protocol')!=spec['protocol'] or value.get('summary_sha256')!=digest(Path(run)/'summary.json')
            or value['checkpoint_sha256']['best.pt']!=summary['best_checkpoint_sha256']
            or value['checkpoint_sha256']['best.pt']!=digest(Path(run)/'best.pt')
            or value['checkpoint_sha256']['last.pt']!=digest(Path(run)/'last.pt')):
        raise ValueError('Completed training receipt does not bind unchanged original best/last')
    if arm=='constant' and summary['best_checkpoint_sha256']!='6b99a2171d5f241d879797371b5a9a406228e48c009db7554e7bc186d69b926d':
        raise ValueError('Original constant best identity changed')
    if arm=='no_direct' and (not value.get('passed') or not value.get('shared_initialization_exact')):
        raise ValueError('Passing no-direct completion/paired stream required')
    history=json.loads((Path(run)/'history.json').read_text())
    if [row['step'] for row in history]!=list(range(250,12001,250)):
        raise ValueError('Exactly48 original model-selection opportunities required')
    if summary['best_step'] not in [row['step'] for row in history]:raise ValueError('Original best step not in sealed history')
    return dict(path=str(receipt_path),sha256=digest(receipt_path),protocol=value['protocol'],
        checkpoint_sha256=value['checkpoint_sha256'],original_best_step=summary['best_step'],
        selection_opportunities=48,reselection=False)


def preflight(args, arm):
    from scripts.export_two_row_observations import verify_export, selection_guard
    data, run, source = args.data.resolve(), args.run.resolve(), args.training_source.resolve()
    config = json.loads((run/'config.json').read_text()); summary = json.loads((run/'summary.json').read_text())
    status = json.loads((run/'status.json').read_text())
    completion = validate_training_identity(arm, run, config, summary, status)
    head_options(config)
    if config.get('two_row_driver_protocol') != 'ordinary_two_row_frozen_qwen_saturation_v1': raise ValueError('Two-row driver required')
    if source.name != config['code_commit'] or not re.fullmatch('[0-9a-f]{40}', source.name): raise ValueError('Explicit actual immutable training source required')
    sources = {}
    dependencies = DEPENDENCIES + ('scripts/evaluate_observed_two_row_online.py',) + (('routeset/observed_geometry_no_direct.py',) if arm == 'no_direct' else ())
    for filename in dependencies:
        current, original = ROOT/filename, source/filename
        if digest(current) != digest(original): raise ValueError('Training source dependency differs: '+filename)
        sources[filename] = digest(current)
    for key, filename in [('source_script_sha256','scripts/train_observed_geometry.py'),('two_row_driver_sha256','scripts/train_observed_two_row.py')]:
        if config[key] != sources[filename]: raise ValueError('Training configuration source SHA differs')
    if arm == 'no_direct' and config['no_direct_model_sha256'] != sources['routeset/observed_geometry_no_direct.py']:
        raise ValueError('No-direct model source SHA differs')
    for field, filename in [('observations','observations.jsonl'),('supervision','supervision.jsonl'),('cache_dir','qwen_cache')]:
        if Path(config[field]).resolve() != data/filename: raise ValueError('Run/export path differs: '+field)
    manifest, gate = verify_export(data)  # Mechanical metadata and integrity bytes, no label-array deserialization.
    selection_guard(manifest['selection'])
    if config['two_row_export_sha256'] != digest(data/'export_manifest.json'): raise ValueError('Training export changed')
    rows = select_observations([json.loads(line) for line in (data/'observations.jsonl').read_text().splitlines() if line.strip()],
        manifest['selection']['requested_parents']['TRAIN'])
    input_hashes = {}
    for row in rows.values():
        for path in input_paths(row,manifest['source_files_sha256']): input_hashes[str(path)] = digest(path)
    cache = json.loads((data/'qwen_cache/cache_config.json').read_text())
    if cache != config['cache_config']: raise ValueError('Actual Qwen cache configuration changed')
    if (cache['model'] != 'Qwen/Qwen3-VL-2B-Instruct' or cache['revision'] != REVISION or cache['processor'] != REVISION
            or cache['model_trainable_parameter_count'] != 0 or cache['manifest_sha256'] != digest(data/'observations.jsonl')
            or set(cache['input_contract']) != INPUT_KEYS or cache['dtype'] != 'torch.bfloat16'):
        raise ValueError('Pinned frozen real Qwen cache required')
    provenance = json.loads((args.model/'provenance.json').read_text())
    if provenance.get('model_id') != cache['model'] or provenance.get('revision') != REVISION or not provenance.get('all_hashes_verified'):
        raise ValueError('Official Qwen provenance mismatch')
    if digest(run/'best.pt') != summary['best_checkpoint_sha256']: raise ValueError('Fixed best checkpoint bytes changed')
    artifact_receipt = verify_training_artifacts(run, config)
    training_sources = json.loads((run/'source_hashes.json').read_text())
    # Hashes establish provenance; reference arrays themselves are never loaded.
    for path, sha in training_sources.items():
        if digest(path) != sha: raise ValueError('Original training source changed: '+path)
    receipt = dict(protocol=PROTOCOL, run=str(run), data=str(data), training_source=str(source),
        source_files_sha256=dict(sources, **{'scripts/evaluate_two_row_12000_online.py':digest(__file__)}),
        checkpoint_sha256=summary['best_checkpoint_sha256'], checkpoint_step=summary['best_step'],
        training_artifact_receipt=artifact_receipt,
        config_sha256=digest(run/'config.json'), training_summary_sha256=digest(run/'summary.json'),
        export_manifest_sha256=digest(data/'export_manifest.json'), model_provenance_sha256=digest(args.model/'provenance.json'),
        observed_input_hashes=input_hashes, live_gate_before=gate, requested_input_ids=DEV_IDS,
        available_input_ids=[i for i in DEV_IDS if i in rows], requested_requests=36, requested_candidates=144,
        arm=arm, completed_training_receipt=completion, training_steps=12000, original_best_step=summary['best_step'],
        no_labels_in_forward=True, no_cached_features_in_forward=True, checkpoint_selection='unchanged saved best only',
        authorized_gpu='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab', memory_fraction=.35, cpu_threads=1)
    return config, summary, manifest, rows, input_hashes, training_sources, receipt


@contextmanager
def entry_adapter(arm, geometry_module, model_class):
    """Reuse the unchanged request/generation/sealing/check loop, restore on exit."""
    original_preflight=online.preflight
    original_class=geometry_module.ObservedGeometryRouteHead
    online.preflight=lambda args:preflight(args,arm)
    geometry_module.ObservedGeometryRouteHead=model_class
    try:yield
    finally:
        online.preflight=original_preflight
        geometry_module.ObservedGeometryRouteHead=original_class


def add_exact_saved_comparison(output, run):
    """Only saved arrays; exact feature bytes and every decision field reported."""
    import hashlib
    import numpy as np
    output,run=Path(output),Path(run)
    config=json.loads((run/'config.json').read_text())
    source_hashes=json.loads((run/'source_hashes.json').read_text())
    original=json.loads((output/'cache_comparison.json').read_text())
    rows={r['id']:r for r in map(json.loads,Path(config['observations']).read_text().splitlines()) if r['split']=='DEV_MODEL'}
    reports=[]
    for item in original['per_scene']:
        identifier=item['id'];row=rows[identifier]
        key=hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()[:20]
        cache=online.checked_file(Path(config['cache_dir'])/(key+'.npz'),source_hashes)
        predicted=output/'requests'/(identifier+'.npz')
        seal=json.loads(predicted.with_suffix('.seal.json').read_text())
        if digest(predicted)!=seal['prediction_sha256']:raise ValueError('Original online prediction seal changed')
        with np.load(cache,allow_pickle=False) as a,np.load(predicted,allow_pickle=False) as b:
            exact={name:bool(a[name].dtype==b[name].dtype and a[name].shape==b[name].shape and a[name].tobytes()==b[name].tobytes())
                for name in ('mean_hidden','last_hidden','input_tokens')}
        reports.append(dict(item,feature_exact_bytes=exact,online_prediction_sha256=seal['prediction_sha256']))
    result=dict(protocol=PROTOCOL,additional_forward_requests=0,additional_qwen_encodings=0,
        examples=len(reports),all_feature_bytes_exact=all(all(r['feature_exact_bytes'].values()) for r in reports),
        all_candidate_decisions_identical=all(r.get('candidate_decisions_identical',False) for r in reports),
        per_scene=reports,scope='All mismatches retained without retry, repair or checkpoint reselection; byte identity is tested, not presumed.')
    online.write_json(output/'exact_saved_comparison.json',result)
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=tuple(ARMS),required=True)
    parser.add_argument('--project',type=Path,default=Path('/home/wzy/dpvlm/route_set_v1'))
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args(argv)
    project=args.project.resolve();spec=ARMS[args.arm]
    if hasattr(os,'sched_getaffinity') and len(os.sched_getaffinity(0))!=1:
        raise ValueError('One authorized CPU affinity required')
    if args.validate_only:
        args.output.mkdir(parents=True,exist_ok=False)
        _,_,_,_,_,_,receipt=preflight(SimpleNamespace(run=project/spec['run'],
            data=project/'data/observation_two_row_prefix76_v1',model=args.model,
            training_source=project/'research_v2/releases'/spec['source']),args.arm)
        online.write_json(args.output/'preflight.json',receipt)
        online.write_json(args.output/'status.json',dict(status='metadata_preflight_only',gpu_executed=False))
        return 0
    from routeset import observed_geometry as geometry_module
    if args.arm=='no_direct':
        from routeset.observed_geometry_no_direct import ObservedGeometryNoDirectHead
        model_class=ObservedGeometryNoDirectHead
    else:model_class=geometry_module.ObservedGeometryRouteHead
    values=['--run',str(project/spec['run']),'--data',str(project/'data/observation_two_row_prefix76_v1'),
        '--model',str(args.model),'--training-source',str(project/'research_v2/releases'/spec['source']),
        '--output',str(args.output)]
    with entry_adapter(args.arm,geometry_module,model_class):code=online.main(values)
    try:
        comparison=add_exact_saved_comparison(args.output,project/spec['run'])
        online.write_json(args.output/'paired12000_online_receipt.json',dict(protocol=PROTOCOL,arm=args.arm,
            source_sha256=digest(__file__),summary_sha256=digest(args.output/'summary.json'),
            exact_saved_comparison_sha256=digest(args.output/'exact_saved_comparison.json'),
            requested_requests=36,requested_candidate_slots=144,paired_total_requests=72,paired_total_candidate_slots=288,
            all_feature_bytes_exact=comparison['all_feature_bytes_exact'],
            all_candidate_decisions_identical=comparison['all_candidate_decisions_identical'],
            source_training_cost_counted_again=False,same_training_budget=True,same_best_step_claim=False,
            generation_exit_code=code,additional_consistency_forward_requests=0))
    except BaseException as error:
        online.write_json(args.output/'postseal_comparison_failure.json',dict(error=repr(error),
            new_forward_requests=0,retry=False,preserve_original_online_summary_and_predictions=True))
        raise
    return code


if __name__=='__main__':raise SystemExit(main())
