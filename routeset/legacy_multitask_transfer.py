"""DEV-only legacy transfer guards, separate from all historical snapshot gates."""
import json
from pathlib import Path
import ast
import numpy as np

from routeset.multitask_fingerprints import physical_layout, json_hash, audit_fingerprints
from scripts.snapshot_multitask_observations import child_path, digest

PROTOCOL = 'legacy_multitask_dev_transfer_export_v1'
CONFIG_PROTOCOL = 'multitask_legacy12_frozen_transfer_prospective_v1'
INPUT_KEYS = {'id','parent_id','split','image','instruction'}
FILES = {'observations.jsonl','supervision.jsonl','attempts.jsonl','parent_inventory.json'}
MODEL_SOURCES = ('routeset/observed_geometry.py','routeset/observed_route_head.py',
    'routeset/observed_multitask.py','scripts/train_observed_routes.py','routeset/common.py',
    'routeset/models.py','routeset/observed_path_refinement.py','scripts/observation_cache_qwen.py')
EVALUATION_FUNCTIONS = ('checkpoint_selection_score','add_tip_evaluation','read_geometry',
    'load_geometry','batch_inputs','evaluate')
LEGACY_SOURCE = {'collector_sha256':'f4f6a49acf48a3fb0a3ce6684f59641436168af28210a4b64bee975767038aaa',
    'restore_helper_sha256':'7d8230abc8ff642e511a35c1933df02ea2906bc8e5b926dfb028820e9302ddff',
    'expected_rlbench_revision':'02720bba4c73fe02eb75df946b8791b806028a9d',
    'expected_pyrep_revision':'8f420be8064b1970aae18a9cfbc978dfb15747ef'}


def validate_selection(config,plan):
    if config.get('protocol')!=CONFIG_PROTOCOL or plan.get('seed')!=281000:
        raise ValueError('Explicit original legacy registration required')
    selected=[row for row in plan['parents'] if row['split']=='DEV_MODEL']
    if (len(selected)!=12 or len({row['task'] for row in selected})!=6
            or any(row['parent_index'] not in (16,17) or row['requested_attempts']!=3 for row in selected)
            or selected!=config['selected_requested_parents']):
        raise ValueError('All and only the twelve original DEV parents required')
    return selected


def validate_checkpoint_plan(config):
    rows=config['requested_checkpoints']
    expected={(a,s,k) for a in ('ordinary','event_supported') for s in (0,1,2) for k in ('best','last')}
    if len(rows)!=12 or {(r['arm'],r['seed'],r['checkpoint_kind']) for r in rows}!=expected:
        raise ValueError('Exactly all twelve frozen checkpoint entries required')
    if config['train_steps']!=0 or config['checkpoint_reselection'] or config['hyperparameter_tuning']:
        raise ValueError('Frozen inference only; no training, selection or tuning')
    for row in rows:
        if row['checkpoint']!=row['checkpoint_kind']+'.pt' or not 0<row['step']<=1500:
            raise ValueError('Unregistered checkpoint identity')
        if row['checkpoint_kind']=='last' and row['step']!=1500:
            raise ValueError('Fixed last1500 required')
    return rows


def validate_cache_compatibility(cache, configs, manifest_sha256):
    """Only the observation manifest may change; feature production stays fixed."""
    if cache.get('manifest_sha256')!=manifest_sha256:
        raise ValueError('New Qwen cache does not match exported inputs')
    production={k:v for k,v in cache.items() if k!='manifest_sha256'}
    for config in configs:
        previous={k:v for k,v in config['cache_config'].items() if k!='manifest_sha256'}
        if previous!=production:raise ValueError('Qwen model/processor/preprocessing/runtime changed')
        if (config['pooling'],config['feature_dim'],config['pixel_stride'],config['geometry_pooling'])!=('both',4096,2,'spatial'):
            raise ValueError('Original feature pooling/dimension or geometry preprocessing required')
    return dict(production=production,pooling='both',mean_hidden_dimension=2048,
                last_hidden_dimension=2048,output_dimension=4096,pixel_stride=2)


def audit_model_source(project_root, runtime, config):
    version=config['code_commit']
    if len(version)!=40 or any(c not in '0123456789abcdef' for c in version):
        raise ValueError('Immutable model source required')
    recorded=Path(project_root)/'research_v2/releases'/version;runtime=Path(runtime)
    old={name:digest(recorded/name) for name in MODEL_SOURCES}
    current={name:digest(runtime/name) for name in MODEL_SOURCES}
    if old!=current:raise ValueError('Model/cache producer source differs from original training')
    trainer='scripts/train_observed_geometry.py'
    if digest(recorded/trainer)!=config['source_script_sha256']:
        raise ValueError('Original trainer source hash changed')
    def functions(path):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        return {node.name:ast.dump(node,include_attributes=False) for node in tree.body
                if isinstance(node,ast.FunctionDef) and node.name in EVALUATION_FUNCTIONS}
    left,right=functions(recorded/trainer),functions(runtime/trainer)
    if set(left)!=set(EVALUATION_FUNCTIONS) or left!=right:
        raise ValueError('Original evaluation/geometry functions changed')
    return dict(training_commit=version,model_cache_source_sha256=old,
                original_trainer_sha256=digest(recorded/trainer),runtime_trainer_sha256=digest(runtime/trainer),
                evaluation_functions_ast_identical=list(EVALUATION_FUNCTIONS))


def aggregate(entries):
    """Pair by explicit seed, never by the order of registered model rows."""
    indexed={(r['arm'],r['seed'],r['checkpoint_kind']):r for r in entries}
    expected={(a,s,k) for a in ('ordinary','event_supported') for s in (0,1,2) for k in ('best','last')}
    if len(entries)!=12 or set(indexed)!=expected:raise ValueError('Twelve unique paired metric entries required')
    result={}
    keys=('candidate_matched_ADE_m','candidate_endpoint_error_m','event_state_accuracy','event_sequence_accuracy')
    for kind in ('best','last'):
        result[kind]={}
        for key in keys:
            arms={arm:[indexed[(arm,seed,kind)]['metrics'][key] for seed in (0,1,2)]
                  for arm in ('ordinary','event_supported')}
            result[kind][key]={arm:dict(seeds=[0,1,2],values=values,mean=float(np.mean(values)),sample_std=float(np.std(values,ddof=1)))
                for arm,values in arms.items()}
            delta=np.asarray(arms['event_supported'])-np.asarray(arms['ordinary'])
            result[kind][key]['paired_aux_minus_ordinary']=dict(seeds=[0,1,2],values=delta.tolist(),mean=float(delta.mean()),sample_std=float(delta.std(ddof=1)))
    return result


def mechanical_gate(config):
    """Only old DEV reference world and new TRAIN/DEV saved mechanical hashes."""
    source=Path(config['source_dataset']).resolve();compare=Path(config['compare_dataset']).resolve()
    hashes={}
    def read(path,boundary=None):
        if boundary is None:
            try:path.relative_to(source);boundary=source
            except ValueError:boundary=compare
        path=child_path(boundary,path.relative_to(boundary))
        hashes[str(path)]=digest(path)
        return json.loads(path.read_text(encoding='utf-8'))
    plan=read(source/'partition_manifest.json')
    if hashes[str(source/'partition_manifest.json')]!=config['source_manifest_sha256']:
        raise ValueError('Legacy partition registration changed')
    selected=validate_selection(config,plan)
    if read(source/'source_manifest.json')!=LEGACY_SOURCE:
        raise ValueError('Legacy collector source changed')
    comparison=read(compare/'partition_manifest.json')
    if (comparison.get('seed')!=282000 or config['compare_roles']!=['TRAIN','DEV_MODEL']
            or hashes[str(compare/'partition_manifest.json')]!=config['compare_manifest_sha256']):
        raise ValueError('Explicit new TRAIN/DEV mechanical comparison required')
    rows=[]
    for spec in selected:
        folder=child_path(source/'DEV_MODEL'/'parents',spec['parent_id'])
        closure=read(folder/'closed.json',folder)
        if (folder/'worker.lock').exists():raise ValueError('Legacy DEV still running')
        if closure.get('requested_attempts')!=3 or closure.get('status') not in ('complete','setup_failed'):
            raise ValueError('Legacy DEV not fully closed')
        if closure['status']=='setup_failed':continue
        if closure.get('completed_attempts')!=3:raise ValueError('Missing registered attempt')
        pointer=read(folder/'reference_pointer.json',folder)
        ref=child_path(folder,Path(pointer['directory'])/'reference.json')
        metadata=read(ref,folder)
        if hashes[str(ref)]!=pointer['reference_sha256']:raise ValueError('Legacy reference changed')
        row={key:spec[key] for key in ('parent_id','task','split')}
        row.update(physical_layout_sha256=json_hash(physical_layout(metadata['world'])),
            physical_layout_quantized_sha256=json_hash(physical_layout(metadata['world'],True)),
            rgb_file_sha256=pointer['image_sha256'],source_root=str(source),derived_legacy=True)
        rows.append(row)
    for spec in comparison['parents']:
        if spec['split'] not in ('TRAIN','DEV_MODEL'):continue
        folder=child_path(compare/spec['split']/'parents',spec['parent_id'])
        path=folder/'mechanical_fingerprint.json'
        row=read(path,folder)
        if any(row.get(key)!=spec[key] for key in ('parent_id','task','split')):
            raise ValueError('Comparison mechanical identity changed')
        for key in ('physical_layout_sha256','physical_layout_quantized_sha256'):
            value=row.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Complete physical hash required')
        rows.append(dict(row,source_root=str(compare)))
    if sum(row['source_root']==str(compare) for row in rows)!=108:
        raise ValueError('All new96 TRAIN +12 DEV mechanical references required')
    result=audit_fingerprints(rows)
    # Same-role old/new DEV duplication also invalidates an unseen-parent claim.
    if result['duplicate_groups']:raise ValueError('Physical or RGB duplicate blocks independent-parent transfer')
    return dict(result,rows=rows,source_files_sha256=hashes,
        scope='Old DEV initial world only; new TRAIN/DEV mechanical metadata only. No locked per-parent files.')


def verify_export(directory,config):
    directory=Path(directory).resolve();manifest=json.loads((directory/'export_manifest.json').read_text())
    if manifest.get('protocol')!=PROTOCOL or manifest['registration']!=config:
        raise ValueError('Unrecognized or changed legacy transfer registration')
    if set(manifest['output_files_sha256'])!=FILES:raise ValueError('Incomplete export hashes')
    for name,expected in manifest['output_files_sha256'].items():
        if digest(directory/name)!=expected:raise ValueError('Export file changed: '+name)
    for name,expected in manifest['source_files_sha256'].items():
        if digest(name)!=expected:raise ValueError('Legacy committed source changed: '+name)
    gate=mechanical_gate(config)
    inputs=[json.loads(row) for row in (directory/'observations.jsonl').read_text().splitlines() if row.strip()]
    labels=[json.loads(row) for row in (directory/'supervision.jsonl').read_text().splitlines() if row.strip()]
    selected={row['parent_id'] for row in config['selected_requested_parents']}
    inventory=json.loads((directory/'parent_inventory.json').read_text())
    expected={row['parent_id'] for row in inventory if row['has_initial_observation']}
    if {row['parent_id'] for row in inventory}!=selected or len(inventory)!=12:
        raise ValueError('All twelve requested parents required including failures')
    if {row['parent_id'] for row in inputs}!=expected or len({r['id'] for r in inputs})!=len(inputs):
        raise ValueError('Observed parent denominator changed')
    if any(set(row)!=INPUT_KEYS or row['split']!='DEV_MODEL' or row['parent_id'] not in selected for row in inputs):
        raise ValueError('DEV-only observation whitelist required')
    if {r['id'] for r in inputs}!={r['id'] for r in labels} or len(labels)!=len(inputs):
        raise ValueError('Supervision join mismatch')
    identities={r['id']:r['parent_id'] for r in inputs}
    if any(r['parent_id']!=identities[r['id']] for r in labels):raise ValueError('Supervision parent identity mismatch')
    if any(r['split']!='DEV_MODEL' or r['semantic_targets'] is not None for r in labels):
        raise ValueError('No invented semantic labels or other roles')
    return manifest,gate
