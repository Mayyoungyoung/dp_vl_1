"""Metadata-only sampling and reference aggregation for variable task language.

Task IDs and parent IDs organize exposure/reporting. They are never condition
features and do not supply semantic, collision or task-success labels.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np


REFERENCE_FIELDS=('candidate_matched_ADE_m','reference_matched_ADE_m','candidate_endpoint_error_m',
                  'best_endpoint_error_m','event_state_accuracy','event_sequence_accuracy')


def draw_observation_batch(data,train_ids,sampler,batch_size,mode='uniform'):
    if mode=='uniform':
        return sampler.choice(train_ids,batch_size,replace=True)
    if mode!='task_parent_language':raise ValueError('unknown observation sampling mode')
    groups=defaultdict(lambda:defaultdict(list))
    for idx in train_ids:
        task=data['tasks'][idx]
        if not isinstance(task,str) or not task:raise ValueError('task metadata required for balanced sampling')
        groups[task][str(data['parent_ids'][idx])].append(int(idx))
    if not groups:raise ValueError('positive-reference TRAIN observations required')
    tasks=sorted(groups);out=[]
    for _ in range(batch_size):
        task=tasks[int(sampler.integers(len(tasks)))];parents=sorted(groups[task])
        languages=groups[task][parents[int(sampler.integers(len(parents)))]]
        out.append(languages[int(sampler.integers(len(languages)))])
    return np.asarray(out,dtype=np.int64)


def aggregate_task_parent_reference(result,rows,tasks):
    if len(rows)!=len(tasks) or any(not isinstance(task,str) or not task for task in tasks):
        raise ValueError('one explicit task metadata value per evaluated observation required')
    by_parent=defaultdict(list);parent_tasks={}
    for row,task in zip(rows,tasks):
        parent=row['parent_id']
        if parent_tasks.setdefault(parent,task)!=task:raise ValueError('parent assigned to more than one task')
        by_parent[(task,parent)].append(row)
        row['task']=task
    def mean(values):
        values=[value for value in values if value is not None]
        return float(np.mean(values)) if values else None
    parents=[]
    for (task,parent),items in sorted(by_parent.items()):
        parents.append(dict(task=task,parent_id=parent,language_examples=len(items),
            reference_language_examples=sum(row['reference_count']>0 for row in items),
            **{field:mean([row[field] for row in items]) for field in REFERENCE_FIELDS}))
    per_task=[]
    for task in sorted(set(tasks)):
        items=[row for row in parents if row['task']==task]
        per_task.append(dict(task=task,parents=len(items),reference_evaluable_parents=sum(row['candidate_matched_ADE_m'] is not None for row in items),
            **{field:mean([row[field] for row in items]) for field in REFERENCE_FIELDS}))
    result['instruction_weighted_reference_metrics']={field:result[field] for field in REFERENCE_FIELDS}
    result.update({field:mean([row[field] for row in per_task]) for field in REFERENCE_FIELDS})
    result.update(metric_aggregation='language-within-parent, parent-within-task, macro-across-evaluable-tasks',
        evaluated_tasks=len(per_task),reference_evaluable_tasks=sum(row['candidate_matched_ADE_m'] is not None for row in per_task),
        reference_evaluable_parents=sum(row['candidate_matched_ADE_m'] is not None for row in parents),
        task_parent_reference_metrics=dict(per_parent=parents,per_task=per_task),
        unavailable_parent_policy='parents without observed inputs remain in snapshot collection accounting; no fabricated predictions',
        unsupported_task_metrics=dict(semantic_goal_accuracy=None,collision_validity=None,execution_success=None,
            task_success=None,UniqueValidAtK=None,ReferenceCoverageAtK=None))


def validate_multitask_resume(current,saved):
    for key,default in (('sampling_mode','uniform'),('metric_aggregation','instruction')):
        if current.get(key,default)!=saved.get(key,default):raise ValueError('resume config mismatch: '+key)


def check_multitask_model_gate(observations,supervision,manifest_path=None):
    """Recheck saved mechanical hashes before opening any model data.

    Historical reach datasets have no such manifest and are unaffected. A
    discovered six-task snapshot cannot silently bypass its live gate. The
    audit implementation reads mechanical metadata across roles, never their
    images, routes, target descriptions or outcome labels.
    """
    observations,supervision=Path(observations).resolve(),Path(supervision).resolve()
    discovered=observations.parent/'snapshot_manifest.json'
    if manifest_path is None and not discovered.exists():return None
    path=Path(manifest_path).resolve() if manifest_path is not None else discovered
    manifest=json.loads(path.read_text(encoding='utf-8'))
    declares_multitask=(str(manifest.get('protocol','')).startswith('multitask_')
        or 'current_gate_required_before_model_use' in manifest)
    # Learning-curve snapshots predate this gate and use the same basename.
    # They must keep the historical path unless explicitly presented as a new
    # multitask snapshot. A damaged/disabled new gate remains a hard failure.
    if manifest_path is None and not declares_multitask:return None
    if path!=discovered or supervision.parent!=observations.parent:
        raise ValueError('multitask snapshot manifest must accompany both input manifests')
    if manifest.get('protocol')!='multitask_closed_prefix_snapshot_v1' or manifest.get('current_gate_required_before_model_use') is not True:
        raise ValueError('unrecognized or disabled multitask model-use gate')
    selected=manifest['selected_requested_parents']
    if not selected or any(row['split'] not in ('TRAIN','DEV_MODEL') for row in selected):
        raise ValueError('only pre-registered TRAIN/DEV_MODEL parents may enter this baseline')
    selected_ids={row['parent_id']:row for row in selected}
    if len(selected_ids)!=len(selected):raise ValueError('duplicate requested parent in snapshot')
    def digest(filename):return hashlib.sha256(Path(filename).read_bytes()).hexdigest()
    if observations.name!='observations.jsonl' or supervision.name!='supervision.jsonl':
        raise ValueError('use the original snapshot observation/supervision manifests')
    snapshot_hashes=manifest.get('snapshot_files_sha256',{})
    if set(snapshot_hashes)!={'observations.jsonl','supervision.jsonl','attempts.jsonl','parent_inventory.json'}:
        raise ValueError('complete immutable snapshot artifact hashes required')
    for filename,expected in snapshot_hashes.items():
        file_path=observations.parent/filename
        if file_path.parent!=observations.parent or digest(file_path)!=expected:
            raise ValueError('snapshot artifact changed: '+filename)
    inputs=[json.loads(line) for line in observations.read_text(encoding='utf-8').splitlines() if line.strip()]
    labels=[json.loads(line) for line in supervision.read_text(encoding='utf-8').splitlines() if line.strip()]
    if {row['id'] for row in inputs}!={row['id'] for row in labels}:raise ValueError('snapshot input/supervision identity mismatch')
    for row in inputs+labels:
        parent=selected_ids.get(row['parent_id'])
        if parent is None or row['split']!=parent['split']:raise ValueError('input parent is outside registered development snapshot')
        if 'task' in row and row['task']!=parent['task']:raise ValueError('snapshot task metadata mismatch')
    if any(row.get('semantic_targets') is not None for row in labels):
        raise ValueError('generic multitask snapshot must not invent semantic target labels')
    sources=[Path(manifest['source_dataset']).resolve()]+[Path(value).resolve() for value in manifest['compare_sources']]
    if len(sources)<2 or len(set(sources))!=len(sources):raise ValueError('explicit distinct cross-batch sources required')
    # A previously observed selected parent may not regress to legacy/unknown
    # metadata. Do not impose this on uncollected parents or old comparison
    # corpora, for which unknown mechanical coverage remains explicit.
    selected_mechanical={}
    for parent_id in sorted({row['parent_id'] for row in inputs}):
        parent=selected_ids[parent_id]
        mechanical_path=sources[0]/parent['split']/'parents'/parent_id/'mechanical_fingerprint.json'
        if not mechanical_path.is_file():raise ValueError('selected observed parent has no current mechanical fingerprint: '+parent_id)
        mechanical=json.loads(mechanical_path.read_text(encoding='utf-8'))
        if any(mechanical.get(key)!=parent[key] for key in ('parent_id','split','task')):
            raise ValueError('selected mechanical identity mismatch: '+parent_id)
        if mechanical.get('protocol')!='multitask_initial_layout_fingerprint_v1':
            raise ValueError('selected observed parent requires a complete physical fingerprint: '+parent_id)
        for key in ('physical_layout_sha256','physical_layout_quantized_sha256'):
            value=mechanical.get(key)
            if not isinstance(value,str) or len(value)!=64 or any(char not in '0123456789abcdef' for char in value):
                raise ValueError('selected observed parent requires a complete physical fingerprint: '+parent_id)
        actual=digest(mechanical_path)
        if actual!=manifest.get('source_files_sha256',{}).get(str(mechanical_path)):
            raise ValueError('selected mechanical fingerprint changed since snapshot: '+parent_id)
        selected_mechanical[str(mechanical_path)]=actual
    from scripts.audit_multitask_layout_hashes import audit
    current=audit(sources)
    tasks=sorted({row['task'] for row in selected})
    statuses=current['task_usage_status']
    if any(statuses.get(task,'unverified_no_mechanical_metadata').startswith('blocked_') for task in tasks):
        raise ValueError('current mechanical layout gate blocks selected snapshot tasks')
    return dict(checked_at=datetime.now(timezone.utc).isoformat(),snapshot_manifest=str(path),
        snapshot_manifest_sha256=digest(path),observation_manifest_sha256=digest(observations),
        supervision_manifest_sha256=digest(supervision),selected_tasks=tasks,current_audit=current,
        selected_mechanical_files_sha256=selected_mechanical,
        collection_denominators={key:manifest[key] for key in ('requested_parents','requested_attempts',
            'actual_attempt_records','setup_failed_parents','parents_with_observation','parents_with_zero_reference',
            'successful_reference_routes')},
        scope='mechanical cross-role audit before model data loading; only snapshot TRAIN/DEV_MODEL observations enter model')
