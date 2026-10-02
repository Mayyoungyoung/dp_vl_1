"""Train-only landmark selection for the existing observed spatial attention.

No labels are supplied to the model forward. Event/endpoint proximity is an
unlabelled observed-surface proxy, not target identity or contact certification.
"""
import hashlib
import json
from pathlib import Path
import numpy as np

PROTOCOL='positive_event_or_endpoint_nearest_support_v1'
MODES=('endpoint','event_supported')


def choose_landmark(path,opened,world_xyz,valid_mask):
    """First close-after vs endpoint; exact ties and no close choose endpoint."""
    path,opened=np.asarray(path),np.asarray(opened)
    points=np.asarray(world_xyz)[np.asarray(valid_mask,dtype=bool)]
    if path.ndim!=2 or path.shape[1]!=3 or len(path)<2 or opened.shape!=(len(path),):raise ValueError('Invalid positive route/event')
    if not len(points) or points.ndim!=2 or points.shape[1]!=3:raise ValueError('Valid observed points required')
    if not np.isfinite(path).all() or not np.isfinite(opened).all() or not np.isfinite(points).all():raise ValueError('Nonfinite positive or observed support')
    distance=lambda point:float(np.linalg.norm(points.astype(np.float64)-np.asarray(point,dtype=np.float64),axis=1).min())
    states=opened>.5;closes=np.flatnonzero(states[:-1]&~states[1:])+1
    end=path[-1];end_distance=distance(end);selected=end.copy();kind='endpoint';close=None;close_distance=None;index=None
    if len(closes):
        index=int(closes[0]);close=path[index];close_distance=distance(close)
        if close_distance<end_distance:selected=close.copy();kind='first_close_after'
    return selected,dict(selected_kind=kind,selected_xyz=selected.tolist(),
        selected_nearest_observed_m=close_distance if kind=='first_close_after' else end_distance,
        endpoint_xyz=end.tolist(),endpoint_nearest_observed_m=end_distance,first_close_index=index,
        first_close_after_xyz=close.tolist() if close is not None else None,first_close_nearest_observed_m=close_distance,
        tie_policy='endpoint',no_threshold_or_reference_filter=True)


def prepare_event_grounding_targets(data,geometry,train_ids):
    """Only positive TRAIN references are accessed; repeated paraphrases reuse targets.

    This optional mode is for the sealed generic multitask corpus, whose parent
    paraphrases share one reference set. Refuse ambiguous reuse across goals.
    """
    ids=np.asarray(train_ids,dtype=int)
    if ids.ndim!=1 or len(set(map(int,ids)))!=len(ids) or np.any(ids<0) or np.any(ids>=len(data['splits'])):
        raise ValueError('Unique in-range TRAIN observation ids required')
    expected=np.flatnonzero((data['splits']=='TRAIN')&data['path_mask'].any(1))
    if not np.array_equal(ids,expected):raise ValueError('All and only positive TRAIN examples required; never inspect DEV targets')
    target=np.zeros((len(data['splits']),data['paths'].shape[1],3),dtype=data['paths'].dtype)
    records=[];seen={};counts={};positive_slots=0
    for idx in ids:
        parent=str(data['parent_ids'][idx]);mask=data['path_mask'][idx];gidx=int(geometry['index'][idx])
        if parent in seen:
            first=seen[parent]
            if not (np.array_equal(mask,data['path_mask'][first]) and np.array_equal(data['paths'][idx][mask],data['paths'][first][mask])
                    and np.array_equal(data['events'][idx][mask],data['events'][first][mask]) and gidx==int(geometry['index'][first])):
                raise ValueError('Event-supported generic paraphrases must share the exact positive set/current observation')
            target[idx]=target[first];positive_slots+=int(mask.sum());continue
        seen[parent]=int(idx)
        task=data['tasks'][idx] # Reporting only; never used in point choice.
        counts.setdefault(task,dict(parents=0,positive_reference_routes=0,first_close_after=0,endpoint=0))['parents']+=1
        for ref in np.flatnonzero(mask):
            chosen,record=choose_landmark(data['paths'][idx,ref],data['events'][idx,ref],
                geometry['points']['world_xyz'][gidx],geometry['points']['valid_mask'][gidx])
            target[idx,ref]=chosen;record.update(parent_id=parent,task=task,reference_index=int(ref),representative_train_observation=str(data['scene_ids'][idx]))
            records.append(record);counts[task]['positive_reference_routes']+=1;counts[task][record['selected_kind']]+=1
        positive_slots+=int(mask.sum())
    serial=dict(protocol=PROTOCOL,data_geometry_fingerprint=geometry['fingerprint'],records=records,
        source_helper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    fingerprint=hashlib.sha256(json.dumps(serial,sort_keys=True,allow_nan=False).encode()).hexdigest()
    metadata=dict(**serial,target_fingerprint=fingerprint,positive_train_observations=len(ids),positive_train_parents=len(seen),
        positive_reference_observation_slots=positive_slots,unique_parent_reference_slots=len(records),per_task=counts,
        zero_reference_train_observations=int(((data['splits']=='TRAIN')&~data['path_mask'].any(1)).sum()),
        reference_filtering=False,dev_labels_read=False,
        scope='TRAIN target for existing attention only; nearest unlabelled point may be robot/background. No task ID, positive path, event or chosen landmark enters inference forward; no new model parameters.')
    return target,metadata


def validate_grounding_target_resume(current,saved):
    mode=current.get('grounding_target','endpoint')
    if mode not in MODES or mode!=saved.get('grounding_target','endpoint'):raise ValueError('resume config mismatch: grounding_target')
    if mode=='event_supported':
        fingerprint=current.get('grounding_target_fingerprint')
        if not fingerprint or fingerprint!=saved.get('grounding_target_fingerprint'):
            raise ValueError('resume config mismatch: grounding_target_fingerprint')
