"""Mechanical initial-layout duplicate audit; no task outcome or trajectory read."""
import hashlib
import json
import numpy as np

PROTOCOL = 'multitask_initial_layout_fingerprint_v1'


def json_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def array_hash(value):
    array=np.asarray(value)
    if not np.isfinite(array).all():raise ValueError('Nonfinite initial observation')
    header=json.dumps(dict(dtype=str(array.dtype),shape=list(array.shape)),sort_keys=True).encode()
    return hashlib.sha256(header+b'\0'+np.ascontiguousarray(array).tobytes()).hexdigest()


def canonical_pose(value, quantized=False):
    pose=np.asarray(value,dtype=float).copy()
    if pose.shape!=(7,) or not np.isfinite(pose).all():raise ValueError('Invalid initial pose')
    # q and -q encode the same rotation. No temporal or geometry alignment.
    if pose[3+int(np.argmax(np.abs(pose[3:]))) ]<0:pose[3:]*=-1
    if quantized:
        pose[:3]=np.round(pose[:3],3);pose[3:]=np.round(pose[3:],4)
    pose[pose==0]=0.
    return pose.tolist()


def physical_layout(world, quantized=False):
    result={}
    for name,record in sorted(world.items()):
        if name=='_robot' or not ('color' in record or 'joint_position' in record):continue
        row=dict(pose=canonical_pose(record['pose'],quantized))
        if 'joint_position' in record:
            joint=np.asarray(record['joint_position'],dtype=float)
            if not np.isfinite(joint).all():raise ValueError('Invalid initial joint')
            if quantized:joint=np.round(joint,4)
            joint[joint==0]=0.;row['joint_position']=joint.tolist()
        result[name]=row
    if not result:raise ValueError('No physical shapes/joints in initial world')
    return result


def fingerprint(world, inputs, spec):
    return dict(protocol=PROTOCOL,parent_id=spec['parent_id'],task=spec['task'],split=spec['split'],
        canonical_world_sha256=json_hash(world),current_pose_sha256=array_hash(inputs['gripper_pose']),
        rgb_array_sha256=array_hash(inputs['rgb']),depth_array_sha256=array_hash(inputs['depth']),
        physical_layout_sha256=json_hash(physical_layout(world)),
        physical_layout_quantized_sha256=json_hash(physical_layout(world,True)),
        quantization=dict(position_m=.001,quaternion_component=.0001,joint_position=.0001),
        layout_exclusions=['color','velocity','robot','dummy/waypoint/success sensor'],
        limitation='Task shapes/joints available in world audit, not complete mesh equivalence. Quantized matches are conservative candidate duplicates; different hashes do not certify independent layouts.')


def audit_fingerprints(rows):
    groups=[];statuses={row['task']:'no_duplicate_detected_not_independence_certificate' for row in rows}
    for kind in ('physical_layout_sha256','physical_layout_quantized_sha256','rgb_array_sha256','rgb_file_sha256'):
        buckets={}
        for row in rows:
            if row.get(kind):buckets.setdefault((row['task'],row[kind]),[]).append(row)
        for (task,digest),members in buckets.items():
            identities={(row.get('source_root',''),row['parent_id']) for row in members}
            if len(identities)<2:continue
            roles=sorted({row['split'] for row in members});cross=len(roles)>1
            groups.append(dict(task=task,kind=kind,sha256=digest,cross_split=cross,roles=roles,
                members=[{key:row.get(key) for key in ('source_root','parent_id','split')} for row in members]))
            if cross:statuses[task]='blocked_cross_split_layout_duplicate'
    return dict(protocol=PROTOCOL,inspected_mechanical_fingerprints=len(rows),duplicate_groups=groups,
        task_usage_status=statuses,model_use_requires_gate_check=True,
        action='Retain every original record and role. Block new cross-split use of flagged tasks pending grouped redesign; do not silently drop or relabel parents.',
        unknown_policy='Missing physical hashes remain unverified; seed uniqueness and hash differences are not independence certificates.')
