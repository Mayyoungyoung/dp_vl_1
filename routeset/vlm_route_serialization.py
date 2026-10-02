"""Strict observation/answer serialization for direct-VLM route baselines.

This is a data interface, not a trained model. No geometry repair, fallback,
candidate filtering, or reference trajectory is used to build the prompt.
"""
import json
import numpy as np

PROTOCOL='vlm_world_mm_xyz_open_v1'


def depth_image(depth):
    """Encode observed metric depth at 1 mm resolution, with explicit validity.

    RGB=(high byte,low byte,255) for valid depth; (0,0,0) for invalid.
    A vision processor may subsequently rescale pixels; this is not a claim
    that a pretrained VLM already understands this nonstandard encoding.
    """
    depth=np.asarray(depth)
    if depth.ndim!=2:raise ValueError('depth must be a metric HxW array')
    valid=np.isfinite(depth)&(depth>0)
    if (depth[valid]>65.535).any():raise ValueError('Depth exceeds declared encoding; do not clip silently')
    millimeters=np.zeros(depth.shape,dtype=np.uint16)
    millimeters[valid]=np.maximum(1,np.rint(depth[valid]*1000)).astype(np.uint16)
    encoded=np.stack([millimeters//256,millimeters%256,valid.astype(np.uint16)*255],axis=-1)
    return encoded.astype(np.uint8)


def prompt(instruction,current,camera_intrinsics,camera_extrinsics,k,horizon):
    if not isinstance(instruction,str) or not instruction.strip():raise ValueError('Task instruction required')
    if k not in (1,2,4,8) or horizon<2:raise ValueError('Supported positive candidate/horizon budget required')
    arrays=[np.asarray(value,dtype=float) for value in (current,camera_intrinsics,camera_extrinsics)]
    for value,shape in zip(arrays,[(8,),(3,3),(4,4)]):
        if value.shape!=shape or not np.isfinite(value).all():raise ValueError('Invalid current observation/calibration')
    metadata=dict(current_gripper_xyz_quaternion_open=arrays[0].tolist(),
                  camera_intrinsics=arrays[1].tolist(),camera_to_world=arrays[2].tolist())
    return ('Predict '+str(k)+' alternative task-level end-effector paths for this instruction: '+instruction+'\n'
        'Image 1 is current RGB. Image 2 encodes current metric depth: z_m=(256*R+G)/1000 when B=255; B=0 is unknown. '
        'Use only these observations, calibration, and current state.\n'+json.dumps(metadata,separators=(',',':'))+'\n'
        'Return only one JSON array containing exactly '+str(k)+' routes. Each route contains exactly '+str(horizon)+
        ' points [x_mm,y_mm,z_mm,open]. World coordinates are integers in millimeters; open is 0 or 1. '
        'The first point is the current end-effector state. Do not return scores, explanations, or extra routes.')


def serialize_paths(paths,opened):
    paths=np.asarray(paths);opened=np.asarray(opened)
    if paths.ndim!=3 or paths.shape[-1]!=3 or opened.shape!=paths.shape[:2]:raise ValueError('Expected KxHx3 and KxH')
    if not np.isfinite(paths).all() or not np.isfinite(opened).all():raise ValueError('Nonfinite positive reference')
    if np.abs(paths).max()>10:raise ValueError('Reference exceeds declared 10m serialization range')
    values=np.concatenate([np.rint(paths*1000).astype(np.int64),(opened>.5)[...,None].astype(np.int64)],axis=-1)
    return json.dumps(values.tolist(),separators=(',',':'))


def parse_paths(text,k,horizon):
    """Retain malformed/missing slots, and never hide overgeneration as K.

    The returned fixed-shape arrays support the common evaluator. An output
    with more than K routes is budget-ineligible; none is selected from it.
    """
    if k not in (1,2,4,8) or horizon<2:raise ValueError('Unsupported candidate/horizon budget')
    paths=np.full((k,horizon,3),np.nan,dtype=np.float32)
    opened=np.full((k,horizon),np.nan,dtype=np.float32)
    receipt=dict(protocol=PROTOCOL,requested_candidates=k,decoded_route_count=None,
        charged_candidate_slots=k,budget_exceeded=False,format_valid_candidates=0,errors=[])
    try:
        decoded=json.loads(text)
        if not isinstance(decoded,list):raise ValueError('Top-level route array required')
    except (ValueError,TypeError) as error:
        receipt['errors'].append(str(error));return paths,opened,receipt
    receipt['decoded_route_count']=len(decoded)
    receipt['charged_candidate_slots']=max(k,len(decoded))
    if len(decoded)>k:
        receipt['budget_exceeded']=True;receipt['errors'].append('Overgeneration; no top-K filtering')
        return paths,opened,receipt
    for index,route in enumerate(decoded):
        try:
            array=np.asarray(route)
            if array.shape!=(horizon,4):raise ValueError('Wrong waypoint count/shape')
            # JSON booleans and floating strings are not valid coordinate tokens.
            if any(type(v) is not int for point in route for v in point):raise ValueError('Expected JSON integers')
            if any(abs(v)>10000 for point in route for v in point[:3]) or any(point[3] not in (0,1) for point in route):
                raise ValueError('Out-of-range coordinate/event')
            paths[index]=array[:,:3]/1000.;opened[index]=array[:,3]
            receipt['format_valid_candidates']+=1
        except (ValueError,TypeError,OverflowError) as error:
            receipt['errors'].append('slot '+str(index)+': '+str(error))
    if len(decoded)<k:receipt['errors'].append('Missing '+str(k-len(decoded))+' requested candidate slots')
    return paths,opened,receipt


def assistant_only_labels(full_ids,prompt_ids):
    """Mask the entire actual processor prompt, rejecting changed prefixes.

    Called on processor token IDs, never on a guessed token-length estimate.
    Generation must use prompt_ids alone; teacher-forced answers are labels.
    """
    full=np.asarray(full_ids,dtype=np.int64);prefix=np.asarray(prompt_ids,dtype=np.int64)
    if full.ndim!=1 or prefix.ndim!=1 or not len(prefix) or len(full)<=len(prefix):
        raise ValueError('Nonempty prompt plus answer required')
    if not np.array_equal(full[:len(prefix)],prefix):raise ValueError('Processor prompt prefix differs from training sequence')
    labels=full.copy();labels[:len(prefix)]=-100
    return labels
