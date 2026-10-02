import numpy as np
import json
from routeset.multitask_fingerprints import audit_fingerprints,fingerprint


def make(role='TRAIN',parent='one',x=.1,color=0.,quaternion_sign=1):
    world={'shape':dict(pose=[x,0.,1.,0.,0.,0.,quaternion_sign],color=[color,0.,0.],velocity=[0.]*6),
        'waypoint0':dict(pose=[.5,0.,1.,0.,0.,0.,1.])}
    inputs=dict(gripper_pose=np.arange(7.),rgb=np.full((2,2,3),int(color*100),np.uint8),depth=np.ones((2,2)))
    return fingerprint(world,inputs,dict(parent_id=parent,task='push_button',split=role))


def test_same_physical_layout_color_language_versions_are_grouped():
    a,b=make(),make('DEV_MODEL','two',color=1.,quaternion_sign=-1)
    assert a['physical_layout_sha256']==b['physical_layout_sha256']
    assert a['rgb_array_sha256']!=b['rgb_array_sha256']
    result=audit_fingerprints([a,b])
    assert result['task_usage_status']['push_button']=='blocked_cross_split_layout_duplicate'


def test_quantized_near_duplicate_is_conservative_and_not_relabelled():
    a,b=make(),make('TEST_LOCKED','two',x=.10001,color=1.)
    assert a['physical_layout_sha256']!=b['physical_layout_sha256']
    assert a['physical_layout_quantized_sha256']==b['physical_layout_quantized_sha256']
    result=audit_fingerprints([a,b])
    assert result['duplicate_groups'][0]['roles']==['TEST_LOCKED','TRAIN']
    assert result['model_use_requires_gate_check']


def test_different_layout_hash_is_not_independence_certificate():
    result=audit_fingerprints([make(),make('DEV_MODEL','two',x=.15,color=1.)])
    assert not result['duplicate_groups']
    assert result['task_usage_status']['push_button']=='no_duplicate_detected_not_independence_certificate'


def test_legacy_locked_role_uses_only_existing_pointer_hash(tmp_path):
    from scripts.audit_multitask_layout_hashes import audit
    plan={'parents':[dict(parent_id='one',task='push_button',split='TRAIN'),dict(parent_id='two',task='push_button',split='TEST_LOCKED')]}
    (tmp_path/'partition_manifest.json').write_text(json.dumps(plan))
    for spec in plan['parents']:
        folder=tmp_path/spec['split']/'parents'/spec['parent_id'];folder.mkdir(parents=True)
        (folder/'reference_pointer.json').write_text(json.dumps(dict(image_sha256='same-existing-mechanical-image-hash')))
        for name in ('reference.json','front.png','observation.npz','record.json'):
            (folder/name).write_text('Forbidden raw content; not valid JSON or image')
    result=audit([tmp_path])
    assert result['parents_with_only_legacy_image_hash']==2
    assert result['parents_with_physical_hash']==0
    assert result['task_usage_status']['push_button']=='blocked_cross_split_layout_duplicate'
