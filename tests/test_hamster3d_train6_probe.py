"""Offline contract tests only: no Torch import/model/SSH/download/simulator."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts import probe_hamster3d_train6 as p


def rows():
    return [dict(id=key,parent_id=p.PARENTS[i//3],split='TRAIN',image='/observed/front.png',
                 instruction='Reach the observed colored target.') for i,key in enumerate(p.IDS)]


def test_fixed_contract_and_conserved_budget():
    config=json.loads(Path('configs/hamster3d_train6_probe_v1.json').read_text())
    assert p.validate_policy(config)==p.EXPECTED
    assert config['maximum_generate_calls']*config['candidates_per_input']==6
    assert config['maximum_generate_calls']*config['maximum_new_tokens_per_call']==6144
    assert config['warmup_calls']==0 and not config['read_supervision']


@pytest.mark.parametrize('key,value',[('split','DEV_MODEL'),('maximum_generate_calls',7),
    ('retry',True),('dtype','float16'),('candidates_per_input',4),('gpu_memory_fraction',.36)])
def test_policy_cannot_expand(key,value):
    config=copy.deepcopy(p.EXPECTED);config[key]=value
    with pytest.raises(ValueError):p.validate_policy(config)


def test_exact_six_lines_never_parse_following_dev_line(tmp_path):
    path=tmp_path/'observations.jsonl'
    path.write_text('\n'.join(json.dumps(row) for row in rows())+'\nTHIS IS NOT JSON DEV\n')
    assert p.exact_rows(path)==rows()


@pytest.mark.parametrize('change',['role','extra_goal','id'])
def test_answer_token_or_role_identity_change_rejected(tmp_path,change):
    r=rows()
    if change=='role':r[0]['split']='TEST_LOCKED'
    elif change=='id':r[0]['id']='another'
    else:r[0]['goal_xyz']=[0,0,0]
    path=tmp_path/'observations.jsonl';path.write_text('\n'.join(json.dumps(v) for v in r))
    with pytest.raises(ValueError):p.exact_rows(path)


def test_signed_intrinsics_matches_official_inverse_projection_no_flip():
    k=np.array([[-100.,0,112],[0,-120.,112],[0,0,1.]])
    pose=np.eye(4);pose[:3,3]=[.1,-.2,.3]
    camera,world=p.uvd_to_world([[250,750,2.]],k,pose,224,224)
    np.testing.assert_allclose(camera,[[1.12,-112./120.,2.]])
    np.testing.assert_allclose(world,camera+pose[:3,3])
    scaled=k.copy();scaled[:2]*=640/224
    same,_=p.uvd_to_world([[250,750,2.]],scaled,np.eye(4),640,640)
    np.testing.assert_allclose(camera,same)


def test_projection_does_not_clamp_or_replace_predicted_depth():
    camera,_=p.uvd_to_world([[1200,-100,-.3]],np.eye(3),np.eye(4),224,224)
    np.testing.assert_allclose(camera,[[-80.64,6.72,-.3]])
    with pytest.raises(ValueError):p.uvd_to_world([[0,0,float('nan')]],np.eye(3),np.eye(4),224,224)


def test_empty_official_parse_is_empty_candidate_not_retry():
    camera,world=p.uvd_to_world([],np.eye(3),np.eye(4),224,224)
    assert camera.shape==world.shape==(0,3)


def test_raw_path_escape_rejected(tmp_path):
    allowed=tmp_path/'TRAIN';allowed.mkdir();outside=tmp_path/'DEV';outside.write_text('x')
    with pytest.raises(ValueError):p.confined(outside,allowed)


class FakeTensor:
    def __init__(self,data):self.data=data
    def detach(self):return self
    def cpu(self):return self
    def tolist(self):return self.data


def test_streamed_tokens_preserve_partial_without_forward():
    audit=p.TokenAudit([1,2,3]);audit.put(FakeTensor([[1,2,3]]))
    audit.put(FakeTensor([7]));audit.put(FakeTensor([8]));audit.end()
    assert audit.generated_ids==[7,8] and audit.prompt_seen
    with pytest.raises(ValueError):audit.put(FakeTensor([9]*1023))


def test_streamer_rejects_prompt_or_batch_change():
    with pytest.raises(ValueError):p.TokenAudit([1]).put(FakeTensor([[2]]))
    with pytest.raises(ValueError):p.TokenAudit([1]).put(FakeTensor([[1],[1]]))


def test_offline_network_scope_restores_socket():
    before=p.socket.socket.connect
    with p.offline_network():
        assert p.socket.socket.connect is not before
        with pytest.raises(RuntimeError):p.socket.socket.connect(None,('example.org',443))
    assert p.socket.socket.connect is before


def test_resident_set_keeps_encode_and_direct_device_readers_off_meta():
    assert 'model.geometry_encoder' in p.RESIDENT
    assert 'model.visual' in p.RESIDENT
    assert 'model.language_model.rotary_emb' in p.RESIDENT
    assert not any('layers' in name for name in p.RESIDENT)
