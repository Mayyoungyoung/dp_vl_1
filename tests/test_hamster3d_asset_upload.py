import hashlib
import json
from pathlib import Path

import pytest

import scripts.upload_hamster3d_assets as u


def row():
    return {"path":"tiny","bytes":3,"expected_sha256":hashlib.sha256(b'abc').hexdigest()}


def test_publish_verifies_and_atomically_creates_complete_file(tmp_path):
    stage=tmp_path/'file.part';target=tmp_path/'file';stage.write_bytes(b'abc')
    assert u.publish(stage,target,row())=='published_verified'
    assert target.read_bytes()==b'abc' and not stage.exists()


def test_good_target_reused_without_mutating_staging(tmp_path):
    stage=tmp_path/'file.part';target=tmp_path/'file';stage.write_bytes(b'bad');target.write_bytes(b'abc')
    assert u.publish(stage,target,row())=='reused_verified'
    assert stage.read_bytes()==b'bad' and target.read_bytes()==b'abc'


@pytest.mark.parametrize('stage_bytes,target_bytes',[(b'ab',None),(b'bad',None),(b'abc',b'bad')])
def test_bad_complete_or_staging_preserved_without_publish(tmp_path,stage_bytes,target_bytes):
    stage=tmp_path/'file.part';target=tmp_path/'file';stage.write_bytes(stage_bytes)
    if target_bytes is not None:target.write_bytes(target_bytes)
    with pytest.raises(u.IntegrityError):u.publish(stage,target,row())
    assert stage.read_bytes()==stage_bytes
    if target_bytes is None:assert not target.exists()
    else:assert target.read_bytes()==target_bytes


def test_no_clobber_when_target_appears_between_check_and_link(tmp_path,monkeypatch):
    stage=tmp_path/'file.part';target=tmp_path/'file';stage.write_bytes(b'abc')
    def racing_link(*args):target.write_bytes(b'bad');raise FileExistsError()
    monkeypatch.setattr(u.os,'link',racing_link)
    with pytest.raises(u.IntegrityError):u.publish(stage,target,row())
    assert target.read_bytes()==b'bad' and stage.read_bytes()==b'abc'


def remote_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(u,'SERVER_ROOT',str(tmp_path))
    manifest=tmp_path/'research_v2/releases'/u.SOURCE_REVISION/'configs/hamster3d_assets_v1.json'
    manifest.parent.mkdir(parents=True);manifest.write_text(json.dumps({'model_files':[row()]}))
    monkeypatch.setattr(u,'MANIFEST_SHA',u.sha(manifest))
    (tmp_path/u.MODEL_RELATIVE).mkdir(parents=True)
    common={'run_id':'test1','local_pid':123,'source_sha256':'a'*64}
    return common,manifest


def test_remote_lock_and_complete_transfer_lifecycle(tmp_path,monkeypatch):
    common,_=remote_fixture(tmp_path,monkeypatch)
    assert u.remote_action(dict(common,action='begin'))['status']=='begun'
    with pytest.raises(FileExistsError):u.remote_action(dict(common,action='begin'))
    state=u.remote_action(dict(common,action='inspect',filename='tiny'))
    assert state['status']=='missing';Path(state['staging']).write_bytes(b'abc')
    assert u.remote_action(dict(common,action='publish',filename='tiny'))['status']=='published_verified'
    assert u.remote_action(dict(common,action='inspect',filename='tiny'))['status']=='reused_verified'
    assert u.remote_action(dict(common,action='verify_all'))['files'][0]['sha256']==row()['expected_sha256']
    assert u.remote_action(dict(common,action='release',result={'exit_code':0}))['status']=='released'
    assert not (tmp_path/'runs/hamster3d_upload_v1/upload.lock').exists()


def test_partial_staging_does_not_trigger_overwriting_transfer(tmp_path,monkeypatch):
    common,_=remote_fixture(tmp_path,monkeypatch);u.remote_action(dict(common,action='begin'))
    state=u.remote_action(dict(common,action='inspect',filename='tiny'));stage=Path(state['staging']);stage.write_bytes(b'a')
    with pytest.raises(u.IntegrityError):u.remote_action(dict(common,action='inspect',filename='tiny'))
    assert stage.read_bytes()==b'a'


def test_assets_download_lock_prevents_upload_start(tmp_path,monkeypatch):
    common,_=remote_fixture(tmp_path,monkeypatch)
    lock=tmp_path/'runs/hamster3d_preparation_v1/preparation.lock';lock.parent.mkdir(parents=True);lock.write_text('999')
    with pytest.raises(u.IntegrityError):u.remote_action(dict(common,action='begin'))
    assert not (tmp_path/'runs/hamster3d_upload_v1/upload.lock').exists()


def test_remote_manifest_change_rejected_before_asset_access(tmp_path,monkeypatch):
    common,manifest=remote_fixture(tmp_path,monkeypatch);manifest.write_text('changed')
    with pytest.raises(u.IntegrityError):u.remote_action(dict(common,action='begin'))


def test_remote_program_keeps_posix_paths_on_windows_and_compiles():
    code=u.remote_script({'run_id':'test','action':'begin','local_pid':1,'source_sha256':'a'*64})
    assert "SERVER_ROOT='/home/wzy/dpvlm/route_set_v1'" in code
    assert 'SERVER_ROOT=Path' not in code
    compile(code,'remote_upload','exec')


def test_ssh_timeout_preserves_safe_action_record(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise u.subprocess.TimeoutExpired(args[0], 600, output='not copied')
    monkeypatch.setattr(u.subprocess, 'run', fail)
    with pytest.raises(u.subprocess.TimeoutExpired):
        u.ssh_action({'action':'inspect','filename':'tiny'}, tmp_path, 3)
    saved=json.loads((tmp_path/'remote_003.json').read_text())
    assert saved['exception_type']=='TimeoutExpired'
    assert saved['status']=='ssh_action_failed'
    assert 'not copied' not in json.dumps(saved)
