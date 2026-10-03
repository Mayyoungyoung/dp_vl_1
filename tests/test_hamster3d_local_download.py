import hashlib
import io
import json
from pathlib import Path
import subprocess

import pytest

import scripts.download_hamster3d_local as d


def row():
    return {"path": "tiny", "bytes": 3, "expected_sha256": hashlib.sha256(b"abc").hexdigest(),
            "url": "https://huggingface.co/public/fixed/tiny"}


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body); self.status = status; self.headers = headers or {}


class Opener:
    def __init__(self, response): self.response = response; self.requests = []
    def open(self, request, timeout):
        self.requests.append(request)
        assert request.get_header('Authorization') is None
        return self.response


def test_exact_original_manifest_without_reports_fixture(tmp_path):
    # Git's archived server file has CRLF; local checkout may have LF.
    raw = (Path(__file__).parents[1] / 'configs/hamster3d_assets_v1.json').read_bytes()
    raw = raw.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    p = tmp_path / 'manifest.json'; p.write_bytes(raw)
    assert hashlib.sha256(raw).hexdigest() == d.MANIFEST_SHA256
    m = d.load_manifest(p)
    assert len(m['model_files']) == 18
    assert sum(x['path'].endswith('.safetensors') for x in m['model_files']) == 4
    p.write_bytes(raw + b' ')
    with pytest.raises(d.IntegrityError): d.load_manifest(p)


def test_200_cannot_be_appended_to_partial(tmp_path):
    p = tmp_path / 'tiny.part'; p.write_bytes(b'ab')
    opener = Opener(Response(b'abc', 200, {'Content-Length': '3'}))
    with pytest.raises(d.RangeProtocolError): d.transfer_once(row(), p, tmp_path, tmp_path/'logs', opener=opener)
    assert p.read_bytes() == b'ab'
    assert opener.requests[0].get_header('Range') == 'bytes=2-'


def test_valid_206_appends_only_validated_range(tmp_path):
    p = tmp_path / 'tiny.part'; p.write_bytes(b'ab')
    opener = Opener(Response(b'c', 206, {'Content-Length': '1', 'Content-Range': 'bytes 2-2/3'}))
    result = d.transfer_once(row(), p, tmp_path, tmp_path/'logs', opener=opener)
    assert p.read_bytes() == b'abc' and result['received_bytes'] == 1
    assert d.verify_complete(p, row())


@pytest.mark.parametrize('headers', [
    {'Content-Range': 'bytes 1-2/3', 'Content-Length': '2'},
    {'Content-Range': 'bytes 2-2/4', 'Content-Length': '1'},
    {'Content-Range': 'bytes 2-2/3', 'Content-Length': '2'},
    {'Content-Length': '1'},
    {'Content-Range': 'bytes 2-2/3', 'Content-Encoding': 'gzip'},
])
def test_bad_content_range_length_or_encoding_preserves_partial(tmp_path, headers):
    p=tmp_path/'tiny.part'; p.write_bytes(b'ab')
    with pytest.raises(d.RangeProtocolError):
        d.transfer_once(row(), p, tmp_path, tmp_path/'logs', opener=Opener(Response(b'c',206,headers)))
    assert p.read_bytes() == b'ab'


def test_short_body_is_retained_for_range_resume(tmp_path):
    p=tmp_path/'tiny.part'
    with pytest.raises(OSError):
        d.transfer_once(row(),p,tmp_path,tmp_path/'logs',opener=Opener(Response(b'ab',200,{'Content-Length':'3'})))
    assert p.read_bytes()==b'ab'
    d.transfer_once(row(),p,tmp_path,tmp_path/'logs',opener=Opener(Response(b'c',206,{'Content-Length':'1','Content-Range':'bytes 2-2/3'})))
    assert d.verify_complete(p,row())


def test_good_complete_reused_and_bad_complete_preserved(tmp_path, monkeypatch):
    dest=tmp_path/'assets'; dest.mkdir(); logs=tmp_path/'logs'; logs.mkdir()
    monkeypatch.setattr(d,'DESTINATION',dest); monkeypatch.setattr(d,'RUN_ROOT',logs)
    p=dest/'tiny';p.write_bytes(b'abc')
    monkeypatch.setattr(d.subprocess,'Popen',lambda *a,**k:pytest.fail('correct file must not download'))
    result=d.download_file(row(),'small',tmp_path/'manifest',logs)
    assert result['status']=='reused_verified' and result['attempts']==[]
    p.write_bytes(b'bad')
    with pytest.raises(d.IntegrityError):d.download_file(row(),'small',tmp_path/'manifest',logs)
    assert p.read_bytes()==b'bad'


def test_wrong_completed_partial_is_not_promoted(tmp_path, monkeypatch):
    dest=tmp_path/'assets';dest.mkdir();logs=tmp_path/'logs';logs.mkdir()
    monkeypatch.setattr(d,'DESTINATION',dest);monkeypatch.setattr(d,'RUN_ROOT',logs)
    (dest/'tiny.part').write_bytes(b'bad')
    d.write_json(dest/'tiny.part.identity.json',{'revision':d.REVISION,'path':'tiny','bytes':3,
        'sha256':row()['expected_sha256'],'manifest_sha256':d.MANIFEST_SHA256})
    class Child:
        pid=123
        def wait(self,timeout=None):return 1
    def spawn(command,**kw):
        receipt=Path(command[command.index('--worker-receipt')+1])
        d.write_json(receipt,{'status':'failed','exception_type':'IntegrityError','cause_type':None})
        return Child()
    monkeypatch.setattr(d.subprocess,'Popen',spawn)
    with pytest.raises(d.IntegrityError):d.download_file(row(),'small',tmp_path/'manifest',logs)
    assert not (dest/'tiny').exists() and (dest/'tiny.part').read_bytes()==b'bad'


def test_wall_timeout_stops_only_own_worker_and_keeps_partial(tmp_path, monkeypatch):
    dest=tmp_path/'assets';dest.mkdir();logs=tmp_path/'logs';logs.mkdir()
    monkeypatch.setattr(d,'DESTINATION',dest);monkeypatch.setattr(d,'RUN_ROOT',logs)
    class Child:
        pid=123; returncode=None
        def wait(self,timeout=None):
            if timeout is not None:raise subprocess.TimeoutExpired('local worker',timeout)
            self.returncode=-1;return -1
        def kill(self):self.killed=True
    child=Child()
    def spawn(*a,**kw):(dest/'tiny.part').write_bytes(b'a');return child
    monkeypatch.setattr(d.subprocess,'Popen',spawn)
    with pytest.raises(TimeoutError):d.download_file(row(),'small',tmp_path/'manifest',logs)
    assert child.killed and (dest/'tiny.part').read_bytes()==b'a'
    assert json.loads((logs/'tiny.file.json').read_text())['attempts'][0]['status']=='file_wall_timeout'


def test_exception_record_omits_library_message():
    error=OSError('https://signed.invalid/?secret=NEVER_LOG_THIS')
    assert 'NEVER_LOG_THIS' not in json.dumps(d.exception_info(error))


def test_network_failures_have_at_most_three_attempts(tmp_path, monkeypatch):
    dest=tmp_path/'assets';dest.mkdir();logs=tmp_path/'logs';logs.mkdir()
    monkeypatch.setattr(d,'DESTINATION',dest);monkeypatch.setattr(d,'RUN_ROOT',logs)
    calls=[]
    class Child:
        pid=123
        def wait(self,timeout=None):return 1
    def spawn(command,**kw):
        calls.append(command)
        receipt=Path(command[command.index('--worker-receipt')+1])
        d.write_json(receipt,{'status':'failed','exception_type':'URLError','cause_type':'TimeoutError'})
        return Child()
    monkeypatch.setattr(d.subprocess,'Popen',spawn)
    with pytest.raises(RuntimeError):d.download_file(row(),'small',tmp_path/'manifest',logs)
    assert len(calls)==3 and not (dest/'tiny').exists()
    saved=json.loads((logs/'tiny.file.json').read_text())
    assert len(saved['attempts'])==3
    assert all(x['exception_type']=='URLError' for x in saved['attempts'])


@pytest.mark.parametrize('used,free', [(d.CAP,10*d.RESERVE),(0,d.RESERVE)])
def test_disk_cap_and_one_gib_reserve_prevent_new_transfer(tmp_path, monkeypatch, used, free):
    monkeypatch.setattr(d,'owned_bytes',lambda *a:used)
    monkeypatch.setattr(d.shutil,'disk_usage',lambda *a:type('Disk',(),{'free':free})())
    with pytest.raises(RuntimeError):d.disk_guard(tmp_path,tmp_path/'logs',additional=1)


def test_weights_does_not_implicitly_run_small_or_accept_missing_marker(tmp_path, monkeypatch):
    repo=tmp_path;dest=repo/'assets';logs=repo/'logs'
    monkeypatch.setattr(d,'REPO',repo);monkeypatch.setattr(d,'DESTINATION',dest);monkeypatch.setattr(d,'RUN_ROOT',logs)
    monkeypatch.setattr(d,'load_manifest',lambda _: {'model_files':[row()]})
    monkeypatch.setattr(d,'download_file',lambda *a,**k:pytest.fail('missing small marker must block weights'))
    assert d.main(['--stage','weights','--manifest',str(tmp_path/'manifest')])==1
    assert not (logs/'weights.completed.json').exists() and not (logs/'download.lock').exists()
