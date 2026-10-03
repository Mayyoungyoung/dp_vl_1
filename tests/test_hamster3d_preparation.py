import copy
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest

from scripts.prepare_hamster3d_external import child, valid_existing, validate_manifest, prepare_assets


ROOT = Path(__file__).resolve().parents[1]


def manifest():
    return json.loads((ROOT / "configs/hamster3d_assets_v1.json").read_text())


def test_manifest_exact_allowlist_and_all_hashes():
    m = manifest()
    validate_manifest(m)
    assert sum(r["bytes"] for r in m["model_files"]) == 18295875781
    assert sum(r["bytes"] for r in m["code_files"]) == 310785
    assert sum(r["path"].endswith(".safetensors") for r in m["model_files"]) == 4
    assert all(len(r["expected_sha256"]) == 64 for g in ("model_files", "code_files") for r in m[g])


@pytest.mark.parametrize("key,value", [("model_revision", "main"), ("code_revision", "main"), ("model_repo", "other/model")])
def test_reject_revision_or_repository_substitution(key, value):
    m = manifest(); m[key] = value
    with pytest.raises(ValueError): validate_manifest(m)


def test_reject_implicit_or_explicit_authentication():
    m = manifest(); m["download"]["token"] = None
    with pytest.raises(ValueError): validate_manifest(m)


def test_reject_changed_asset_url():
    m = manifest(); m["model_files"][0]["url"] += "?token=not-allowed"
    with pytest.raises(ValueError): validate_manifest(m)


def test_existing_valid_bytes_are_reused_and_bad_bytes_are_preserved(tmp_path):
    p = tmp_path / "shard"; p.write_bytes(b"abc")
    row = {"bytes": 3, "expected_sha256": hashlib.sha256(b"abc").hexdigest()}
    assert valid_existing(p, row)
    bad = copy.deepcopy(row); bad["expected_sha256"] = "0" * 64
    with pytest.raises(ValueError): valid_existing(p, bad)
    assert p.read_bytes() == b"abc"
    assert not valid_existing(tmp_path / "missing", row)


def test_child_rejects_parent_escape(tmp_path):
    with pytest.raises(ValueError): child(tmp_path, "../outside")
    assert child(tmp_path, "inside/x") == tmp_path / "inside/x"


def test_child_rejects_symlink_escape_when_supported(tmp_path):
    root = tmp_path / "root"; root.mkdir()
    out = tmp_path / "outside"; out.mkdir()
    try: (root / "link").symlink_to(out, target_is_directory=True)
    except (OSError, NotImplementedError): pytest.skip("symlink privilege unavailable")
    with pytest.raises(ValueError): child(root, "link/file")


def test_downloader_explicit_public_auth_and_reuses_correct_complete_file(tmp_path, monkeypatch):
    import importlib.metadata
    m = manifest()
    row = dict(m["model_files"][0], path="tiny", bytes=3, expected_sha256=hashlib.sha256(b"abc").hexdigest())
    m["model_files"] = [row]; m["code_files"] = []
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        result = Path(kwargs["local_dir"]) / kwargs["filename"]
        result.write_bytes(b"abc")
        return str(result)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.35.3")
    session = tmp_path / "session"; session.mkdir()
    first = prepare_assets(tmp_path, m, session)
    second = prepare_assets(tmp_path, m, session)
    assert len(calls) == 1
    assert calls[0]["token"] is False and calls[0]["force_download"] is False
    assert calls[0]["revision"] == m["model_revision"]
    assert first["files"][0]["status"] == "downloaded_verified"
    assert second["files"][0]["status"] == "reused_verified"


def test_download_hash_failure_is_not_retried_or_overwritten(tmp_path, monkeypatch):
    import importlib.metadata
    m = manifest()
    row = dict(m["model_files"][0], path="tiny", bytes=3, expected_sha256=hashlib.sha256(b"abc").hexdigest())
    m["model_files"] = [row]; m["code_files"] = []
    calls = []
    def download(**kwargs):
        calls.append(kwargs)
        result = Path(kwargs["local_dir"]) / kwargs["filename"]
        result.write_bytes(b"bad")
        return str(result)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(hf_hub_download=download))
    monkeypatch.setattr(importlib.metadata, "version", lambda _: "0.35.3")
    session = tmp_path / "session"; session.mkdir()
    with pytest.raises(ValueError): prepare_assets(tmp_path, m, session)
    assert len(calls) == 1
    assert (tmp_path / m["model_relative"] / "tiny").read_bytes() == b"bad"
