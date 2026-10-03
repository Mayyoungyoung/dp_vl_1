"""One-worker, anonymous, fixed-manifest local download; never SCP or load models."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

MANIFEST_SHA256 = "53c93bdebfa519391173d0a0987b994de4793b84b05554051d63cc5618b24032"
REVISION = "ddc5987a56cdcb14e5e2297817612532e46e912b"
CAP = 22 * 2 ** 30
RESERVE = 2 ** 30
LIMIT_SECONDS = {"small": 300, "weights": 7200}
REPO = Path(__file__).resolve().parents[1]
DESTINATION = REPO / ".bootstrap/hamster3d_model_ddc5987a"
RUN_ROOT = REPO / ".bootstrap/hamster3d_local_preparation_v1"


class IntegrityError(Exception): pass
class RangeProtocolError(Exception): pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path); temp = path.with_suffix(path.suffix + ".writing")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def exception_info(error):
    cause = error.__cause__ or error.__context__
    out = {"exception_type": type(error).__name__, "cause_type": type(cause).__name__ if cause else None}
    if isinstance(error, urllib.error.HTTPError): out["http_status"] = error.code
    if isinstance(error, (IntegrityError, RangeProtocolError)): out["safe_reason"] = str(error)
    return out  # Never stringify an HTTP exception or signed redirect URL.


def load_manifest(path):
    path = Path(path)
    if sha(path) != MANIFEST_SHA256:
        raise IntegrityError("expected original server manifest bytes, not a reserialized copy")
    m = json.loads(path.read_text())
    rows = m["model_files"]
    if m["model_revision"] != REVISION or len(rows) != 18 or sum(r["bytes"] for r in rows) != 18295875781:
        raise IntegrityError("fixed model identity changed")
    if len({r["path"] for r in rows}) != 18:
        raise IntegrityError("duplicate manifest filename")
    for r in rows:
        if Path(r["path"]).name != r["path"]:
            raise IntegrityError("only fixed flat model filenames allowed")
        if r["url"] != "https://huggingface.co/DAVIAN-Robotics/3D_HAMSTER/resolve/" + REVISION + "/" + r["path"]:
            raise IntegrityError("unexpected public URL")
    return m


def verify_complete(path, row):
    path = Path(path)
    if not path.exists(): return False
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["expected_sha256"]:
        raise IntegrityError("complete or completed partial bytes differ; preserved")
    return True


def owned_bytes(destination, run_root):
    return sum(p.stat().st_size for root in (destination, run_root) if Path(root).exists()
               for p in Path(root).rglob("*") if p.is_file())


def disk_guard(destination, run_root, additional=0):
    used = owned_bytes(destination, run_root)
    free = shutil.disk_usage(Path(destination).parent).free
    if used + additional > CAP or free < additional + RESERVE:
        raise RuntimeError("22 GiB owned cap or 1 GiB free reserve would be exceeded")
    return {"owned_bytes": used, "owned_cap_bytes": CAP, "free_bytes": free, "free_reserve_bytes": RESERVE}


def validate_headers(status, headers, offset, total):
    """A 200 response may never be appended to a partial file."""
    if offset and status != 206:
        raise RangeProtocolError("resume requires 206; existing partial preserved")
    if status not in (200, 206):
        raise RangeProtocolError("unexpected HTTP status")
    expected = total - offset
    if status == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", headers.get("Content-Range", ""))
        if not match or tuple(map(int, match.groups())) != (offset, total - 1, total):
            raise RangeProtocolError("Content-Range does not match the requested remaining bytes")
    length = headers.get("Content-Length")
    if length is not None and (not length.isdecimal() or int(length) != expected):
        raise RangeProtocolError("Content-Length differs from remaining manifest bytes")
    if headers.get("Content-Encoding", "identity").lower() not in ("identity", ""):
        raise RangeProtocolError("encoded transfer is not byte-range safe")
    return expected


class HTTPSOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, newurl):
        if urllib.parse.urlsplit(newurl).scheme != "https":
            raise RangeProtocolError("non-HTTPS redirect rejected")
        redirected = super().redirect_request(request, response, code, message, headers, newurl)
        if redirected is not None: redirected.remove_header("Authorization")
        return redirected


def transfer_once(row, partial, destination, run_root, timeout=30, opener=None):
    partial = Path(partial)
    if partial.is_symlink(): raise IntegrityError("symlink partial rejected")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > row["bytes"]: raise IntegrityError("partial exceeds fixed file size")
    if offset == row["bytes"]:
        return {"transfer_invocations": 0, "initial_partial_bytes": offset, "final_partial_bytes": offset}
    headers = {"Accept-Encoding": "identity", "User-Agent": "dpvlm-public-assets/1"}
    if offset: headers["Range"] = "bytes=%d-" % offset
    request = urllib.request.Request(row["url"], headers=headers, method="GET")
    # stdlib urllib does not read .netrc or HF tokens. No auth handler is installed.
    opener = opener or urllib.request.build_opener(HTTPSOnlyRedirect())
    before = time.monotonic(); written = 0
    with opener.open(request, timeout=timeout) as response:
        expected = validate_headers(response.status, response.headers, offset, row["bytes"])
        # Opening only after header validation protects a partial against HTTP 200.
        with partial.open("ab" if offset else "wb", buffering=0) as f:
            while True:
                block = response.read(min(1024 * 1024, expected - written + 1))
                if not block: break
                if written + len(block) > expected:
                    raise RangeProtocolError("response exceeds declared remaining size")
                f.write(block); written += len(block)
                if written % (32 * 1024 * 1024) == 0:
                    disk_guard(destination, run_root)
                    os.fsync(f.fileno())
            os.fsync(f.fileno())
    if written != expected:
        raise OSError("short HTTP body; partial preserved for a later bounded attempt")
    return {"transfer_invocations": 1, "initial_partial_bytes": offset, "received_bytes": written,
            "final_partial_bytes": partial.stat().st_size, "elapsed_seconds": time.monotonic() - before}


def worker(args):
    started = time.monotonic()
    result = {"status": "failed", "pid": os.getpid(), "model_calls": 0}
    try:
        m = load_manifest(args.manifest)
        row = next(r for r in m["model_files"] if r["path"] == args.filename)
        result.update(transfer_once(row, DESTINATION / (row["path"] + ".part"), DESTINATION, RUN_ROOT))
        verify_complete(DESTINATION / (row["path"] + ".part"), row)
        result.update(status="transferred_verified", verified_sha256=row["expected_sha256"])
    except Exception as error:
        result.update(exception_info(error))
    result["worker_elapsed_seconds"] = time.monotonic() - started
    write_json(args.worker_receipt, result)
    return 0 if result["status"] == "transferred_verified" else 1


def download_file(row, stage, manifest, session):
    started = time.monotonic(); deadline = started + LIMIT_SECONDS[stage]
    target = DESTINATION / row["path"]; partial = target.with_name(target.name + ".part")
    result = {"path": row["path"], "expected_bytes": row["bytes"], "expected_sha256": row["expected_sha256"],
              "file_wall_limit_seconds": LIMIT_SECONDS[stage], "attempts": []}
    if verify_complete(target, row):
        result.update(status="reused_verified", elapsed_seconds=time.monotonic() - started)
        return result
    part_identity = {"revision": REVISION, "path": row["path"], "bytes": row["bytes"],
                     "sha256": row["expected_sha256"], "manifest_sha256": MANIFEST_SHA256}
    meta = target.with_name(target.name + ".part.identity.json")
    if meta.exists():
        if json.loads(meta.read_text()) != part_identity: raise IntegrityError("partial identity differs")
    elif partial.exists():
        raise IntegrityError("unidentified partial preserved")
    else:
        write_json(meta, part_identity)
    for number in range(1, 4):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > row["bytes"]: raise IntegrityError("oversize partial preserved")
        disk_guard(DESTINATION, RUN_ROOT, row["bytes"] - offset)
        remaining = deadline - time.monotonic()
        if remaining <= 0: raise TimeoutError("total per-file wall budget exhausted")
        receipt = session / (row["path"] + ".attempt%d.json" % number)
        issued = dict(number=number, status="issued", offset=offset,
                      file_elapsed_seconds=time.monotonic() - started, maximum_remaining_seconds=remaining)
        result["attempts"].append(issued)
        write_json(session / (row["path"] + ".file.json"), result)
        command = [sys.executable, str(Path(__file__).resolve()), "--transfer-worker", "--manifest", str(manifest),
                   "--filename", row["path"], "--worker-receipt", str(receipt)]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        child = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, creationflags=flags)
        issued["worker_pid"] = child.pid
        write_json(session / (row["path"] + ".file.json"), result)
        try:
            issued["exit_code"] = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            child.kill(); child.wait()
            issued.update(status="file_wall_timeout", exit_code=child.returncode)
            write_json(session / (row["path"] + ".file.json"), result)
            raise TimeoutError("bounded network worker stopped; partial preserved") from None
        except BaseException:
            child.kill(); child.wait()
            issued.update(status="interrupted", exit_code=child.returncode)
            write_json(session / (row["path"] + ".file.json"), result)
            raise
        actual = json.loads(receipt.read_text()) if receipt.exists() else {"status": "worker_exit_without_receipt"}
        issued.update(actual)
        write_json(session / (row["path"] + ".file.json"), result)
        if issued["exit_code"] == 0:
            if (actual.get("status") != "transferred_verified"
                    or actual.get("verified_sha256") != row["expected_sha256"]
                    or not partial.is_file() or partial.is_symlink() or partial.stat().st_size != row["bytes"]):
                raise IntegrityError("worker verification receipt differs")
            break
        if actual.get("exception_type") in ("IntegrityError", "RangeProtocolError"):
            raise IntegrityError("HTTP range/integrity failure; partial preserved without retry")
        if number == 3: raise RuntimeError("three transfer attempts exhausted; partial preserved")
    if target.exists(): raise IntegrityError("complete target appeared unexpectedly")
    os.replace(partial, target)
    result.update(status="downloaded_verified", sha256=row["expected_sha256"],
                  elapsed_seconds=time.monotonic() - started, actual_bytes=target.stat().st_size)
    write_json(session / (row["path"] + ".file.json"), result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("small", "weights"))
    p.add_argument("--manifest", default=str(REPO / "reports/hamster3d_preparation_validation_v1/source/configs/hamster3d_assets_v1.json.source.txt"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--transfer-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--filename", help=argparse.SUPPRESS)
    p.add_argument("--worker-receipt", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.transfer_worker: return worker(args)
    if args.stage is None: p.error("explicit --stage small or weights required")
    m = load_manifest(args.manifest)
    DESTINATION.resolve().relative_to(REPO.resolve()); RUN_ROOT.resolve().relative_to(REPO.resolve())
    DESTINATION.mkdir(parents=True, exist_ok=True); RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if DESTINATION.is_symlink() or RUN_ROOT.is_symlink(): raise IntegrityError("owned root symlink rejected")
    lock = RUN_ROOT / "download.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_%s_%d" % (args.stage, os.getpid())
    session = RUN_ROOT / stamp; session.mkdir()
    identity = {"manifest_sha256": MANIFEST_SHA256, "source_sha256": sha(__file__), "destination": str(DESTINATION.resolve())}
    status = {"run_id": stamp, "pid": os.getpid(), "stage": args.stage, "status": "running", "identity": identity,
              "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "model_calls": 0, "scp_calls": 0,
              "resume_command": [sys.executable, str(Path(__file__).resolve()), "--stage", args.stage,
                                 "--manifest", str(Path(args.manifest).resolve()), "--resume"], "files": []}
    started = time.monotonic()
    write_json(session / "status.json", status)
    try:
        marker = RUN_ROOT / "identity.json"
        if marker.exists():
            if json.loads(marker.read_text()) != identity: raise IntegrityError("source/manifest identity changed")
            if (RUN_ROOT / (args.stage + ".started.json")).exists() and not args.resume:
                raise IntegrityError("existing stage requires explicit --resume")
        else: write_json(marker, identity)
        if args.stage == "weights":
            previous = RUN_ROOT / "small.completed.json"
            if not previous.exists() or json.loads(previous.read_text())["identity"] != identity:
                raise IntegrityError("same-identity small stage must finish first")
            for row in m["model_files"]:
                if not row["path"].endswith(".safetensors") and not verify_complete(DESTINATION / row["path"], row):
                    raise IntegrityError("previous small asset missing")
        write_json(RUN_ROOT / (args.stage + ".started.json"), {"run_id": stamp, "identity": identity})
        selected = [r for r in m["model_files"] if r["path"].endswith(".safetensors") == (args.stage == "weights")]
        for row in selected:
            status["current_file"] = row["path"]; write_json(session / "status.json", status)
            status["files"].append(download_file(row, args.stage, Path(args.manifest).resolve(), session))
            write_json(session / "status.json", status)
        status.update(status="completed", exit_code=0)
    except Exception as error:
        status.update(status="failed", exit_code=1, **exception_info(error))
    except KeyboardInterrupt as error:
        status.update(status="interrupted", exit_code=130, **exception_info(error))
    finally:
        status["ended_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        status["elapsed_seconds"] = time.monotonic() - started
        try: status["disk"] = disk_guard(DESTINATION, RUN_ROOT)
        except Exception as error: status.update(status="failed", exit_code=1, disk_failure=exception_info(error))
        try:
            write_json(session / "status.json", status)
            if status["exit_code"] == 0:
                write_json(RUN_ROOT / (args.stage + ".completed.json"),
                           {"identity": identity, "session": str(session), "files": status["files"]})
        finally: lock.unlink()
    print(json.dumps({k: status[k] for k in ("run_id", "status", "exit_code", "elapsed_seconds")}))
    return status["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
