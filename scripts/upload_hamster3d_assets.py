"""Draft fixed-manifest SCP publication; no downloads, environment or model calls."""
import argparse
import datetime
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

MANIFEST_SHA = "53c93bdebfa519391173d0a0987b994de4793b84b05554051d63cc5618b24032"
SERVER_ROOT = "/home/wzy/dpvlm/route_set_v1"
SOURCE_REVISION = "1b0348ef48393a2d98113956575e99388c40d4a6"
MODEL_RELATIVE = "data/3d_hamster_ddc5987a56cdcb14e5e2297817612532e46e912b"
REPO = Path(__file__).resolve().parents[1]


class IntegrityError(Exception): pass


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(b)
    return h.hexdigest()


def verify(path, row):
    path = Path(path)
    if not path.exists() and not path.is_symlink(): return False
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["expected_sha256"]:
        raise IntegrityError("existing complete or staging file differs; preserved")
    return True


def write_json(path, obj):
    path = Path(path); tmp = path.with_suffix(path.suffix + ".writing")
    tmp.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8"); os.replace(tmp, path)


def publish(staging, target, row):
    """Publish a fully verified file without ever overwriting a racing target."""
    staging, target = Path(staging), Path(target)
    if verify(target, row): return "reused_verified"
    if not verify(staging, row): raise IntegrityError("staging file is missing")
    with staging.open("r+b") as f: os.fsync(f.fileno())
    try:
        os.link(staging, target)  # Same-filesystem atomic no-clobber publication.
    except FileExistsError:
        verify(target, row)
        return "reused_verified"
    staging.unlink()
    return "published_verified"


def remote_action(payload):
    """Executed with stdlib on CPU0 through the existing authorized SSH host."""
    started = time.monotonic()
    root = Path(SERVER_ROOT)
    manifest_path = root / "research_v2/releases" / SOURCE_REVISION / "configs/hamster3d_assets_v1.json"
    if sha(manifest_path) != MANIFEST_SHA: raise IntegrityError("remote immutable manifest differs")
    manifest = json.loads(manifest_path.read_text())
    model = root / MODEL_RELATIVE
    if model.is_symlink() or model.resolve().parent != (root / "data").resolve():
        raise IntegrityError("remote model root differs")
    run_root = root / "runs/hamster3d_upload_v1"
    run_root.mkdir(parents=True, exist_ok=True)
    lock = run_root / "upload.lock"
    identity = {"run_id": payload["run_id"], "local_pid": payload["local_pid"],
                "source_sha256": payload["source_sha256"], "manifest_sha256": MANIFEST_SHA}
    if payload["action"] == "begin":
        if (root / "runs/hamster3d_preparation_v1/preparation.lock").exists():
            raise IntegrityError("assets preparation currently owns its lock")
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps(identity).encode()); os.close(fd)
        return {"status": "begun", "identity": identity, "pid": os.getpid()}
    if not lock.exists() or json.loads(lock.read_text()) != identity:
        raise IntegrityError("upload lock identity differs")
    if payload["action"] == "release":
        session = run_root / payload["run_id"]; session.mkdir(exist_ok=True)
        write_json(session / "status.json", dict(identity, result=payload["result"], model_calls=0))
        lock.unlink()
        return {"status": "released"}
    if payload["action"] == "verify_all":
        rows = []
        for row in manifest["model_files"]:
            if not verify(model / row["path"], row): raise IntegrityError("one of eighteen final assets is missing")
            rows.append({"path": row["path"], "bytes": row["bytes"], "sha256": row["expected_sha256"]})
        return {"status": "all_18_verified", "files": rows, "elapsed_seconds": time.monotonic() - started}
    row = next(r for r in manifest["model_files"] if r["path"] == payload["filename"])
    if Path(row["path"]).name != row["path"]: raise IntegrityError("nonflat filename")
    target = model / row["path"]
    staging_root = model / ".upload_staging"
    if staging_root.is_symlink(): raise IntegrityError("staging root symlink")
    staging_root.mkdir(exist_ok=True)
    staging = staging_root / (row["path"] + ".part")
    if verify(target, row):
        status = "reused_verified"
    elif payload["action"] == "publish":
        status = publish(staging, target, row)
    elif payload["action"] == "inspect":
        if staging.exists() or staging.is_symlink():
            verify(staging, row)
            status = "staging_verified"
        else:
            used = sum(p.stat().st_size for p in model.rglob("*") if p.is_file())
            if used + row["bytes"] > 22 * 2 ** 30 or shutil.disk_usage(model).free < row["bytes"] + 2 ** 30:
                raise IntegrityError("remote staging space budget would be exceeded")
            status = "missing"
    else: raise IntegrityError("unknown remote action")
    return {"status": status, "path": row["path"], "bytes": row["bytes"],
            "sha256": row["expected_sha256"] if status != "missing" else None,
            "staging": str(staging), "pid": os.getpid(), "elapsed_seconds": time.monotonic() - started}


def remote_script(payload):
    functions = "\n\n".join(inspect.getsource(f) for f in (sha, verify, write_json, publish, remote_action))
    header = "import json,os,hashlib,time,shutil\nfrom pathlib import Path\nclass IntegrityError(Exception): pass\n"
    constants = "SERVER_ROOT=%r\nSOURCE_REVISION=%r\nMODEL_RELATIVE=%r\nMANIFEST_SHA=%r\n" % (
        str(SERVER_ROOT), SOURCE_REVISION, MODEL_RELATIVE, MANIFEST_SHA)
    footer = "\ntry:\n    print(json.dumps(remote_action(%r)))\nexcept Exception as error:\n    print(json.dumps({'status':'failed','exception_type':type(error).__name__,'safe_reason':str(error) if isinstance(error,IntegrityError) else None}))\n    raise SystemExit(1)\n" % payload
    return header + constants + functions + footer


def ssh_action(payload, session, number):
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
               "-o", "ServerAliveCountMax=2", "wzy3090",
               "taskset -c 0 /home/wzy/dpvlm/route_set_v1/.venv-qwen/bin/python -"]
    started = time.monotonic()
    try:
        result = subprocess.run(command, input=remote_script(payload), text=True, capture_output=True, timeout=600,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    except Exception as error:
        write_json(session / ("remote_%03d.json" % number), {
            "request": payload, "status": "ssh_action_failed", "exception_type": type(error).__name__,
            "elapsed_seconds": time.monotonic() - started})
        raise
    response = json.loads(result.stdout) if result.stdout.strip() else {"status": "ssh_failed_without_json"}
    record = {"request": payload, "response": response, "exit_code": result.returncode,
              "elapsed_seconds": time.monotonic() - started, "stderr_bytes": len(result.stderr.encode())}
    write_json(session / ("remote_%03d.json" % number), record)
    if result.returncode or response.get("status") == "failed":
        raise IntegrityError("remote check rejected; inspect safe receipt")
    return response


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(REPO / "reports/hamster3d_preparation_validation_v1/source/configs/hamster3d_assets_v1.json.source.txt"))
    args = p.parse_args(argv)
    manifest_path = Path(args.manifest)
    if sha(manifest_path) != MANIFEST_SHA: raise IntegrityError("original manifest bytes required")
    manifest = json.loads(manifest_path.read_text()); local = REPO / ".bootstrap/hamster3d_model_ddc5987a"
    for row in manifest["model_files"]:
        if Path(row["path"]).name != row["path"] or not verify(local / row["path"], row):
            raise IntegrityError("all eighteen local complete assets must be verified before any SSH")
    run_root = REPO / ".bootstrap/hamster3d_upload_v1"; run_root.mkdir(exist_ok=True)
    lock = run_root / "upload.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY); os.write(fd, str(os.getpid()).encode()); os.close(fd)
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_%d" % os.getpid()
    session = run_root / run_id; session.mkdir()
    common = {"run_id": run_id, "local_pid": os.getpid(), "source_sha256": sha(__file__)}
    status = dict(common, status="running", manifest_sha256=MANIFEST_SHA, files=[], model_calls=0, scp_calls=0,
                  start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    begun = False; number = 0; started = time.monotonic()
    def remote(action, **fields):
        nonlocal number
        number += 1
        return ssh_action(dict(common, action=action, **fields), session, number)
    write_json(session / "status.json", status)
    try:
        remote("begin"); begun = True
        for row in manifest["model_files"]:
            state = remote("inspect", filename=row["path"])
            if state["status"] == "missing":
                expected_staging = SERVER_ROOT + "/" + MODEL_RELATIVE + "/.upload_staging/" + row["path"] + ".part"
                if state["staging"] != expected_staging: raise IntegrityError("remote staging path differs from fixed target")
                command = ["scp", "-B", "-q", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=30",
                           "-o", "ServerAliveCountMax=2", str(local / row["path"]), "wzy3090:" + state["staging"]]
                issued = {"path": row["path"], "status": "scp_issued", "command": command,
                          "expected_bytes": row["bytes"], "expected_sha256": row["expected_sha256"]}
                status["files"].append(issued); status["scp_calls"] += 1
                write_json(session / "status.json", status)
                transfer_started = time.monotonic()
                with (session / (row["path"] + ".scp.log")).open("wb") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    issued["pid"] = child.pid; write_json(session / "status.json", status)
                    try: code = child.wait(timeout=3600)
                    except BaseException:
                        child.kill(); child.wait(); issued["status"] = "scp_interrupted_partial_preserved"
                        issued["elapsed_seconds"] = time.monotonic() - transfer_started
                        write_json(session / "status.json", status); raise
                issued["exit_code"] = code
                issued["elapsed_seconds"] = time.monotonic() - transfer_started
                if code: raise RuntimeError("SCP failed; remote staging preserved; no automatic retry")
                issued["publication"] = remote("publish", filename=row["path"])
                issued["status"] = "published_verified"
            elif state["status"] == "staging_verified":
                status["files"].append(remote("publish", filename=row["path"]))
            else: status["files"].append(state)
            write_json(session / "status.json", status)
        status["final_verification"] = remote("verify_all")
        status.update(status="completed", exit_code=0)
    except Exception as error:
        status.update(status="failed", exit_code=1, exception_type=type(error).__name__)
    except KeyboardInterrupt:
        status.update(status="interrupted", exit_code=130)
    finally:
        status.update(end_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), elapsed_seconds=time.monotonic() - started)
        if begun:
            try: remote("release", result={k: status.get(k) for k in ("status", "exit_code", "files", "scp_calls", "elapsed_seconds")})
            except Exception as error: status.update(status="failed", exit_code=1, release_error=type(error).__name__)
        try: write_json(session / "status.json", status)
        finally: lock.unlink()
    print(json.dumps({k: status[k] for k in ("run_id", "status", "exit_code", "scp_calls", "elapsed_seconds")}))
    return status["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
