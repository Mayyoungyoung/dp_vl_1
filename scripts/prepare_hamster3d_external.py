"""Prepare fixed public 3D HAMSTER assets/environment; never load a model or data."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


MODEL_REVISION = "ddc5987a56cdcb14e5e2297817612532e46e912b"
CODE_REVISION = "97216a8493f46301bf569d398462b8bb21c458c5"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def child(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("unsafe relative path")
    result = (Path(root) / relative).resolve()
    try:
        result.relative_to(Path(root).resolve())
    except ValueError:
        raise ValueError("path leaves owned root")
    return result


def validate_manifest(m):
    if (m["protocol"] != "hamster3d_external_assets_v1"
            or m["model_repo"] != "DAVIAN-Robotics/3D_HAMSTER"
            or m["model_revision"] != MODEL_REVISION or m["code_revision"] != CODE_REVISION):
        raise ValueError("fixed public revision changed")
    if m["download"]["token"] is not False or m["resources"]["gpu_calls"] != 0:
        raise ValueError("download must be unauthenticated and use no GPU")
    for group, count in (("model_files", 18), ("code_files", 50)):
        rows = m[group]
        if len(rows) != count or len({r["path"] for r in rows}) != count:
            raise ValueError("allowlist changed")
        for r in rows:
            child(Path.cwd(), r["path"])
            if len(r["expected_sha256"]) != 64 or r["bytes"] < 0:
                raise ValueError("missing exact size/hash")
            revision = MODEL_REVISION if group == "model_files" else CODE_REVISION
            prefix = ("https://huggingface.co/DAVIAN-Robotics/3D_HAMSTER/resolve/"
                      if group == "model_files" else
                      "https://raw.githubusercontent.com/DAVIAN-Robotics/3D_HAMSTER/")
            if r["url"] != prefix + revision + "/" + r["path"]:
                raise ValueError("download URL differs from fixed allowlist")
    if sum(r["bytes"] for r in m["model_files"]) != 18295875781:
        raise ValueError("model bytes changed")


def valid_existing(path, row):
    path = Path(path)
    if not path.exists():
        return False
    if not path.is_file() or path.stat().st_size != row["bytes"] or sha(path) != row["expected_sha256"]:
        raise ValueError("existing complete file differs; preserving it without overwrite")
    return True


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".writing")
    temp.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def tree_bytes(path):
    path = Path(path)
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file()) if path.exists() else 0


def check_disk(root, manifest, next_bytes=0):
    paths = [child(root, manifest[k]) for k in
             ("model_relative", "code_relative", "environment_relative", "run_relative")]
    used = sum(tree_bytes(p) for p in paths)
    cap = int(manifest["resources"]["additional_disk_budget_GiB"] * 2 ** 30)
    if used + next_bytes > cap or shutil.disk_usage(root).free < next_bytes + 2 ** 30:
        raise RuntimeError("preparation disk envelope would be exceeded")
    return {"owned_bytes": used, "cap_bytes": cap, "filesystem_free_bytes": shutil.disk_usage(root).free}


def append_record(path, row):
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def prepare_assets(root, m, session):
    # Import only after disabling all implicit credentials and Xet caches.
    from huggingface_hub import hf_hub_download
    # HF's HTTP-backoff logger can include public signed redirect URLs. Preserve
    # only our safe per-call categories instead of exporting those library logs.
    logging.disable(logging.CRITICAL)
    import importlib.metadata
    if importlib.metadata.version("huggingface-hub") != "0.35.3":
        raise RuntimeError("download library version differs")
    ledger = session / "files.jsonl"
    results = []
    for group, relative in (("code_files", m["code_relative"]), ("model_files", m["model_relative"])):
        directory = child(root, relative)
        directory.mkdir(parents=True, exist_ok=True)
        for row in m[group]:
            target = child(directory, row["path"])
            record = {"group": group, "path": row["path"], "bytes": row["bytes"],
                      "expected_sha256": row["expected_sha256"], "git_blob_sha1": row["git_blob_sha1"]}
            started = time.monotonic()
            append_record(ledger, dict(record, status="inspection_started"))
            if valid_existing(target, row):
                record.update(status="reused_verified", library_calls=0, sha256=row["expected_sha256"])
            else:
                check_disk(root, m, row["bytes"])
                for attempt in range(1, 4):
                    append_record(ledger, dict(record, status="issued", library_call=attempt))
                    try:
                        if group == "model_files":
                            downloaded = Path(hf_hub_download(
                                repo_id=m["model_repo"], filename=row["path"],
                                revision=MODEL_REVISION, token=False, local_dir=str(directory),
                                force_download=False, etag_timeout=30))
                            if downloaded.resolve() != target:
                                raise ValueError("hub returned unexpected path")
                        else:
                            target.parent.mkdir(parents=True, exist_ok=True)
                            partial = target.with_name(target.name + ".partial." + str(os.getpid()) + "." + str(attempt))
                            with urllib.request.urlopen(row["url"], timeout=60) as response, partial.open("xb") as f:
                                shutil.copyfileobj(response, f)
                            # Validate before publishing; a bad partial is retained.
                            valid_existing(partial, row)
                            if target.exists():
                                raise ValueError("target appeared while downloading")
                            os.replace(partial, target)
                        valid_existing(target, row)
                        record.update(status="downloaded_verified", library_calls=attempt, sha256=row["expected_sha256"])
                        break
                    except ValueError:
                        append_record(ledger, dict(record, status="integrity_failure_preserved", library_call=attempt))
                        raise
                    except Exception as error:
                        # Do not print signed redirect URLs, authorization headers, or environment values.
                        append_record(ledger, dict(record, status="network_or_library_failure", library_call=attempt,
                                                  exception_type=type(error).__name__))
                        if target.exists():
                            valid_existing(target, row)
                        if attempt == 3:
                            raise RuntimeError("bounded download calls exhausted; partial files preserved") from None
                        time.sleep(attempt)
            record["elapsed_seconds"] = time.monotonic() - started
            append_record(ledger, record)
            results.append(record)
    return {"files": results, "total_model_bytes": m["total_model_bytes"], "model_loaded": False,
            "model_forward_calls": 0, "data_inputs_read": 0, "disk": check_disk(root, m)}


def prepare_environment(root, m, session, requirements):
    envdir = child(root, m["environment_relative"])
    base = Path(m["environment"]["python_base"])
    if not base.is_file():
        raise RuntimeError("recorded private Python base missing")
    check_disk(root, m, max(0, 8 * 2 ** 30 - tree_bytes(envdir)) + 6 * 2 ** 30)
    marker = envdir / "hamster3d_environment_identity.json"
    identity = {"requirements_sha256": sha(requirements), "python_base": str(base),
                "python_version": m["environment"]["python_version"], "system_site_packages": False}
    if envdir.exists():
        if not marker.exists() or json.loads(marker.read_text()) != identity:
            raise ValueError("existing environment has no matching preparation identity")
    else:
        envdir.mkdir()
        # The marker is written before venv creation so an interrupted venv can be resumed.
        write_json(marker, identity)
    if not (envdir / "bin/python").is_file():
        subprocess.run([str(base), "-m", "venv", str(envdir)], check=True)
    py = str(envdir / "bin/python")
    cfg = (envdir / "pyvenv.cfg").read_text().lower()
    if "include-system-site-packages = false" not in cfg:
        raise ValueError("environment isolation guard failed")
    command = [py, "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
               "install", "--no-cache-dir", "--retries", "2", "--timeout", "60",
               "--index-url", "https://pypi.org/simple",
               "--extra-index-url", "https://download.pytorch.org/whl/cu121", "--report",
               str(session / "pip_install_report.json"), "-r", str(requirements)]
    write_json(session / "environment_command.json", {"command": command, "identity": identity,
               "policy": "Minimal official inference dependencies; no GUI/PEFT/xformers; no model load."})
    subprocess.run(command, check=True)
    subprocess.run([py, "-m", "pip", "check"], check=True)
    with (session / "pip_freeze.txt").open("w") as f:
        subprocess.run([py, "-m", "pip", "freeze", "--all"], stdout=f, check=True)
    smoke = "\n".join([
        "import importlib.metadata as m, json, sys",
        "from pathlib import Path",
        "expected=json.loads(Path(sys.argv[1]).read_text())['environment']['core_pins']",
        "for pin in expected:",
        "    name,version=pin.split('=='); assert m.version(name)==version, name",
        "import torch, torchvision, cv2, transformers, accelerate",
        "from hamster3d.inference.preprocessing import prepare_inputs",
        "from hamster3d.model import register_qwen3_vl_geometry",
        "register_qwen3_vl_geometry()",
        "assert not torch.cuda.is_initialized()",
        "print(json.dumps({'imports_passed':True,'cuda_initialized':False,'model_loads':0}))",
    ])
    local_env = dict(os.environ, PYTHONPATH=str(child(root, m["code_relative"])))
    subprocess.run([py, "-c", smoke, str(session / "manifest.json")], env=local_env, check=True)
    return {"environment": str(envdir), "identity": identity, "model_loaded": False,
            "model_forward_calls": 0, "data_inputs_read": 0, "disk": check_disk(root, m)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("assets", "environment"), required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--requirements", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if hasattr(os, "sched_getaffinity") and os.sched_getaffinity(0) != {0}:
        raise RuntimeError("preparation must run on CPU0 only")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("GPU must be explicitly hidden")
    m = json.loads(Path(args.manifest).read_text())
    validate_manifest(m)
    root = Path(args.root).resolve()
    run = child(root, m["run_relative"])
    run.mkdir(parents=True, exist_ok=True)
    session = Path(args.session).resolve()
    session.relative_to(run)
    session.mkdir(parents=True, exist_ok=False)
    os.environ.update(HF_HUB_DISABLE_IMPLICIT_TOKEN="1", HF_HUB_DISABLE_TELEMETRY="1",
                      HF_HUB_DISABLE_XET="1", HF_HUB_DISABLE_PROGRESS_BARS="1", HF_HOME=str(run / "hf_metadata"),
                      PIP_NO_CACHE_DIR="1", XFORMERS_DISABLED="1")
    temporary = run / "tmp"
    temporary.mkdir(exist_ok=True)
    os.environ["TMPDIR"] = str(temporary)
    lock = run / "preparation.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    started = time.monotonic()
    status = {"stage": args.stage, "status": "running", "pid": os.getpid(),
              "source_sha256": sha(__file__), "manifest_sha256": sha(args.manifest),
              "requirements_sha256": sha(args.requirements), "source_commit": os.environ.get("CODE_COMMIT"),
              "gpu_calls": 0, "model_loaded": False, "data_inputs_read": 0,
              "resume_requested": args.resume,
              "resume_command": ["taskset", "-c", "0", "bash",
                                 str(Path(__file__).resolve().with_name("launch_hamster3d_preparation_v1.sh")),
                                 args.stage, "--resume"]}
    write_json(session / "manifest.json", m)
    write_json(session / "status.json", status)
    try:
        identity_file = run / "preparation_identity.json"
        identity = {"manifest_sha256": sha(args.manifest), "requirements_sha256": sha(args.requirements),
                    "source_sha256": sha(__file__)}
        if identity_file.exists():
            if json.loads(identity_file.read_text()) != identity:
                raise ValueError("preparation source/config identity changed")
            if args.stage == "assets" and not args.resume:
                raise ValueError("existing preparation requires explicit --resume")
        else:
            write_json(identity_file, identity)
        if args.stage == "environment":
            receipt = run / "assets.completed.json"
            if not receipt.exists() or json.loads(receipt.read_text())["identity"] != identity:
                raise ValueError("environment requires successful same-source assets stage")
            for group, dest in (("model_files", m["model_relative"]), ("code_files", m["code_relative"])):
                for row in m[group]:
                    if not valid_existing(child(child(root, dest), row["path"]), row):
                        raise ValueError("completed assets missing")
            result = prepare_environment(root, m, session, Path(args.requirements).resolve())
        else:
            result = prepare_assets(root, m, session)
        write_json(session / "receipt.json", result)
        write_json(run / (args.stage + ".completed.json"), {"identity": identity,
                   "session": str(session), "receipt_sha256": sha(session / "receipt.json")})
        status.update(status="completed", exit_code=0)
    except Exception as error:
        status.update(status="failed", exit_code=1, exception_type=type(error).__name__)
        # Error messages from HTTP/pip libraries may contain signed URLs; detailed per-file
        # safe categories and pip subprocess logs remain available without dumping environment.
    finally:
        status["elapsed_seconds"] = time.monotonic() - started
        try:
            status["disk"] = check_disk(root, m)
        except Exception as error:
            status.update(status="failed", exit_code=1, disk_envelope_failure=type(error).__name__)
        write_json(session / "status.json", status)
        lock.unlink()
    print(json.dumps({k: status[k] for k in ("stage", "status", "exit_code", "elapsed_seconds")}))
    return status["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
