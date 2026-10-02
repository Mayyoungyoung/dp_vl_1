"""Read-only job metadata and opaque closure hashing; never parse closure data."""
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path('/home/wzy/dpvlm/route_set_v1')
CORPUS = PROJECT/'data/observed_two_row_formal116_v1'
SESSION = PROJECT/'runs/observed_two_row_formal116_v1/sessions/20261002T183023Z_485548'
assert os.sched_getaffinity(0) == {2}, 'Metadata audit uses only authorized CPU2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


registration = CORPUS/'registration.json'
plan = json.loads(registration.read_text())  # Public prospective registration only.
assert len(plan['parent_plan']) == 116
roles, closures = {}, []
for expected, item in enumerate(plan['parent_plan']):
    assert item['index'] == expected
    path = CORPUS/'closures'/('%03d.json'%expected)
    exists = path.is_file()
    assert not path.is_symlink()
    role = roles.setdefault(item['role'], dict(requested=0, closure_marker_present=0))
    role['requested'] += 1
    role['closure_marker_present'] += int(exists)
    # Opaque bytes are hashed only. Never json.loads(), copy or print a closure.
    closures.append(dict(filename=path.name, index=expected, role_from_registration=item['role'],
        parent_id_from_registration=item['parent_id'], exists=exists,
        bytes=path.stat().st_size if exists else None, sha256=sha(path) if exists else None))

names = ('status','pid','started_at','finished_at','mode','source_release','source_sha256.txt',
    'shard0_pid','shard1_pid','shard0_exit_code','shard1_exit_code','frozen_launcher.sh',
    'targeted_tests.status.json','registration_prepare.status.json',
    'shard0/coordinator.status.json','shard1/coordinator.status.json',
    'shard0/exit_code','shard1/exit_code',
    'shard0/frozen_worker_wrapper.sh','shard1/frozen_worker_wrapper.sh')
metadata = []
for name in names:
    path = SESSION/name
    assert path.is_file() and not path.is_symlink()
    content = path.read_bytes()
    metadata.append(dict(relative_path=name, source_path=str(path), bytes=len(content), sha256=sha(path),
        base64=base64.b64encode(content).decode('ascii')))

release = Path((SESSION/'source_release').read_text().strip())
assert release.parent == PROJECT/'research_v2/releases'
assert release.name == '262648796fd47b25a2051cc227b6838515b651c6'
source_checks = []
for line in (SESSION/'source_sha256.txt').read_text().splitlines():
    expected, name = line.split(maxsplit=1)
    path = Path(name)
    assert release in path.parents and path.is_file()
    actual = sha(path)
    source_checks.append(dict(path=str(path), recorded_sha256=expected, actual_sha256=actual, equal=expected==actual))

print(json.dumps(dict(protocol='formal116_completion_mechanical_metadata_only_v1',
    checked_at=datetime.now(timezone.utc).isoformat(), cpu_affinity=sorted(os.sched_getaffinity(0)),
    corpus=str(CORPUS), session=str(SESSION), source_release=str(release),
    registration_sha256=sha(registration), registration_bytes=registration.stat().st_size,
    roles=roles, total_closure_markers=sum(r['closure_marker_present'] for r in roles.values()),
    closure_markers=closures, session_metadata=metadata, source_checks=source_checks,
    scope='Registration decoded; explicit coordinator status and source metadata copied. Closure files only hashed as opaque bytes, never decoded or copied. No worker/coordinator logs, raw images, trajectories, outcomes, labels or metrics read. Closure does not imply a successful route.'),indent=2))
