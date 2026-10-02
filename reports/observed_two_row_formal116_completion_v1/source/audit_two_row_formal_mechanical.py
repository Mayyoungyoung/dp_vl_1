"""Metadata-only closure existence; never decode locked outcomes or arrays."""
from pathlib import Path
import hashlib,json,datetime
p=Path('/home/wzy/dpvlm/route_set_v1')
d=p/'data/observed_two_row_formal116_v1'
session=p/'runs/observed_two_row_formal116_v1/sessions/20261002T183023Z_485548'
registration=d/'registration.json'
plan=json.loads(registration.read_text())
roles={}
for item in plan['parent_plan']:
    entry=roles.setdefault(item['role'],dict(requested=0,closure_marker_present=0))
    entry['requested']+=1
    entry['closure_marker_present']+=int((d/'closures'/('%03d.json'%item['index'])).is_file())
metadata={}
for name in ('status','pid','started_at','finished_at','shard0_exit_code','shard1_exit_code','source_release'):
    f=session/name
    metadata[name]=f.read_text().strip() if f.is_file() else None
print(json.dumps(dict(protocol='two_row_formal116_closure_existence_only_v1',
    checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    registration_sha256=hashlib.sha256(registration.read_bytes()).hexdigest(),
    roles=roles,session_metadata=metadata,
    total_closed=sum(v['closure_marker_present'] for v in roles.values()),
    scope='Registration, job status, and closure filename existence only. No closure contents, attempts, images, trajectories, labels, or model metrics opened. Closure is not success.'),indent=2))
