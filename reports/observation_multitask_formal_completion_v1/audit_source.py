import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
p=Path('/home/wzy/dpvlm/route_set_v1');r=p/'runs/observation_multitask_validated_five_formal_v1';d=Path((r/'data_root').read_text().strip());manifest=d/'partition_manifest.json';plan=json.loads(manifest.read_text())
role={}
for row in plan['parents']:
 entry=role.setdefault(row['split'],dict(requested=0,closed_marker_present=0,worker_lock_present=0));entry['requested']+=1
 parent=d/row['split']/'parents'/row['parent_id'];entry['closed_marker_present']+=int((parent/'closed.json').is_file());entry['worker_lock_present']+=int((parent/'worker.lock').exists())
meta={n:(r/n).read_text().strip() if (r/n).exists() else None for n in ['status','exit_code','started_at','finished_at','pid','source_release']}
print(json.dumps(dict(protocol='mechanical_completion_only_v1',checked_at=datetime.now(timezone.utc).isoformat(),run_directory=str(r),data_directory=str(d),run_metadata=meta,roles=role,requested_parents=len(plan['parents']),closed_markers_present=sum(x['closed_marker_present'] for x in role.values()),partition_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),scope='Only job metadata, registered partition manifest, and closed.json / worker.lock existence inspected. No closed marker contents, attempts, images, routes, labels or locked outcomes read. Closure is not success.'),indent=2))
