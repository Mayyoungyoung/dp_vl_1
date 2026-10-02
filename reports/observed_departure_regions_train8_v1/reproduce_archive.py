import hashlib, json
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
folder=root/'reports/observed_departure_regions_train8_v1'
binary=root/'runs/observed_departure_regions_train8_v1'
server='/home/wzy/dpvlm/route_set_v1/runs/observed_departure_regions_train8_v1'
digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
summary=json.loads((folder/'summary.json').read_text())
generation=json.loads((folder/'generation_complete.json').read_text())
references=json.loads((folder/'reference_evaluation.json').read_text())
assert digest(folder/'generation_complete.json')==summary['generation_complete_sha256']
assert digest(folder/'reference_evaluation.json')==summary['reference_evaluation_sha256']
artifacts=[]
for name, expected in sorted(summary['artifact_sha256'].items()):
    path=binary/name
    assert digest(path)==expected
    artifacts.append(dict(local=str(path),server=server+'/'+name,sha256=expected,bytes=path.stat().st_size))
assert len(artifacts)==24
index=dict(all_24_npz_hashes_verified=True,artifacts=artifacts,
    metadata_sha256={p.name:digest(p) for p in folder.glob('*.json') if p.name not in ('artifact_index.json','diagnostic_summary.json')},
    code_commit='d73a4f2da9459a663d2ba9769b889c1a2b3c2e78',
    deployed_script_sha256='23a6a11cd19375a7eeb602cc7507f0490ab756708cb955207f6fc0e1e5a88287',
    canonical_git_script_sha256='3151cc1327722446636763708c475343cd31415362f3e8983ce625089ba4fb3b',
    normalized_source_equals_git_blob=True,
    launcher_sha256='35e457fe13204054eafe2469012e1ef71a7c283182ab9fdf7d25028f06cf30b7')
(folder/'artifact_index.json').write_text(json.dumps(index,indent=2))
diagnostic=dict(references=Counter(str(r['known_type']) for r in references),
    generation_seconds_median=float(np.median([r['generation_seconds'] for r in generation['records']])),
    generation_seconds_max=float(max(r['generation_seconds'] for r in generation['records'])),radii={})
for n,radius in enumerate(summary['config']['radii_m']):
    rows=[r['radii'][n] for r in generation['records']]
    diagnostic['radii'][str(radius)]={key:dict(min=min(r[key] for r in rows),max=max(r[key] for r in rows),total=sum(r[key] for r in rows))
        for key in ('admitted_nodes','candidate_neighbor_edges','supercover_voxel_checks','reachable_nodes','boundary_nodes')}
(folder/'diagnostic_summary.json').write_text(json.dumps(diagnostic,indent=2))
print(json.dumps(diagnostic))

chosen='obstacle_reach_272000_target0'
archive=np.load(binary/(chosen+'_regions.npz'))
fig=plt.figure(figsize=(14,4.5))
colors={'left':'#d95f02','right':'#1b9e77','above':'#7570b3','None':'#555555'}
for n,radius in enumerate(summary['config']['radii_m']):
    ax=fig.add_subplot(1,3,n+1,projection='3d')
    center=archive['current'][:3]
    cells=archive['lower']+archive['cells%d'%n]*.025-center
    ax.scatter(*cells.T,s=4,alpha=.15,color='#377eb8',label='One connected boundary')
    ax.scatter([0],[0],[0],marker='*',s=70,color='black',label='Current tip')
    for record in references:
        if record['id']!=chosen: continue
        evaluation=record['radii'][n]
        if 'first_exit_xyz' not in evaluation: continue
        point=np.asarray(evaluation['first_exit_xyz'])-center
        kind=str(record['known_type'])
        covered=bool(evaluation['assigned_regions'])
        ax.scatter(*point,s=60,marker='o' if covered else 'x',color=colors.get(kind,'#a65628'),label=kind+(' mapped' if covered else ' unmapped'))
    ax.set(title='Radius %.2fm: J=1'%radius,xlabel='Relative x (m)',ylabel='Relative y (m)',zlabel='Relative z (m)')
    ax.legend(fontsize=7,loc='upper left')
fig.suptitle('Fixed first TRAIN parent / target0; frozen observed boundary and reference exits')
fig.tight_layout()
fig.savefig(folder/'first_parent_boundary.png',dpi=160)
