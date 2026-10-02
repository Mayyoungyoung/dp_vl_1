"""Read-only local verification/plotting of the completed TRAIN support audit."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[2]
folder=Path(__file__).resolve().parent
binary=root/'runs/observed_local_support_train8_v1'
server='/home/wzy/dpvlm/route_set_v1/runs/observed_local_support_train8_v1'
digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
summary=json.loads((folder/'summary.json').read_text())
queries=json.loads((folder/'per_query.json').read_text())
assert digest(folder/'per_query.json')==summary['artifact_sha256']['per_query.json']
assert digest(binary/'queries.npz')==summary['artifact_sha256']['queries.npz']
with np.load(binary/'queries.npz') as archive:
    np.testing.assert_array_equal(archive['xyz'],np.asarray([row['xyz'] for row in queries]))
    assert len(archive['xyz'])==713
collision=[row for row in queries if row['kind']=='prediction_first_contact']
fractions={str(fraction):dict(exact_contact_arc=sum(row['normalized_arc_fraction']<=fraction for row in collision),
    contact_H24_index=sum(row['normalized_vertex_fraction']<=fraction for row in collision),
    first_intersecting_segment_start_arc=sum(row['arc_before_segment']<=fraction for row in collision)) for fraction in (.25,.5,.75,1.)}
parent_weighted={kind:{radius:float(np.mean([row['clouds']['exact_original_head_pixel_stride']['radii'][radius]['coverage']
    for row in values['per_parent'].values() if row['queries']])) for radius in ('0.05','0.1','0.2')}
    for kind,values in summary['all_queries'].items()}
(folder/'comparison.json').write_text(json.dumps(dict(contact_progress_counts=fractions,parent_macro_coverage=parent_weighted),indent=2))
index=dict(code_commit=summary['code_commit'],all_local_hashes_verified=True,
    artifact=dict(local=str(binary/'queries.npz'),server=server+'/queries.npz',sha256=summary['artifact_sha256']['queries.npz']),
    metadata_sha256={name:digest(folder/name) for name in ('summary.json','per_query.json','comparison.json')},
    canonical_git_script_sha256='c1be847e5b9df59f4dcd13e0b5eea311b2be9cfccaaf54b9c060efd0a3f0632b',
    deployed_script_sha256=summary['source_sha256']['script'],
    launcher_sha256=(folder/'jobs/launcher.sha256').read_text().split()[0])
(folder/'artifact_index.json').write_text(json.dumps(index,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4))
radii=np.array([.05,.1,.2])
for kind,label,color in [('positive_reference_prefix','40 reference prefixes / 680 samples','#377eb8'),
                         ('prediction_first_contact','33 first collision contacts','#e41a1c')]:
    for cloud,style in [('exact_original_head_pixel_stride','-'),('all_valid_depth_pixels','--')]:
        values=summary['all_queries'][kind]['clouds'][cloud]['radii']
        axes[0].plot(radii,[100*values[str(r)]['coverage'] for r in radii],style,marker='o',color=color,
            label=label+(' (head stride2)' if style=='-' else ' (full cloud)'))
axes[0].set(xlabel='Query-centered radius (m)',ylabel='Queries with >=1 observed point (%)',ylim=(0,105),
    title='Fixed TRAIN support; absence does not mean free')
axes[0].legend(fontsize=7)
positions=np.arange(3)
for offset,key,label in [(-.22,'exact_contact_arc','Exact contact arc'),(0.,'first_intersecting_segment_start_arc','Segment-start arc lower bound'),
                         (.22,'contact_H24_index','Contact H24 index fraction')]:
    axes[1].bar(positions+offset,[fractions[str(f)][key] for f in (.25,.5,.75)],width=.22,label=label)
axes[1].set(xticks=positions,xticklabels=['0.25','0.50','0.75'],xlabel='Normalized progress threshold',
    ylabel='First collision contacts covered / 33',ylim=(0,46),title='Arc and index fractions differ')
axes[1].legend(fontsize=7)
fig.suptitle('8 fixed TRAIN parents; frozen old32 peak last1000 predictions; no DEV selection')
fig.tight_layout()
fig.savefig(folder/'support_and_progress.png',dpi=160)
print(json.dumps(dict(contact_progress_counts=fractions,parent_macro_coverage=parent_weighted,index=index)))
