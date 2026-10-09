"""Real predicted cells/paths and corresponding RGB in a registered edit pair."""
import argparse,itertools,json
import numpy as np
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from research_feasible_space_v1.prepare import RUN
from research_realized_coverage_v1.core import DATA,lines,read,write,sha
from scripts.evaluate_paired_modes import references
from research_feasible_space_v1.geometry import node_radii_numpy

def main(name,analysis,xyz,bounded):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    all_records=read(RUN/analysis/'adaptation_rows.json');left=all_records[xyz+':adaptive'];right=all_records[bounded+':adaptive'];cases=[]
    for condition,label in ((lambda a,b:a['category']!='valid_same_mode_repair' and b['category']=='valid_same_mode_repair','actual_same_mode_repair_improvement'),
        (lambda a,b:a['category']=='valid_same_mode_repair' and b['category']!='valid_same_mode_repair','actual_same_mode_repair_loss'),
        (lambda a,b:b['category']!='valid_same_mode_repair','actual_remaining_failure')):
        options=[dict(b,comparison_category=label) for a,b in zip(left,right) if condition(a,b) and not any(c['source_id']==b['source_id'] and c['destination_id']==b['destination_id'] and c['mode']==b['mode'] for c in cases)]
        if options:cases.append(options[0])
    labels={r['id']:r for r in lines(DATA/'export/supervision.jsonl') if r['split']=='DEV_MODEL'}
    obs={r['id']:r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL'}
    pools={};rows={};indices={}
    for n in (xyz,bounded):
        with np.load(RUN/n/'eval_adaptive/pool.npz') as z:pools[n]={k:z[k] for k in z.files}
        rows[n]={r['id']:r for r in read(RUN/n/'eval_adaptive/rows.json')};indices[n]={str(v):i for i,v in enumerate(pools[n]['ids'])}
    offsets=np.array(list(itertools.product((-1,1),repeat=3)))
    hashes={};records=[]
    if not cases:raise RuntimeError('No actual diagnostic cases')
    for ci,case in enumerate(cases[:3]):
        fig=plt.figure(figsize=(15,10))
        for side,ident in enumerate((case['source_id'],case['destination_id'])):
            image=obs[ident]['image'];hashes[image]=sha(image)
            ax=fig.add_subplot(2,3,side*3+1);ax.imshow(plt.imread(image));ax.axis('off');ax.set_title(('Source' if side==0 else 'Edited')+' observed RGB\n'+ident,fontsize=8)
            ref=references(labels[ident]);cs=ref['truth']['obstacle_centers'];hs=ref['truth']['obstacle_halfsizes']
            for key in ('route_config','verification_only'):hashes[ref['label'][key]]=sha(ref['label'][key])
            for col,n in enumerate((xyz,bounded)):
                ax=fig.add_subplot(2,3,side*3+col+2,projection='3d');i=indices[n][ident];pool=pools[n];row=rows[n][ident]
                ax.bar3d(*(cs-hs).T,*(2*hs).T,alpha=.15,color='gray',shade=False)
                matching=[j for j,w in enumerate(row['words']) if w==case['mode'] and row['valid'][j]]
                if not matching:matching=[j for j,w in enumerate(row['assigned_modes']) if w==case['mode']]
                slot=matching[0] if matching else 0
                for j,p in enumerate(pool['paths'][i]):
                    color='green' if row['valid'][j] else '#cc5555';ax.plot(*p.T,color=color,alpha=1 if j==slot else .25,lw=2 if j==slot else .8)
                if col==1:
                    center=pool['centers'][i,slot];radius=pool['radii'][i,slot];rho=node_radii_numpy(radius)
                    cell_form=read(RUN/n/'eval_adaptive/metrics.json').get('cell_form','uniform_swept_boxes')
                    for j in range(23):
                        r0,r1=(rho[j],rho[j+1]) if cell_form=='tapered_endpoint_boxes' else (radius[j],radius[j])
                        vertices=np.r_[center[j]+r0*offsets,center[j+1]+r1*offsets]
                        if max(r0,r1)>1e-8:
                            hull=ConvexHull(vertices);poly=Poly3DCollection(vertices[hull.simplices],facecolors='#4b95dd',edgecolors='none',alpha=.05);ax.add_collection3d(poly)
                    ax.plot(*center.T,'b--',lw=1)
                ax.set_title(n+'\nvalid='+str(sum(row['valid']))+'/8; requested '+row['assigned_modes'][slot]+'\nactual '+str(row['words'][slot]),fontsize=7)
                ax.set_xlabel('x m');ax.set_ylabel('y m');ax.set_zlabel('z m');ax.view_init(35,-65)
                records.append(dict(case=ci,id=ident,model=n,focus_slot=slot,requested_mode=row['assigned_modes'][slot],valid=row['valid'][slot],actual_mode=row['words'][slot]))
        fig.suptitle(case['mode']+' | '+case['comparison_category']+'\nGray: evaluation truth posts; blue: predicted reachable cells; green/red: actual valid/invalid paths',fontsize=10)
        fig.tight_layout(rect=(0,0,1,.95));fig.savefig(out/('predicted_corridors_case%d.png'%ci),dpi=150);plt.close(fig)
    write(out/'MANIFEST.json',dict(cases=cases,records=records,images_sha256=hashes,source_pools_sha256={n:sha(RUN/n/'eval_adaptive/pool.npz') for n in pools},locked_access=False))
    print(json.dumps(dict(cases=min(3,len(cases)),records=len(records))),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--analysis',required=True);p.add_argument('--xyz',required=True);p.add_argument('--bounded',required=True);main(**vars(p.parse_args()))
