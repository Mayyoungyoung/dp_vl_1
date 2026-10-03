"""All original candidates in the K1 and K8 residual scenes; zero new paths."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from analyze_saved_pools import HERE,SYNC,checker,sha,write


def main():
    check=checker();rows=json.loads((HERE/'PER_PARENT_FINAL_POOLS.json').read_text())
    with np.load(SYNC/'development.npz',allow_pickle=False) as z:
        mask=z['splits']=='DEV_MODEL';data={n:z[n][mask] for n in z.files}
    records=[]
    for k in (1,8):
        source=SYNC/'seed0/selection/step3000'/('k%d'%k)/'predictions.npz'
        with np.load(source,allow_pickle=False) as z:paths=z['paths']
        for i,row in enumerate(rows[str(k)]):
            if not row['invalid_candidates']:continue
            cols=3 if k==8 else 2;nr=3 if k==8 else 1
            fig,axes=plt.subplots(nr,cols,figsize=(15,12) if k==8 else (12,5),squeeze=False)
            lower,upper=check.wall_boxes(data['scenes'][i]);refs=data['paths'][i][data['path_mask'][i]];modes=data['modes'][i][data['path_mask'][i]]
            panels=[None]+list(range(k))
            for ax,candidate in zip(axes.flat,panels):
                for lo,hi in zip(lower,upper):
                    ax.add_patch(Rectangle(lo[:2],*(hi-lo)[:2],facecolor='.72'))
                    ax.add_patch(Rectangle((lo-.02)[:2],*((hi-lo)+.04)[:2],facecolor='none',edgecolor='.5',ls=':',lw=.7))
                for ref,mode in zip(refs,modes):
                    if candidate is None or int(mode)==row['candidate_raw_modes'][candidate]:
                        ax.plot(ref[:,0],ref[:,1],'--',color='green',lw=.8,alpha=.35)
                        if candidate is None:ax.text(ref[12,0],ref[12,1],str(int(mode)),color='green',fontsize=6)
                indices=range(k) if candidate is None else [candidate]
                for j in indices:
                    p=paths[i,j];good=row['candidate_valid'][j]
                    ax.plot(p[:,0],p[:,1],color='tab:blue' if good else 'tab:red',lw=1.3,marker='.',ms=2)
                    ax.text(p[9,0],p[9,1],'q%d'%j,color='black',fontsize=7)
                if candidate is None:title='All original candidates + all stored reference types'
                else:
                    j=candidate
                    title='q%d: raw type %d, valid=%s\nz=[%.4f,%.4f], length ratio %.3f'%(j,row['candidate_raw_modes'][j],row['candidate_valid'][j],paths[i,j,:,2].min(),paths[i,j,:,2].max(),row['length_ratio'][j])
                ax.set(title=title,xlim=(-1,1),ylim=(-1,1),xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
            fig.suptitle('%s | K%d | R%d | known U%d | invalid%d | shaded solid, dotted +2cm clearance'%(row['parent_id'],k,row['R_known_positive_types'],row['known_unique'],row['invalid_candidates']))
            fig.tight_layout();name='RESIDUAL_%s_K%d.png'%(row['parent_id'],k);fig.savefig(HERE/name,dpi=130);plt.close(fig)
            records.append(dict(parent_id=row['parent_id'],k=k,all_candidate_slots=list(range(k)),reference_types=modes.tolist(),
                                source_prediction_sha256=sha(source),image=name,image_sha256=sha(HERE/name),
                                all_paths_original=True,new_model_calls=0,repairs=0))
    write(HERE/'RESIDUAL_FIGURE_INDEX.json',dict(figures=records,scope='all original K1/K8 invalid scenes; no candidate dropped; all128 capacity chart separate'))
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
