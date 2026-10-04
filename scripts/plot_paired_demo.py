"""Observed RGB, actual sealed routes and independently calibrated q."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write,lines
from scripts.paired_modes_data import RUN,DATA
from scripts.plot_paired_results import setup


def main(arm,seed,output):
    plt=setup()
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib import colors,cm
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    folder=RUN/'evaluation/paired_dev'/('%s_seed%d'%(arm,seed))
    rows={r['id']:r for r in read(folder/'rows.json')};refs=read(folder/'REFERENCE_GEOMETRY.json')
    observations={r['id']:r for r in lines(DATA/'export/observations.jsonl') if r['split']=='DEV_MODEL'}
    with np.load(folder/'pool.npz') as a:paths=dict(zip(a['ids'],a['paths']))
    calibration=RUN/'reliability'/('%s_seed%d'%(arm,seed))/('calibration_seed%d'%seed)
    with np.load(calibration/'paired_dev.npz') as a:qs=dict(zip(a['ids'],a['q']))
    groups={}
    for ident,row in rows.items():
        scene,target=ident.rsplit('_target',1);family=scene.rsplit('_',1)[0]
        groups.setdefault((family,target),[]).append(row)
    ranked=sorted(groups,key=lambda k:(np.mean([r['K']['8']['GeometricModeCount'] for r in groups[k]]),k))
    examples={'median_modes':ranked[len(ranked)//2],'fewest_modes':ranked[0]}
    write(output/'selection.json',dict(arm=arm,seed=seed,rule='Median and minimum family-target mean valid mode count; deterministic ID tie-break, no filtering',
        examples={k:dict(family=v[0],target=v[1]) for k,v in examples.items()},
        scope='All M8 saved predictions; checker labels used for diagnostic line style only, never model inference or q.'))
    for name,(family,target) in examples.items():
        fig=plt.figure(figsize=(13,8),layout='constrained');grid=fig.add_gridspec(2,3,height_ratios=[1,1.4])
        ids=[family+'_'+v+'_target'+target for v in ('open','closed','shifted')]
        allpoints=np.concatenate([paths[i].reshape(-1,3) for i in ids])
        lo=np.minimum(allpoints.min(0),[-.02,-.35,.74]);hi=np.maximum(allpoints.max(0),[.55,.35,1.05])
        for col,(variant,ident) in enumerate(zip(('open','closed','shifted'),ids)):
            top=fig.add_subplot(grid[0,col]);top.imshow(plt.imread(observations[ident]['image']));top.axis('off')
            top.set_title(variant+' / '+observations[ident]['instruction'],fontsize=10)
            ax=fig.add_subplot(grid[1,col],projection='3d');ref=refs[ident];row=rows[ident]
            for center,half in zip(ref['truth']['obstacle_centers'],ref['truth']['obstacle_halfsizes']):
                c,h=np.array(center),np.array(half)
                corners=np.array([c+h*np.array([i,j,k]) for i in (-1,1) for j in (-1,1) for k in (-1,1)])
                faces=[[corners[j] for j in face] for face in ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))]
                ax.add_collection3d(Poly3DCollection(faces,facecolors='.6',edgecolors='.45',alpha=.2,linewidths=.4))
            for index,(path,candidate,q) in enumerate(zip(paths[ident],row['candidates'],qs[ident])):
                ax.plot(*path.T,c=plt.get_cmap('viridis')(q),ls='-' if candidate['TipValid'] else '--',lw=1.7)
                ax.text(*path[8+index%8],str(index),fontsize=7)
            goal=ref['config']['goal_xyz'][int(target)];ax.scatter(*goal,c='black',marker='*',s=80)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
            ax.view_init(elev=30,azim=-58)
            qtext=' '.join('%d:%.2f'%(i,q) for i,q in enumerate(qs[ident]))
            ax.set_title('Valid %d/8; distinct modes %d\nq = %s'%(row['K']['8']['ValidCount'],row['K']['8']['GeometricModeCount'],qtext),fontsize=8)
        fig.suptitle('%s seed%d / %s / %s target%s\nSolid: checker-valid; dashed: checker-invalid. Color: model q, not checker output.'%(arm,seed,name,family,target),fontsize=11)
        fig.colorbar(cm.ScalarMappable(norm=colors.Normalize(0,1),cmap='viridis'),ax=fig.axes,label='Predicted path validity q',shrink=.5,pad=.025)
        fig.savefig(output/(name+'.png'),dpi=180);fig.savefig(output/(name+'.pdf'));plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.arm,a.seed,a.output)
