"""Static research figures from completed saved predictions only."""
import argparse
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write
from scripts.paired_modes_data import RUN


def setup():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    return plt


def main(arms,seeds,output):
    plt=setup();output=Path(output);output.mkdir(parents=True,exist_ok=False)
    summaries={(a,s):read(RUN/'evaluation/paired_dev'/('%s_seed%d'%(a,s))/'RESULTS.json') for a in arms for s in seeds}
    fig,axes=plt.subplots(1,4,figsize=(14,3.6),layout='constrained')
    metrics=[('Candidate validity (%)',lambda r:r['CandidateValidRate']*100),
             ('Distinct valid modes @8',lambda r:r['K']['8']['GeometricModeCount']),
             ('Shared relation recall (%)',lambda r:r['paired']['metrics']['shared_recall']*100),
             ('Opened relation recall (%)',lambda r:r['paired']['metrics']['opened_recall']*100)]
    for ax,(title,extract) in zip(axes,metrics):
        for i,arm in enumerate(arms):
            values=[extract(summaries[arm,s]) for s in seeds]
            ax.bar(i,np.mean(values),color={'R0':'#8495a7','R1':'#66a9a1','R_full':'#bdb0d9','R2':'#e19c55','R3':'#7463a8'}.get(arm,'#8495a7'),width=.65)
            ax.scatter(np.full(len(values),i)+np.linspace(-.08,.08,len(values)),values,color='#273746',s=20,zorder=4)
        ax.set_xticks(range(len(arms)),arms);ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle('Paired-scene development results — fixed last checkpoints; dots are generator seeds',fontsize=12)
    fig.savefig(output/'main_results.png',dpi=200);fig.savefig(output/'main_results.pdf');plt.close(fig)
    if len(arms)>=2:qualitative(plt,arms,output)


def qualitative(plt,arms,output):
    rows={};pools={};refs={}
    for arm in arms:
        folder=RUN/'evaluation/paired_dev'/(arm+'_seed0')
        rows[arm]={r['id']:r for r in read(folder/'rows.json')}
        with np.load(folder/'pool.npz') as a:pools[arm]={str(i):p for i,p in zip(a['ids'],a['paths'])}
        refs[arm]=read(folder/'REFERENCE_GEOMETRY.json')
    keys={}
    for ident in rows[arms[0]]:
        scene,target=ident.rsplit('_target',1);family=scene.rsplit('_',1)[0]
        keys.setdefault((family,target),[]).append(ident)
    delta={k:np.mean([rows[arms[-1]][i]['K']['8']['GeometricModeCount']-rows[arms[-2]][i]['K']['8']['GeometricModeCount'] for i in ids]) for k,ids in keys.items() if len(ids)==3}
    examples={'largest_gain':max(delta,key=delta.get),'largest_loss':min(delta,key=delta.get)}
    write(output/'qualitative_selection.json',dict(rule='Seed0 family-target mean mode change between last two arms; show maximum and minimum without filtering failures',
        examples={name:dict(family=k[0],target=k[1],delta=float(delta[k])) for name,k in examples.items()}))
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    def boxes(ax,truth):
        for c,h in zip(truth['obstacle_centers'],truth['obstacle_halfsizes']):
            c,h=np.array(c),np.array(h)
            corners=np.array([c+h*np.array([i,j,k]) for i in (-1,1) for j in (-1,1) for k in (-1,1)])
            faces=[[corners[j] for j in face] for face in ((0,1,3,2),(4,5,7,6),(0,1,5,4),(2,3,7,6),(0,2,6,4),(1,3,7,5))]
            ax.add_collection3d(Poly3DCollection(faces,facecolors='.55',edgecolors='.45',alpha=.2,linewidths=.4))
    for name,(family,target) in examples.items():
        fig=plt.figure(figsize=(12,3.6*len(arms)),layout='constrained')
        allpaths=np.concatenate([pools[a][family+'_'+v+'_target'+target].reshape(-1,3) for a in arms for v in ('open','closed','shifted')])
        low=np.minimum(allpaths.min(0),[-.02,-.3,.74]);high=np.maximum(allpaths.max(0),[.53,.3,1.02])
        for r,arm in enumerate(arms):
            for col,variant in enumerate(('open','closed','shifted')):
                ident=family+'_'+variant+'_target'+target;ax=fig.add_subplot(len(arms),3,r*3+col+1,projection='3d')
                ref=refs[arm][ident];record=rows[arm][ident];boxes(ax,ref['truth'])
                for path,candidate in zip(pools[arm][ident],record['candidates']):
                    valid=candidate['TipValid'];ax.plot(*path.T,color='#187b70' if valid else '#ce5950',ls='-' if valid else '--',lw=1.3,alpha=.8)
                goal=ref['config']['goal_xyz'][int(target)];ax.scatter(*goal,marker='*',s=85,c='black',depthshade=False)
                ax.set(xlim=(low[0],high[0]),ylim=(low[1],high[1]),zlim=(low[2],high[2]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
                ax.view_init(elev=27,azim=-58);ax.set_title('%s / %s: valid %d/8, modes %d'%(arm,variant,record['K']['8']['ValidCount'],record['K']['8']['GeometricModeCount']))
        fig.suptitle('%s — %s, target%s\nGreen solid: valid; red dashed: invalid. All eight predictions shown.'%(name.replace('_',' '),family,target),fontsize=12)
        fig.savefig(output/(name+'.png'),dpi=160);fig.savefig(output/(name+'.pdf'));plt.close(fig)


def reliability(folder,output):
    plt=setup();summary=read(Path(folder)/'summary.json');output=Path(output);output.mkdir(parents=True,exist_ok=False)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,split in zip(axes,('paired_dev','old_dev','dev32')):
        results=summary['results'][split]
        ax.plot([0,1],[0,1],c='.6',ls='--',lw=1)
        for key,label,color in [('frozen_transfer','Frozen q','#8495a7'),('uncalibrated','Refit q','#66a9a1'),('calibrated','Refit + temperature','#e19c55')]:
            bins=[b for b in results[key]['reliability_bins'] if b['n']]
            ax.plot([b['confidence'] for b in bins],[b['valid'] for b in bins],marker='o',ms=4,label=label,color=color)
        ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted validity',ylabel='Observed validity',title=split);ax.grid(alpha=.15)
    axes[-1].legend(loc='upper left',fontsize=8)
    fig.suptitle('Actual-route reliability — calibration fitted on a separate historical role',fontsize=12)
    fig.savefig(output/'calibration.png',dpi=200);fig.savefig(output/'calibration.pdf');plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['R0','R1','R2']);p.add_argument('--seeds',nargs='+',type=int,default=[0]);p.add_argument('--output',required=True);p.add_argument('--reliability-folder')
    a=p.parse_args()
    if a.reliability_folder:reliability(a.reliability_folder,a.output)
    else:main(a.arms,a.seeds,a.output)
