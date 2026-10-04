"""Static figures from saved candidate pools; no new inference or metrics."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(root):
    result=json.loads((root/'RESULTS.json').read_text())
    fig,axes=plt.subplots(2,4,figsize=(13,6),layout='constrained')
    for row,split in enumerate(('old_dev','dev32')):
        for col,(key,title) in enumerate((('CandidateValidRate','Candidate valid (%)'),('CollisionFailureRate','Collision failure (%)'),('GoalFailureRate','Target failure (%)'),('GeometricModeCount8','Geometric modes @8'))):
            ax=axes[row,col];values=[result[split][a+'_seed0'][key]*(1 if col==3 else 100) for a in ('B','C','D')]
            ax.bar(['B','C','D'],values,color=['#426eaa','#c65746','#db9c38'])
            ax.bar_label(ax.containers[0],fmt='%.2f',padding=3,fontsize=9)
            ax.set_ylim(0,max(values)*1.23);ax.set_title(split+' / '+title,fontsize=10);ax.grid(axis='y',alpha=.15)
    fig.suptitle('Paired seed0, fixed last3000 | B clearance; C goal-decoupled; D adaptive constraint',fontsize=12)
    fig.savefig(root/'SEED0_COMPARISON.png',dpi=160);plt.close(fig)
    fig=plt.figure(figsize=(13,8),layout='constrained')
    baseline=Path('reports/segment_clearance_v1/evaluation')
    for row,(split,scene) in enumerate((('old_dev','two_row_reach_283268_target1'),('dev32','two_row_reach_400269_target0'))):
        references=json.loads((baseline/split/'B_seed0/REFERENCE_GEOMETRY.json').read_text())[scene]
        refs=np.array(references['paths']);cloud=[refs.reshape(-1,3)];pools=[]
        for arm in ('B','C','D'):
            folder=(baseline if arm=='B' else root/'evaluation')/split/(arm+'_seed0')
            with np.load(folder/'pool.npz') as p:
                n=list(p['ids']).index(scene);paths=p['paths'][n];labels=p['labels'][n]
            rows=json.loads((folder/'rows.json').read_text());r=next(r for r in rows if r['id']==scene)
            pools.append((arm,paths,labels,r));cloud.append(paths.reshape(-1,3))
        cloud=np.concatenate(cloud);lo=cloud.min(0)-.03;hi=cloud.max(0)+.03
        for col,(arm,paths,labels,r) in enumerate(pools):
            ax=fig.add_subplot(2,3,row*3+col+1,projection='3d')
            for ref in refs:ax.plot(*ref.T,color='gray',alpha=.15,lw=.7)
            for path,valid in zip(paths,labels):
                color='#258055' if valid else '#ba493d';ax.plot(*path.T,color=color,alpha=.65,lw=1)
                ax.scatter(*path[-1],color=color,s=16)
            ax.scatter(*refs[:,-1].T,marker='x',color='black',s=35)
            goal_fail=sum(not c['semantic_goal_correct'] for c in r['candidates'])
            err=np.mean([c['endpoint_error_m'] for c in r['candidates']])*100
            ax.set_title('%s / %s\nvalid %d/8; target fail %d/8; endpoint %.1f cm'%(scene.replace('two_row_reach_',''),arm,sum(labels),goal_fail,err),fontsize=10)
            ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]),zlim=(lo[2],hi[2]),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
            ax.view_init(27,-63)
    fig.suptitle('Fixed failure cases, seed0 | green valid; red invalid; black x reference endpoints',fontsize=12)
    fig.savefig(root/'ENDPOINT_CASES.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();plot(a.root)
