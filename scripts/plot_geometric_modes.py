"""Portable figures and diagnostics from saved, hash-indexed evaluation outputs."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def diagnostic(folder):
    result=read(folder/'RESULTS.json');arms=('ordinary','balanced_probability')
    rows={a:read(folder/(a+'_rows.json')) for a in arms}
    paired={};rng=np.random.default_rng(20261004)
    for a in arms:
        rr=rows[a];parents=sorted({r['parent'] for r in rr});draw=rng.integers(len(parents),size=(10000,len(parents)))
        delta=np.array([np.mean([r['K']['1']['ValidCount']-r['K']['8']['ValidCount']/8 for r in rr if r['parent']==p]) for p in parents])
        paired[a]=dict(q_vs_random_validity_gain=float(delta.mean()),CI95=np.percentile(delta[draw].mean(1),[2.5,97.5]).tolist(),
            unknown_only_groups=sum(r['unknown_only_modes'] for r in rr),
            conditions_with_unknown_only_groups=sum(r['unknown_only_modes']>0 for r in rr),
            selected_top1_invalid=[dict(id=r['id'],candidate=r['K']['1']['selected'][0],q=r['q'][r['K']['1']['selected'][0]]) for r in rr if not r['K']['1']['ValidCount']],
            all_invalid=[r['id'] for r in rr if r['K']['8']['ValidCount']==0],
            q_k2_lost_multimode=[r['id'] for r in rr if r['K']['8']['TwoDistinctValid'] and not r['K']['2']['TwoDistinctValid']],
            q_k4_lost_multimode=[r['id'] for r in rr if r['K']['8']['TwoDistinctValid'] and not r['K']['4']['TwoDistinctValid']])
    if 'ordinary_seed1' in result:
        seedrows={a:[] for a in arms}
        for a in arms:
            seedrows[a]=[rows[a]]+[read(folder/('%s_seed%d_rows.json'%(a,s))) for s in (1,2)]
        parent_ids=sorted({r['parent'] for r in rows[arms[0]]});deltas=[]
        for p in parent_ids:
            deltas.append(np.mean([r['K']['8']['GeometricModeCount'] for rr in seedrows[arms[1]] for r in rr if r['parent']==p])-np.mean([r['K']['8']['GeometricModeCount'] for rr in seedrows[arms[0]] for r in rr if r['parent']==p]))
        draw=rng.integers(len(parent_ids),size=(10000,len(parent_ids)))
        paired['three_generator_seeds']=dict(mean_modes={a:float(np.mean([r['K']['8']['GeometricModeCount'] for rr in seedrows[a] for r in rr])) for a in arms},
            mode_difference=float(np.mean(deltas)),CI95=np.percentile(np.array(deltas)[draw].mean(1),[2.5,97.5]).tolist(),
            scope='Three existing generator seeds, same pretrained parent; q only evaluated for frozen seed0 generator.')
    write(folder/'DIAGNOSTICS.json',paired)
    with (folder/'CORE_TABLE.csv').open('w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['arm','CandidateValidRate','AnyValid@8','ValidCount@8','GeometricModeCount@8','TwoDistinctValid@2','TwoDistinctValid@4','ReferenceModeCoverage@4','qSelectedValid@1'])
        for a in arms:
            r=result[a];writer.writerow([a,r['CandidateValidRate'],r['AnyValidAt8'],r['K']['8']['ValidCount'],r['K']['8']['GeometricModeCount'],r['K']['2']['TwoDistinctValid'],r['K']['4']['TwoDistinctValid'],r['K']['4']['ReferenceModeCoverage'],r['K']['1']['ValidCount']])
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),layout='constrained')
    labels=['Ordinary','Balanced + pi'];colors=['#3465a4','#d87925']
    for i,a in enumerate(arms):
        r=result[a];known=r['CandidateValidRate']*8-r['UnknownValidCount']/r['requests'];unknown=r['UnknownValidCount']/r['requests']
        axes[0,0].bar(i,known,color='royalblue');axes[0,0].bar(i,unknown,bottom=known,color='darkorange');axes[0,0].bar(i,8-known-unknown,bottom=known+unknown,color='lightcoral')
        axes[0,1].bar(np.arange(3)+(i-.5)*.34,[r['K'][k]['GeometricModeCount'] for k in ('8','2','4')],width=.34,color=colors[i],label=labels[i])
        axes[0,2].bar(np.arange(3)+(i-.5)*.34,[100*r['K'][k]['TwoDistinctValid'] for k in ('8','2','4')],width=.34,color=colors[i])
        axes[1,0].bar(np.arange(3)+(i-.5)*.34,[100*r['K'][k]['ReferenceModeCoverage'] for k in ('8','2','4')],width=.34,color=colors[i])
        axes[1,1].bar(np.arange(3)+(i-.5)*.34,[100*r['CandidateValidRate'],100*r['K']['1']['ValidCount'],100*r['AnyValidAt8']],width=.34,color=colors[i])
        axes[1,2].bar(np.arange(9)+(i-.5)*.34,[r['ModeCountHistogram'][str(k)] for k in range(9)],width=.34,color=colors[i])
    axes[0,0].set_xticks([0,1],labels);axes[0,0].set_title('All 8 slots: known valid / unknown valid / invalid');axes[0,0].set_ylim(0,8)
    for ax,title in [(axes[0,1],'Mean distinct valid portal modes'),(axes[0,2],'TwoDistinctValid (%)'),(axes[1,0],'Discovered reference-mode coverage (%)')]:
        ax.set_xticks([0,1,2],['M8 pool','q K2','q K4']);ax.set_title(title)
    axes[0,1].legend();axes[0,2].set_ylim(0,100);axes[1,0].set_ylim(0,100)
    axes[1,1].set_xticks([0,1,2],['Uniform random','q Top1','AnyValid@8']);axes[1,1].set_title('Fixed-pool selected validity (%)');axes[1,1].set_ylim(0,100)
    axes[1,2].set_xticks(range(9));axes[1,2].set_title('Request histogram, including zero valid');axes[1,2].set_xlabel('Number of distinct valid modes')
    fig.suptitle('%s | Same frozen seed0 models; no threshold tuning'%folder.name)
    fig.savefig(folder/'CORE_RESULTS.png',dpi=160);plt.close(fig)
    return rows

def examples(folder,rows):
    arms=list(rows);a,b=[rows[k] for k in arms];refs=read(folder/'REFERENCE_GEOMETRY.json')
    pools={k:dict(np.load(folder/(k+'_pool.npz'))) for k in arms}
    gain=[y['K']['8']['GeometricModeCount']-x['K']['8']['GeometricModeCount'] for x,y in zip(a,b)]
    chosen=[('largest_mode_gain',int(np.argmax(gain))),('largest_mode_loss',int(np.argmin(gain)))]
    for reason,predicate in [('all_invalid',lambda r:r['K']['8']['ValidCount']==0),('K2_loses_mode',lambda r:r['K']['8']['TwoDistinctValid'] and not r['K']['2']['TwoDistinctValid']),('high_q_invalid',lambda r:not r['K']['1']['ValidCount']),('unknown_only',lambda r:r['unique_classified']==0 and r['K']['8']['GeometricModeCount']>=2)]:
        matches=[i for i,r in enumerate(b) if predicate(r)]
        if matches:chosen.append((reason,matches[0]))
    write(folder/'CASE_SELECTION.json',[dict(reason=reason,id=b[n]['id']) for reason,n in chosen])
    for reason,n in chosen:
        identifier=b[n]['id'];ref=refs[identifier];centers=np.array(ref['truth']['obstacle_centers']);halves=np.array(ref['truth']['obstacle_halfsizes'])
        fig=plt.figure(figsize=(16,10),layout='constrained')
        for row,arm in enumerate(arms):
            r=rows[arm][n];paths=pools[arm]['paths'][n]
            for col,projection in enumerate([(0,1),(0,2),None]):
                ax=fig.add_subplot(2,3,row*3+col+1,projection='3d' if projection is None else None)
                if projection:
                    u,v=projection
                    for c,h in zip(centers,halves):ax.add_patch(Rectangle((c[u]-h[u],c[v]-h[v]),2*h[u],2*h[v],color='gray',alpha=.2))
                for p in np.array(ref['paths']):
                    if projection:ax.plot(p[:,u],p[:,v],color='gray',alpha=.22,lw=.8)
                    else:ax.plot(*p.T,color='gray',alpha=.22,lw=.8)
                for i,p in enumerate(paths):
                    color='crimson' if not r['valid'][i] else ('darkorange' if r['candidates'][i]['declared_passage_type'] is None else 'royalblue')
                    width=2.8 if i in r['K']['2']['selected'] else 1.1
                    kw=dict(color=color,lw=width,ls='-' if r['valid'][i] else '--',alpha=.85)
                    if projection:
                        ax.plot(p[:,u],p[:,v],**kw);ax.text(p[12,u],p[12,v],str(i),fontsize=9,color=color)
                    else:ax.plot(*p.T,**kw)
                if projection:ax.set_xlabel('xyz'[u]+' (m)');ax.set_ylabel('xyz'[v]+' (m)');ax.set_aspect('equal',adjustable='datalim')
                else:ax.set_xlabel('x');ax.set_ylabel('y');ax.set_zlabel('z');ax.view_init(25,-65)
                ax.set_title('%s | valid %d/8, modes %d\nK2=%s K4=%s'%(arm,r['K']['8']['ValidCount'],r['K']['8']['GeometricModeCount'],r['K']['2']['selected'],r['K']['4']['selected']),fontsize=10)
        fig.suptitle('%s — %s\nBlue: known valid; orange: unknown valid; red dashed: invalid; gray: references. Thick: q K2.'%(reason,identifier),fontsize=14)
        fig.savefig(folder/('CASE_'+reason+'.png'),dpi=150);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path);args=p.parse_args();examples(args.folder,diagnostic(args.folder))
