"""Static scientific figures from sealed result exports; no model calls."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot(folder):
    folder=Path(folder);data=json.loads((folder/'RESULTS.json').read_text())
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    scores=data['scorer_families']['M4']['scorers']
    metrics=[scores['q_seed%d'%i]['calibration']['dev_model_uncalibrated'] for i in range(3)]
    m=metrics[0]
    names=['First','Random\nexpectation','Shortest','q seed0','q seed1','q seed2','AnyValid\nceiling']
    values=[m['first_valid'],m['random_expected_valid'],m['shortest_valid']]+[r['selected_valid'] for r in metrics]+[m['any_valid']]
    ax=axes[0,0];bars=ax.bar(names,np.array(values)*100,color=['#aab7c4']*3+['#227c9d']*3+['#b7a57a'])
    ax.set_ylabel('Selected route valid (%)');ax.set_ylim(0,100)
    ax.set_title('Same M4 pool: old DEV_MODEL, 12 parents / 36 conditions')
    ax.bar_label(bars,fmt='%.1f',padding=3);ax.tick_params(axis='x',labelsize=8)
    ax=axes[0,1];pos=np.arange(3);before=[m['brier'] for m in metrics]
    after=[scores['q_seed%d'%i]['calibration']['dev_model_calibrated']['brier'] for i in range(3)]
    ax.bar(pos-.18,before,.36,label='Raw sigmoid',color='#227c9d');ax.bar(pos+.18,after,.36,label='Temperature calibrated',color='#e3a458')
    ax.set_xticks(pos,['seed0','seed1','seed2']);ax.set_ylabel('Brier score (lower is better)');ax.legend()
    ax.set_title('Calibration: no consistent development improvement')
    ax=axes[1,0];families=list(data['paired_comparisons']);pos=np.arange(len(families));width=.32
    for offset,arm,color in [(-width/2,'ordinary','#aab7c4'),(width/2,'balanced','#227c9d')]:
        vals=[data['paired_comparisons'][k][arm]['TipValidAtK']*100 for k in families]
        bars=ax.bar(pos+offset,vals,width,label=arm,color=color);ax.bar_label(bars,fmt='%.1f')
    ax.set_xticks(pos,['95 training parents' if x=='v2' else '191 training parents' for x in families])
    ax.set_ylabel('Candidate tip validity (%)');ax.set_ylim(0,100);ax.legend();ax.set_title('M8 paired generators: fixed last3000')
    ax=axes[1,1]
    for offset,arm,color in [(-width/2,'ordinary','#aab7c4'),(width/2,'balanced','#227c9d')]:
        vals=[data['paired_comparisons'][k][arm]['UniqueClassifiedTipValidAtK'] for k in families]
        bars=ax.bar(pos+offset,vals,width,label=arm,color=color);ax.bar_label(bars,fmt='%.2f')
    ax.set_xticks(pos,['95 training parents' if x=='v2' else '191 training parents' for x in families]);ax.set_ylim(0,max(2,ax.get_ylim()[1]*1.2))
    ax.set_ylabel('Mean distinct classified valid routes');ax.set_title('Incomplete known types; valid unknown routes retained')
    fig.suptitle('Observed multi-route planning: measured development evidence',fontsize=15)
    fig.savefig(folder/'RESULTS.png',dpi=160);plt.close(fig)

    # Every old DEV condition is shown, with full path extents and q labels.
    pool=np.load(folder/'M4/DEV_MODEL/predictions.npz',allow_pickle=False)
    pred=np.load(folder/'M4/calibration_seed0/predictions.npz',allow_pickle=False)
    temperature=scores['q_seed0']['calibration']['temperature']
    q=1/(1+np.exp(-pred['dev_logits']/temperature))
    for start in range(0,len(pool['ids']),12):
        fig,axes=plt.subplots(3,4,figsize=(15,10),constrained_layout=True)
        for local,ax in enumerate(axes.flat):
            index=start+local
            if index>=len(pool['ids']):ax.axis('off');continue
            paths=pool['paths'][index];y=pool['labels'][index];chosen=int(q[index].argmax())
            for k,path in enumerate(paths):
                ax.plot(path[:,0],path[:,1],color=plt.cm.tab10(k),linewidth=2.5 if k==chosen else 1,
                    linestyle='-' if y[k] else '--',label=f'{k}: q={q[index,k]:.2f}, valid={int(y[k])}')
            ax.scatter(paths[0,0,0],paths[0,0,1],c='black',marker='s',s=14)
            ax.set_title(str(pool['ids'][index]).replace('two_row_reach_',''),fontsize=10)
            ax.set_aspect('equal',adjustable='datalim');ax.legend(fontsize=6,loc='best');ax.set_xlabel('world x (m)');ax.set_ylabel('world y (m)')
        fig.suptitle('All M4 paths, XY projection (not a 3D collision proof); bold = q selected; dashed = checker-invalid')
        fig.savefig(folder/('ALL_DEV_PATHS_%02d.png'%start),dpi=140);plt.close(fig)
    print(str(folder/'RESULTS.png'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');a=p.parse_args();plot(a.folder)
