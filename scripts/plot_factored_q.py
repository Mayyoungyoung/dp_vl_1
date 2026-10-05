"""Publication-style development plots from saved scores; no new inference."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.run_observed_probability import read,write


def plot(root,output,arms=None):
    root,output=Path(root),Path(output);output.mkdir(parents=True,exist_ok=False)
    arms=arms or ['single','joint','marginal','conditional']
    colors=['#555555','#278087','#cf8b2e','#7655a3','#c14356']
    labels={'single':'Single q','joint':'Matched joint','marginal':'Marginal product','conditional':'Conditional product','conditional_endpoint':'Endpoint task product'}
    data={a:[read(root/('%s_g%d_s%d'%(a,g,s))/'evaluation_v2.json')['results']['paired_dev']['calibrated'] for g in range(3) for s in range(3)] for a in arms}
    fig,axes=plt.subplots(1,4,figsize=(15,4.4))
    fields=[('brier','Joint Brier (lower better)',1),('top1','Top-1 valid (%)',100),('k4_all','All four valid (%)',100),('k4_modes','Distinct valid modes / four',1)]
    for ax,(key,title,scale) in zip(axes,fields):
        for i,a in enumerate(arms):
            v=np.array([r['set_metrics'][key] for r in data[a]])*scale
            ax.bar(i,v.mean(),color=colors[i],alpha=.75)
            means=v.reshape(3,3).mean(1)
            ax.scatter(i+np.array([-.13,0,.13]),means,color='black',s=16,zorder=4)
        ax.set_xticks(range(len(arms)));ax.set_xticklabels([labels[a] for a in arms],rotation=28,ha='right',fontsize=8)
        ax.set_title(title,fontsize=10);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Frozen R1: 3 generator seeds x 3 scorer seeds | reused paired DEV\nBars: all-cell means; dots: generator means over three scorer seeds',fontsize=11)
    fig.tight_layout(rect=[0,0,1,.88])
    for ext in ('png','pdf'):fig.savefig(output/('main.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.5))
    for i,a in enumerate(arms):
        xs=[];ys=[]
        for b in range(10):
            bins=[r['reliability_bins'][b] for r in data[a]];n=sum(v['n'] for v in bins)
            if n:
                xs.append(sum(v['n']*(v['confidence'] or 0) for v in bins)/n)
                ys.append(sum(v['n']*(v['valid'] or 0) for v in bins)/n)
        axes[0].plot(xs,ys,'o-',color=colors[i],label=labels[a],markersize=3)
        q=[];y=[]
        for g in range(3):
            for s in range(3):
                with np.load(root/('%s_g%d_s%d'%(a,g,s))/'evaluation_v2/paired_dev.npz') as z:
                    q.extend(z['q'].ravel());y.extend(z['labels'].ravel())
        q=np.array(q);y=np.array(y);order=np.argsort(-q,kind='stable');risk=np.cumsum(1-y[order])/np.arange(1,len(q)+1)
        axes[1].plot(np.arange(1,len(q)+1)/len(q),risk,color=colors[i],label=labels[a])
    axes[0].plot([0,1],[0,1],'--',color='grey');axes[0].set(xlabel='Predicted joint probability',ylabel='Observed validity',xlim=(0,1),ylim=(0,1))
    axes[1].set(xlabel='Fraction of candidate pool retained',ylabel='Empirical invalid fraction',ylim=(0,.5))
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Calibration and candidate risk-coverage | natural unfiltered pools',fontsize=11);fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/('calibration.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    # A fixed public example, not selected using favorable scores.
    model='conditional_endpoint' if 'conditional_endpoint' in arms else 'conditional'
    manifest=read(root/(model+'_g0_s0')/'deployment_v2/manifest.json')
    with np.load(root/(model+'_g0_s0')/'evaluation_v2/paired_dev.npz') as z:
        idx=list(z['ids']).index(manifest['example_id']);qt=z['q_task'][idx];qf=z['q_feas'][idx];q=z['q'][idx];y=z['labels'][idx]
    fig,ax=plt.subplots(figsize=(9,4));x=np.arange(8)
    ax.bar(x-.24,qt,width=.24,label='Task matching');ax.bar(x,qf,width=.24,label='Feasible | task');ax.bar(x+.24,q,width=.24,label='Product')
    ax.set_xticks(x);ax.set_xticklabels(['%d\n%s'%(j+1,'valid' if y[j] else 'invalid') for j in x]);ax.set(ylim=(0,1.05),ylabel='Estimated probability',xlabel='Actual route candidate')
    ax.set_title(manifest['example_id']+' | '+labels[model],fontsize=10);ax.legend(fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    for ext in ('png','pdf'):fig.savefig(output/('factor_example.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)
    write(output/'plot_scope.json',dict(arms=arms,example_id=manifest['example_id'],calibration_version='evaluation_v2',scope='Development descriptive plots; dots are not confidence intervals; repeated model seeds are not independent scene samples.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--arms',nargs='+');a=p.parse_args();plot(a.root,a.output,a.arms)
