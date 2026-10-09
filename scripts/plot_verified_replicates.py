"""Render committed development summaries; no model/data-payload access."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main(folder):
    root=Path(folder)
    result=json.loads((root/'replication/RESULTS.json').read_text())
    parent=json.loads((root/'joint_seed0/RESULTS.json').read_text())['metrics']['parent']
    arms=['ordinary','budget_match','gate','set_point']
    labels=['Ordinary','Cost assignment','Validity gate','Verified point set']
    colors=['#64748b','#b7791f','#207968','#4d5fc1']
    fig,axes=plt.subplots(2,2,figsize=(10.5,6.7),constrained_layout=True)
    panels=[('valid_fraction','Valid candidates @8 (%)',100,parent['raw']['valid_fraction']),
            ('distinct','Distinct valid modes @8',1,parent['raw']['distinct']),
            ('recall','Known-mode recall @8 (%)',100,parent['raw']['recall']),
            ('selected_distinct','Selected distinct modes @4',1,parent['selected']['distinct'])]
    for ax,(key,title,scale,baseline) in zip(axes.flat,panels):
        for i,arm in enumerate(arms):
            values=np.array(result['summary'][arm][key]['seed_values'])*scale
            ax.scatter(i+np.array([-.1,0,.1]),values,s=27,color=colors[i],alpha=.8,zorder=3)
            ax.plot([i-.23,i+.23],[values.mean()]*2,color=colors[i],lw=3,zorder=3)
        ax.axhline(baseline*scale,color='#333333',ls='--',lw=1,label='Historical parent')
        ax.set_title(title,loc='left',fontsize=11)
        ax.set_xticks(range(4),labels,fontsize=8)
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(-.5,3.5)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.suptitle('Fixed-budget route generation: improvements and tradeoffs\n'
                 '288 DEV requests / 32 layout families / 3 continuation seeds from one parent',fontsize=13)
    out=root/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'replication.png',dpi=180)
    fig.savefig(out/'replication.pdf')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',default='reports/verified_set_v1')
    main(p.parse_args().folder)
