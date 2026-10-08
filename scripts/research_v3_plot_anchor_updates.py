"""Static scientific plot of all registered scratch batches, no selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    source=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    update=json.loads((source/'update.json').read_text());optimizer=json.loads((source/'optimizer.json').read_text())
    branches=[('Fresh Adam / straight-through',update,'straight_through_peak','#b45a47'),
              ('Fresh Adam / hard anchor',update,'hard_peak','#cf9b43'),
              ('Saved Adam / straight-through',optimizer,'restored_optimizer','#267d9a')]
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained')
    for j,(label,data,key,color) in enumerate(branches):
        loss=[];movement=[]
        for i,row in enumerate(data['rows']):
            assert row['batch']==i and row['ids']==update['rows'][i]['ids']
            v=row['modes'][key];before=sum(v['before'].values());after=sum(v['after'].values())
            loss.append(100*(after-before)/before);movement.append(100*v['maximum_path_change_m'])
        for ax,values in zip(axes,(loss,movement)):
            ax.bar(np.arange(4)+(j-1)*.24,values,.23,label=label,color=color)
    for ax in axes:
        ax.set_xticks(np.arange(4),[1,2,3,4]);ax.set_xlabel('Fixed TRAIN batch (32 requests)')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].axhline(0,color='#333333',lw=.7);axes[0].set_ylabel('Change in full objective after one step (%)')
    axes[1].set_ylabel('Maximum path-coordinate change (cm)')
    fig.suptitle('First-step instability is much larger than the anchor-gradient effect',fontsize=13)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False,fontsize=9)
    fig.savefig(out/'scratch_updates.png',dpi=180);fig.savefig(out/'scratch_updates.svg');plt.close(fig)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (source/'update.json',source/'optimizer.json')}
    (out/'SOURCES.json').write_text(json.dumps(hashes,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.source,a.output)
