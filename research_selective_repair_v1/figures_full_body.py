"""Actual diagnostic figures; no paper-positive or independence claims."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
def read(n):return json.loads((HERE/'results'/n).read_text(encoding='utf-8'))

def main():
    b=read('FULL_TRAIN_SUMMARY_20261010_0835_V2.json')['reports']
    constant=b['body_crossing_constant_TRAIN_v1/SUMMARY.json']['records']
    names=['GRU position','MLP position','GRU category','MLP category','Constant residual','Planned + binary'];values=[]
    for k,a in [('recurrent','analytic'),('nonrecurrent','analytic'),('recurrent','categorical_aux'),('nonrecurrent','categorical_aux')]:
        values.append([b[f'body_crossing_{k}_{a}_seed{s}_full_v1/COMPOSED_DIAGNOSTICS.json']['held_TRAIN']['conditional_word_accuracy'] for s in range(3)])
    for c in ('constant_position_residual','planned_word_binary'):
        values.append([r['held_TRAIN']['conditional_word_accuracy'] for r in constant if r['control']==c])
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.1),gridspec_kw={'width_ratios':[1.6,1]})
    colors=['#3670A0','#3670A0','#808080','#808080','#C7974C','#202020']
    x=np.arange(len(names));axes[0].bar(x,np.mean(values,1),color=colors,alpha=.8,width=.62)
    for i,v in enumerate(values):axes[0].scatter(i+np.linspace(-.12,.12,3),v,color='black',s=16,zorder=3)
    axes[0].set_xticks(x,names,rotation=23,ha='right');axes[0].set_ylim(0,1);axes[0].set_ylabel('Conditional realized-word accuracy')
    axes[0].set_title('Full-feedback TRAIN diagnostic: 177 positive routes')
    spatial=read('SPATIAL_TRAIN_RETURNED4_V1.json')['records'];labels=[r['method'] for r in spatial];v=[r['E4'] for r in spatial]
    axes[1].bar(np.arange(len(v)),v,color=['#3670A0','#3670A0','#202020','#808080','#C7974C'],alpha=.8,width=.65)
    axes[1].set_xticks(np.arange(len(v)),labels,rotation=23,ha='right');axes[1].set_ylim(0,4);axes[1].set_ylabel('Actual executable distinct words@4')
    axes[1].set_title('XYZ repair: four held TRAIN requests')
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.text(.04,.01,'Dots: three paired head-training seeds with shared C0. Neither panel is fresh-layout confirmation; native execution varies.',fontsize=9)
    fig.subplots_adjust(bottom=.27,top=.9,wspace=.35,left=.065,right=.985)
    out=HERE/'figures/full_body_diagnostics_v1';out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(out/('mechanism_diagnostics.'+ext),dpi=220)
    plt.close(fig)
    print(out)

if __name__=='__main__':main()
