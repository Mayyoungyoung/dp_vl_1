"""Actual development figures; no confirmation/publication claim."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def figures(results,output):
    output.mkdir(parents=True,exist_ok=True)
    read=lambda p:json.loads((results/p).read_text(encoding='utf-8-sig'))
    screen=read('CONSTRAINT_SCREEN.json');transport=read('TRANSPORT_COMMON.json');execution=read('CONSTRAINT_EXECUTION.json')
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(output/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3));ax.axis('off');ax.set(xlim=(0,10),ylim=(0,3))
    blocks=[(.1,'Current RGB-D\nlanguage / public state'),(2.15,'Frozen C0\n8 current drafts'),(4.2,'Learned post constraints\nfinite edit operator'),(6.25,'Fixed full scorer\n8 final outputs -> 4'),(8.3,'Fixed Panda\nactual all-arm audit')]
    for x,label in blocks:
        ax.add_patch(FancyBboxPatch((x,1.35),1.55,.85,boxstyle='round,pad=.06',facecolor='#edf3f8',edgecolor='#315778'));ax.text(x+.775,1.775,label,ha='center',va='center',fontsize=9)
        if x<8:ax.annotate('',xy=(x+2,1.775),xytext=(x+1.65,1.775),arrowprops=dict(arrowstyle='->',color='#315778'))
    ax.text(5,.78,'TRAIN only: physical interventions, found repairs / identity, post extents',ha='center',color='#914834')
    ax.text(5,.3,'Same predicted constraints + 64 steps for all operators. Core selective advantage is not established.',ha='center',fontsize=9)
    save(fig,'observed_constraints_method')
    fig,axes=plt.subplots(1,2,figsize=(10,4));kinds=['global','protected','mode_global'];labels=['Geometry','Protected','Mode global'];x=np.arange(3)
    for a,pop in zip(axes,('old','interventions')):
        values=[screen['constraints_%s_%s_v1'%(k,pop)]['mean_utility'][0] for k in kinds]
        a.bar(x,values,color=['#8199ab','#3d8196','#bc886a']);a.set_xticks(x,labels);a.set_ylim(7.55,8.16);a.axhline(8,color='#555',ls='--');a.axhline(max(values)+.15,color='#a64c43',ls=':',label='Strongest + 0.15 gate');a.set_ylabel('Actual valid modes @8')
        for i,v in enumerate(values):a.text(i,v+.015,'%.3f'%v,ha='center',fontsize=9)
        a.set_title('Old DEV (32 families)' if pop=='old' else 'Physical intervention DEV (8 families)');a.legend(fontsize=8)
    fig.suptitle('Screening: one constraint-training seed, frozen C0 allocation head');fig.tight_layout();save(fig,'constraint_operator_comparison')
    fig,axes=plt.subplots(1,2,figsize=(10,4));names=['Center','Observed points','Geometry','Protected','Mode global']
    keys=['interventions_zero_calibrated_v1_transport_common','interventions_geometry_calibrated_v1_transport_common']+['constraints_%s_interventions_v1_transport_common'%k for k in kinds]
    values=[transport[k]['retained_words'] for k in keys];axes[0].bar(np.arange(5),np.array(values)/838,color=['#9babb6','#8199ab','#8199ab','#3d8196','#bc886a']);axes[0].set_xticks(np.arange(5),names,rotation=25,ha='right');axes[0].set_ylim(.85,.97);axes[0].set_ylabel('Retained feasible transported C0 words');axes[0].set_title('Common 838-word opportunity set')
    for i,v in enumerate(values):axes[0].text(i,v/838+.002,'%d/838'%v,ha='center',fontsize=8)
    added=[screen['constraints_%s_interventions_v1'%k]['added'] for k in kinds];lost=[screen['constraints_%s_interventions_v1'%k]['lost'] for k in kinds]
    axes[1].bar(x-.17,added,.34,label='Added actual words',color='#3d8196');axes[1].bar(x+.17,lost,.34,label='Lost original words',color='#bc886a');axes[1].set_xticks(x,labels);axes[1].set_ylabel('Words over 144 requests');axes[1].set_title('All have zero newly damaged routes');axes[1].legend(fontsize=8);fig.tight_layout();save(fig,'constraint_repair_preservation')
    fig,ax=plt.subplots(figsize=(8,4));x=np.arange(4)
    for offset,k,color in [(-.25,'global','#8199ab'),(0,'protected','#3d8196'),(.25,'mode_global','#bc886a')]:
        d=execution[k];ax.bar(x+offset,[r['success'] for r in d['records']],.24,color=color,label='%s %d/16'%(k.replace('_',' '),d['success']))
    ax.set_xticks(x,['Family016 open','Family016 closed','Family017 open','Family017 closed'],rotation=15,ha='right');ax.set_ylim(0,4.9);ax.set_ylabel('Panda successes / actual returned4');ax.set_title('Fixed preliminary subset; tip validity does not establish execution');ax.legend(fontsize=8);fig.tight_layout();save(fig,'constraint_fixed_executor')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path('research_selective_repair_v1/results'));p.add_argument('--output',type=Path,default=Path('research_selective_repair_v1/figures/constraints'));figures(**vars(p.parse_args()))
