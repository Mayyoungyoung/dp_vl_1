"""Standalone figures from sealed initial diagnostics, explicitly developmental."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def figures(receipts,output):
    output.mkdir(parents=True,exist_ok=True)
    read=lambda name:json.loads((receipts/name).read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    def save(fig,name):
        fig.savefig(output/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(output/(name+'.svg'),bbox_inches='tight');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,2.8));ax.set_xlim(0,10);ax.set_ylim(0,3);ax.axis('off')
    boxes=[(.05,'Current RGB-D\nlanguage / state'),(2.05,'Frozen center\n8 learned curves'),(4.05,'Observed evidence\nper-route repair'),(6.05,'Frozen complete q\n8 outputs -> 4'),(8.05,'Fixed Panda\nall-arm audit')]
    for x,label in boxes:
        ax.add_patch(FancyBboxPatch((x,1.3),1.7,.8,boxstyle='round,pad=.06',facecolor='#edf3f8',edgecolor='#315778'));ax.text(x+.85,1.7,label,ha='center',va='center')
        if x<8:ax.annotate('',xy=(x+2,1.7),xytext=(x+1.77,1.7),arrowprops={'arrowstyle':'->','color':'#315778'})
    ax.text(5,.72,'TRAIN only: independently checked repair / identity targets, geometry and support labels',ha='center',color='#914834')
    ax.text(5,.22,'Current-observation drafts are internal; exactly eight final candidates. Core gain remains unproven.',ha='center',fontsize=9)
    save(fig,'method_development_v1')
    fig,axes=plt.subplots(1,2,figsize=(10,3.8));names=['Center','Selective','Decoupled','Residual','Joint center']
    ds=[None,read('grid_selective.json'),read('grid_decoupled.json'),read('grid_residual.json'),read('grid_recenter.json')]
    means=[7.149305555555555]+[d['selected_mean'][0] for d in ds[1:]]
    axes[0].bar(np.arange(5),means,color=['#8798a5','#b88a83','#b88a83','#3b7a97','#739498']);axes[0].set_xticks(np.arange(5),names,rotation=20,ha='right');axes[0].set_ylim(6.95,7.4);axes[0].set_ylabel('Actual valid modes @8');axes[0].set_title('Reused DEV, seed0; screening head')
    for x,y in enumerate(means):axes[0].text(x,y+.01,'%.3f'%y,ha='center',fontsize=9)
    for d,label,color in [(ds[2],'Decoupled gate','#b15345'),(ds[3],'Ordinary residual scales','#397690')]:
        axes[1].plot([r['damaged'] for r in d['grid']],[r['repaired'] for r in d['grid']],'o-',label=label,color=color)
    axes[1].axvline(2107*.01,color='#777',ls='--',label='1% damage budget');axes[1].set_xlabel('Newly damaged valid drafts');axes[1].set_ylabel('Newly valid failed drafts');axes[1].set_title('Repair versus damage, common center');axes[1].legend(fontsize=8)
    fig.tight_layout();save(fig,'repair_damage_development_v1')
    a,b=read('executor_B0.json'),read('executor_rule.json');fig,ax=plt.subplots(figsize=(7,3.7));x=np.arange(4)
    ax.bar(x-.18,[r['success'] for r in a['records']],.36,label='Center actual returned4',color='#8798a5');ax.bar(x+.18,[r['success'] for r in b['records']],.36,label='Goal rule actual returned4',color='#3b7a97')
    ax.set_xticks(x,['Family128\nopen','Family128\nclosed','Family129\nopen','Family129\nclosed']);ax.set_ylabel('Panda successes / 4');ax.set_ylim(0,4.7);ax.legend(fontsize=9);ax.set_title('Fixed preliminary subset: 7/16 versus 8/16; no advantage established')
    fig.tight_layout();save(fig,'fixed_executor_development_v1')
    plan=read('execution_plan0.json');slot=read('executor_B0.json')['records'][0]['actual_returned_indices'][2];path=np.asarray(plan['execution_paths'])[slot]
    with np.load(receipts/'execution_rank2.npz') as z:trace=z['gripper_pose'][:,:3]
    fig=plt.figure(figsize=(6,4.8));ax=fig.add_subplot(111,projection='3d');ax.plot(*path.T,color='#397690',lw=2,label='Sealed generated route');ax.plot(*trace.T,color='#b15345',lw=2,label='Executed tip, stops at arm collision');ax.scatter(*trace[-1],color='#b15345',s=40)
    ax.set(xlabel='World x (m)',ylabel='World y (m)',zlabel='World z (m)',title='Tip-valid route can collide through the robot arm');ax.legend(fontsize=8);ax.view_init(elev=25,azim=-65)
    save(fig,'arm_collision_case_development_v1')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);figures(**vars(p.parse_args()))
