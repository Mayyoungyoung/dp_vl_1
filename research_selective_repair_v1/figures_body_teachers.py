"""Measured actual TRAIN execution tradeoff; oracle opportunity is labeled."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    root=Path(__file__).resolve().parent
    data=json.loads((root/'results/BODY_EXECUTION_TEACHERS.json').read_text(encoding='utf-8-sig'))
    names=['body_execution_train_lift0_pilot_v1','body_execution_train_lift.08_pilot_v1','body_execution_train_preservedlift_pilot_v1']
    labels=['Original','Uniform 8 cm lift','Crossing-preserved lift'];colors=['#7892a6','#bd896a','#3d8196']
    fig,ax=plt.subplots(1,2,figsize=(10,4));x=np.arange(4)
    for j,name in enumerate(names):
        d=data['summaries'][name]['requests_results'];offset=(j-1)*.24
        ax[0].bar(x+offset,[r['successful_clear_routes']/8 for r in d],.23,label=labels[j],color=colors[j])
        ax[1].bar(x+offset,[r['executable_distinct_words'] for r in d],.23,label=labels[j],color=colors[j])
    for a in ax:
        a.set_xticks(x,['TRAIN000 open','TRAIN000 closed','TRAIN001 open','TRAIN001 closed'],rotation=18,ha='right');a.spines[['top','right']].set_visible(False)
    ax[0].set(ylabel='Actual successful-clear fraction / eight',ylim=(0,1.1))
    bounds=[r['finite8_best_with_original_words_preserved'] for r in data['finite_slot_opportunity']]
    ax[1].scatter(x,bounds,marker='_',s=450,color='#a43d40',label='Finite-eight TRAIN oracle opportunity');ax[1].set(ylabel='Actual distinct executable words / eight',ylim=(0,8))
    fig.suptitle('Fixed Panda TRAIN feedback: execution success and coverage diverge',fontsize=11)
    fig.legend(*ax[1].get_legend_handles_labels(),loc='lower center',bbox_to_anchor=(.5,.03),ncol=2,fontsize=8,frameon=False)
    fig.text(.5,-.005,'All eight routes are teacher trials. Not returned-four advantage or independent confirmation.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.16,1,.98));out=root/'figures/body';out.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):fig.savefig(out/('actual_teacher_tradeoff.'+ext),dpi=180,bbox_inches='tight')
    plt.close(fig)

if __name__=='__main__':main()
