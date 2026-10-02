"""Plot actual saved draft/final effects; no prediction regeneration or filtering."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'reports/observed_refinement_analysis_v1/analysis.json'
    audit=json.loads(source.read_text())
    stages=['global_best','global_last','local_best','local_last']
    names=['Global\nbest2750','Global\nlast3000','Local\nbest1000','Local\nlast3000']
    colors=['#8496ac','#277b68']
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.8))
    for ax,key,label in zip(axes,['TipValidAtK','UniqueClassifiedTipValidAtK'],
                           ['Tip-valid candidates out of 96','Distinct classified tip-valid routes / instruction']):
        scale=96 if key=='TipValidAtK' else 1
        draft=np.array([audit['results'][s]['draft']['metrics'][key]*scale for s in stages])
        final=np.array([audit['results'][s]['metrics'][key]*scale for s in stages])
        if key=='TipValidAtK':np.testing.assert_allclose(draft,np.round(draft))
        x=np.arange(len(stages))
        for offset,values,color,title in [(-.17,draft,colors[0],'Saved draft'),(.17,final,colors[1],'Saved final')]:
            ax.bar(x+offset,values,.31,label=title,color=color)
            for xx,value in zip(x+offset,values):
                ax.text(xx,value+(.65 if scale==96 else .015),f'{value:.0f}' if scale==96 else f'{value:.3f}',ha='center',fontsize=8)
        ax.set_xticks(x,names);ax.set_ylabel(label)
        ax.set_ylim(0,61 if scale==96 else 1.4)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].set_title('A. Within-checkpoint validity changes')
    axes[1].set_title('B. Within-checkpoint coverage changes')
    axes[0].legend(frameon=False,loc='upper left',ncol=2,fontsize=9)
    fig.suptitle('One draft update: 8 DEV parents, 24 instructions, one training seed',fontsize=12)
    fig.text(.5,.06,'Both arms: 4 drafts + 4 finals, 3000 x 32 updates. Training: global 145.1 s; local 201.8 s (+39.1%).',ha='center',fontsize=9)
    fig.text(.5,.015,'Drafts differ across arms after joint training. Box-tip checks only; no full-arm or execution certificate.',ha='center',fontsize=8.5)
    fig.subplots_adjust(left=.07,right=.98,top=.84,bottom=.2,wspace=.32)
    output=source.parent/'figures';output.mkdir(exist_ok=True)
    paths=[output/'draft_to_final.png',output/'draft_to_final.pdf']
    for path in paths:fig.savefig(path,dpi=180)
    plt.close(fig)
    provenance={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,Path(__file__),*paths]}
    (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')


if __name__=='__main__':main()
