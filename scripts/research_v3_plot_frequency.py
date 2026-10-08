"""Static figures from saved DEVELOPMENT rows, without selecting examples."""
import argparse
from pathlib import Path
import json
import numpy as np
from scripts.research_v3_analyze_frequency import row_metrics
from scripts.research_v3_frequency_integrity import initial_rows


def main(root,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root=Path(root);out=Path(output);out.mkdir(parents=True,exist_ok=False)
    arms=['initial','empirical_90','empirical_98','balanced','set_sampled','set_matching']
    labels=['Initial R1','90:10','98:2','Balanced','8 distinct\ntargets','Full set']
    data={a:(initial_rows(root) if a=='initial' else json.loads((root/('frequency_'+a)/'evaluation_fixed_q_v2/rows.json').read_text())) for a in arms}
    columns=[('valid','Candidate validity (%)',100),('modes8','Distinct valid passages @8',1),('rare8','Minority witness recall @8 (%)',100),
        ('brier','Brier (fixed complete scorer)',1),('modes4','Distinct valid passages @4',1),('all_valid4','All returned four valid (%)',100)]
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),layout='constrained')
    colors=['#7C8793','#C48354','#AA5C3F','#6CA89E','#4A93B7','#376080'];stats={}
    for ax,(key,title,scale) in zip(axes.flat,columns):
        means=[];errors=[];stats[key]={}
        for a in arms:
            rr=data[a];families=sorted({r['family'] for r in rr})
            values=np.array([np.mean([row_metrics(r)[key] for r in rr if r['family']==f]) for f in families])
            rng=np.random.default_rng(610091);ci=np.quantile(values[rng.integers(len(values),size=(10000,len(values)))].mean(1),[.025,.975])
            mean=float(values.mean());means.append(mean*scale);errors.append([(mean-ci[0])*scale,(ci[1]-mean)*scale])
            stats[key][a]=dict(mean=mean,CI95=ci.tolist())
        ax.bar(np.arange(len(arms)),means,color=colors,width=.72,yerr=np.array(errors).T,capsize=3,error_kw={'linewidth':1})
        ax.set_xticks(np.arange(len(arms)),labels,fontsize=9);ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
        if scale==100:ax.set_ylim(0,105)
    fig.suptitle('Frequency retention: ordinary methods and actual target budgets\nSingle generator seed; 95% intervals resample 32 development families',fontsize=13)
    fig.savefig(out/'frequency_controls.png',dpi=180);fig.savefig(out/'frequency_controls.svg');plt.close(fig)
    (out/'STATISTICS.json').write_text(json.dumps(stats,indent=2)+'\n')
    (out/'CAPTION.md').write_text('''# Frequency controls

Initial R1 has no extra training. All remaining arms receive1200x32 updates.
Balanced and the uniform arm are exactly identical, so they are shown once.
90:10,98:2,balanced and8-distinct arms process307,200 target slots each; full-set
matching processes1,566,630. These target budgets are not independent sample
counts. All q values use the complete unchanged R1 scorer and its original
calibration. Intervals condition on these trained seed0 models and do not
quantify training-seed uncertainty. Known minority witnesses are incomplete;
valid non-reference passages still count toward distinct valid passages.
''')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.root,a.output)
