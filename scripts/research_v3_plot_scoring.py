"""Diagnostic figures from closed snapshots; no fitting or model selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(snapshot,addendum,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    snapshot,addendum,output=map(Path,(snapshot,addendum,output));output.mkdir(parents=True,exist_ok=False)
    files=[];rates=[]
    for role in ('SCORE_TRAIN','DEV_SCORE','CALIBRATION','paired_dev'):
        p=addendum/'matched_q_v1/evaluation'/role/'mean_seed0/pool.npz';files.append(p)
        with np.load(p) as z:rates.append((float(z['labels'].mean()),float(z['q'].mean()),len(z['ids'])))
    oldfile=snapshot/'safety_mean/evaluation_fixed_q_v2/metrics.json';files.append(oldfile)
    old=json.loads(oldfile.read_text(encoding='utf-8'))['reliability']
    newfile=Path('research_v3/matched_q_v1/RESULTS.json');files.append(newfile)
    new=json.loads(newfile.read_text(encoding='utf-8'))['probability_metrics']
    sf=Path('research_v3/selection_audit_v1/RESULTS.json');files.append(sf)
    selection=json.loads(sf.read_text(encoding='utf-8'))['policies']
    fig,axes=plt.subplots(1,3,figsize=(14,4.7),layout='constrained')
    x=np.arange(4);width=.36
    axes[0].bar(x-width/2,[r[0] for r in rates],width,label='Actual validity',color='#376080')
    axes[0].bar(x+width/2,[r[1] for r in rates],width,label='Original q mean',color='#C48354')
    axes[0].set_xticks(x,['Score train\n192 requests','Score dev\n96 requests','Calibration\n96 requests','Paired dev\n288 requests'],fontsize=8)
    axes[0].set_ylim(0,1);axes[0].set_ylabel('Candidate fraction / mean q')
    axes[0].set_title('Same generator, different score domains');axes[0].legend(fontsize=8,loc='upper left')
    axes[1].plot([0,1],[0,1],color='#9B9B9B',ls='--',lw=1)
    for label,r,color in [('Original q',old,'#376080')]+[(f'Old-domain fit seed{s}',new[str(s)]['calibrated'],c) for s,c in enumerate(['#C48354','#6CA89E','#A879A5'])]:
        bins=[b for b in r['reliability_bins'] if b['n']]
        axes[1].plot([b['confidence'] for b in bins],[b['valid'] for b in bins],marker='o',ms=3,label=label,color=color)
    axes[1].set(xlim=(0,1),ylim=(0,1),xlabel='Predicted validity',ylabel='Observed validity',title='Reliability on the identical paired pool')
    axes[1].legend(fontsize=7,loc='lower right')
    a=selection['top_q_K4'];b=selection['public_diverse_K4']
    vals=[a['distinct'],b['distinct'],b['candidate_oracle_distinct']]
    axes[2].bar(np.arange(3),vals,color=['#C48354','#376080','#B4BEC6'])
    axes[2].set_xticks(np.arange(3),['Top-q','Existing\ndiversity','Candidate\noracle bound'],fontsize=9)
    axes[2].set(ylim=(0,4.25),ylabel='Distinct valid passages @4',title='Little remaining selection headroom')
    for i,v in enumerate(vals):axes[2].text(i,v+.06,f'{v:.3f}',ha='center',fontsize=10)
    for ax in axes:
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Ordinary scoring controls: distribution shift and candidate-pool limits\nFixed safety_mean generator seed0; development evidence only',fontsize=12)
    fig.savefig(output/'scoring_diagnosis.png',dpi=180);fig.savefig(output/'scoring_diagnosis.svg');plt.close(fig)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (output/'SOURCES.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    (output/'CAPTION.md').write_text('''# Scoring diagnosis

All panels use the same fixed safety_mean seed0 generator. Left: original
reserved scoring domains and paired DEV have different actual validity rates;
this is not proof of pure label shift. Middle: equal-width reliability bins
on exactly the same288 pairedDEV requests, original q and three ordinary
matched scorers fitted only in the old scoring domain. Bins have unequal counts;
lines are descriptive, not uncertainty bands. Right: public fixed q selector
compared with plain top-q and a non-deployable candidate-pool oracle bound.
Oracle uses actual validity/mode labels only for this upper bound. No new
candidates, threshold fitting, training-seed averaging or final-test claims.
''',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--snapshot',required=True);p.add_argument('--addendum',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    main(a.snapshot,a.addendum,a.output)
