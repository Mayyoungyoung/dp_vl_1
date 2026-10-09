"""Frozen fresh-family diagnostic figure; every seed and stressor is retained."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(source,output):
    tic=time.monotonic()
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    d=json.loads(source.read_text(encoding='utf-8'));variants=['open','shifted','closed','narrow','tall','noise','occluded']
    kinds=['xyz','bounded','refitted_projection','refitted_center','refitted_boundcenter']
    labels=['Free XYZ','Bounded','Projection','XYZ center','Bounded center']
    fig,axes=plt.subplots(1,2,figsize=(13,4.7))
    for k,label in zip(kinds,labels):
        values=np.array([[r['variants'][v]['U8'] for v in variants] for r in d['seed_results'][k]])
        axes[0].plot(range(7),values.mean(0),'o-',label=label,lw=1.4)
        for seed in range(3):axes[0].scatter(range(7),values[seed],s=8,alpha=.5)
    axes[0].set_xticks(range(7),['Open','Shifted','Closed','Narrow','Tall','Noise','Masked'],rotation=22)
    axes[0].set_ylabel('Distinct actual valid modes @8')
    axes[0].set_title('Five rendered geometry states + two synthetic observation corruptions',fontsize=10)
    axes[0].legend(fontsize=8)
    for i,k in enumerate(('xyz','refitted_projection','refitted_center','refitted_boundcenter')):
        c=d['comparisons']['bounded minus '+k];low,high=c['crossed_CI95'][0]
        axes[1].plot([low,high],[i,i],color='#235c98',lw=2);axes[1].plot(c['mean'][0],i,'o',color='#235c98')
    axes[1].axvline(0,color='.4',lw=1)
    axes[1].set_yticks(range(4),['Free XYZ','Projection','XYZ center','Bounded center']);axes[1].invert_yaxis()
    axes[1].set_xlabel('Bounded minus control: valid modes @8')
    axes[1].set_title('Conditional seed × fresh-family bootstrap 95% intervals',fontsize=10)
    for ax in axes:ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
    fig.suptitle('16 new families / 336 requests; frozen checkpoints; no tuning',fontsize=11)
    fig.tight_layout();fig.savefig(output/'fresh_family_diagnosis.png',dpi=180);fig.savefig(output/'fresh_family_diagnosis.pdf');plt.close(fig)
    (output/'MANIFEST.json').write_text(json.dumps(dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope=d['scope'],model_selection_use=False,
            command=[sys.executable]+sys.argv,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime=dict(python=sys.version,numpy=np.__version__,matplotlib=matplotlib.__version__),elapsed_seconds=time.monotonic()-tic),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);main(**vars(p.parse_args()))
