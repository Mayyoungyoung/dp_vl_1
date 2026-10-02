"""Create figures and paired scene-bootstrap differences from saved outputs."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from routeset.data import load_dataset
from routeset.common import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/routes.npz')
    p.add_argument('--runs',default='runs/main')
    p.add_argument('--output',default='reports')
    a=p.parse_args()
    data=load_dataset(a.data)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    kinds=['independent','set_diffusion','regressor']
    names=['Balanced independent diffusion','Joint set diffusion','Matched route-set regression']
    info={}
    preds={}
    for kind in kinds:
        base=Path(a.runs)/(kind+'_seed0')
        info[kind]=json.loads((base/'evaluation/metrics.json').read_text(encoding='utf-8'))
        with np.load(base/'evaluation/test_predictions.npz') as f: preds[kind]={k:f[k] for k in f.files}
    write_json(out/'summary.json',info)
    rng=np.random.default_rng(45)
    compare={}
    for kind in kinds[1:]:
        assert np.array_equal(preds[kind]['scene_ids'],preds[kinds[0]]['scene_ids'])
        compare[kind]={}
        for metric in ['unique_valid','valid_rate','mode_exclusion_survival']:
            diff=preds[kind]['per_scene_'+metric]-preds[kinds[0]]['per_scene_'+metric]
            boot=rng.integers(0,len(diff),size=(2000,len(diff)))
            compare[kind][metric]={'difference':float(diff.mean()),'paired_scene_ci95':np.percentile(diff[boot].mean(1),[2.5,97.5]).tolist()}
    write_json(out/'paired_comparisons.json',compare)
    fig,axes=plt.subplots(1,3,figsize=(12,3.8))
    for ax,metric,label in zip(axes,['unique_valid','valid_rate','mode_exclusion_survival'],['Valid distinct routes @3','Candidate validity','After a mode is excluded']):
        values=[info[k]['test']['metrics'][metric] for k in kinds]
        intervals=[info[k]['test']['scene_bootstrap_ci95'][metric] for k in kinds]
        ax.bar(range(3),values,color=['#8593a5','#5186bf','#25a18e'])
        ax.errorbar(range(3),values,yerr=np.array([[v-c[0] for v,c in zip(values,intervals)],[c[1]-v for v,c in zip(values,intervals)]]),fmt='none',color='black',capsize=4)
        ax.set_xticks(range(3),['Independent','Set diffusion','Set regression'],rotation=15)
        ax.set_title(label)
        ax.set_ylim(0,3.25 if metric=='unique_valid' else 1.08)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Seed 0: held-out toy scenes; 3 raw candidates, identical demonstrations')
    fig.tight_layout()
    fig.savefig(out/'comparison.png',dpi=180)
    plt.close(fig)
    if (out/'multiseed_summary.json').exists():
        multi=json.loads((out/'multiseed_summary.json').read_text(encoding='utf-8'))
        fig,axes=plt.subplots(1,3,figsize=(12,3.8))
        for ax,metric,label in zip(axes,['unique_valid','valid_rate','mode_exclusion_survival'],['Valid distinct routes @3','Candidate validity','After a mode is excluded']):
            records=[multi['summary'][kind]['test'][metric] for kind in kinds]
            values=[x['mean'] for x in records]
            std=[x['std_across_seeds'] or 0 for x in records]
            ax.bar(range(3),values,yerr=std,capsize=4,color=['#8593a5','#5186bf','#25a18e'])
            ax.set_xticks(range(3),['Independent','Set diffusion','Set regression'],rotation=15)
            ax.set_title(label)
            ax.set_ylim(0,3.25 if metric=='unique_valid' else 1.08)
            ax.grid(axis='y',alpha=.2)
        fig.suptitle('Mean ± SD across 3 training seeds; 3 raw candidates per task')
        fig.tight_layout()
        fig.savefig(out/'comparison_multiseed.png',dpi=180)
        plt.close(fig)
    # The first test scene is predetermined, never chosen for favorable scores.
    index=int(preds[kinds[0]]['indices'][0])
    scene=data['scenes'][index]
    center,halfsize=scene[6:9],scene[9:12]
    corners=np.array([center+halfsize*np.array([x,y,z]) for x in [-1,1] for y in [-1,1] for z in [-1,1]])
    faces=[(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]
    fig=plt.figure(figsize=(13,4.5))
    for i,(kind,name) in enumerate(zip(kinds,names)):
        ax=fig.add_subplot(1,3,i+1,projection='3d')
        ax.add_collection3d(Poly3DCollection([corners[list(f)] for f in faces],alpha=.22,facecolor='#607482',edgecolor='#465760'))
        routes=preds[kind]['paths'][0,0]
        valid=preds[kind]['valid'][0,0]
        for k,(path,v) in enumerate(zip(routes,valid)):
            ax.plot(*path.T,color=['#2b73b6','#df942c','#a95291'][k],ls='-' if v else '--',lw=2,label='candidate '+str(k+1)+(' valid' if v else ' invalid'))
        ax.scatter(*scene[:3],c='red',s=30)
        ax.scatter(*scene[3:6],c='green',s=30)
        ax.set_xlim(-1,1); ax.set_ylim(-1,1); ax.set_zlim(0,1.4)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
        ax.view_init(24,-55)
        ax.set_title(name,fontsize=10)
        ax.legend(fontsize=7,loc='upper left')
    fig.suptitle('Predetermined first held-out scene; no repair or extra sampling')
    fig.tight_layout()
    fig.savefig(out/'routes_first_test_scene.png',dpi=180)
    plt.close(fig)


if __name__=='__main__': main()
