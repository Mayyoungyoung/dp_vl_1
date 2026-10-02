"""Plot every saved DEV pool after independent analysis, with no new inference."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation',type=Path,required=True)
    parser.add_argument('--analysis',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeat',type=int,default=0)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve earlier figures')
    generated=read(args.generation/'summary.json');analyzed=read(args.analysis/'summary.json')
    if generated['status']!='completed' or analyzed['generation_summary_sha256']!=digest(args.generation/'summary.json'):
        raise ValueError('Completed, independently checked generation required')
    if not 0<=args.repeat<generated['repeats']:raise ValueError('Unknown separate sampling repeat')
    data=Path(generated['config']['observations']).parent
    if digest(data/'export_manifest.json')!=generated['export_manifest_sha256']:
        raise ValueError('Original export changed')
    labels={r['id']:r for r in map(json.loads,(data/'supervision.jsonl').read_text().splitlines()) if r['split']=='DEV_MODEL'}
    observations={r['id']:r for r in map(json.loads,(data/'observations.jsonl').read_text().splitlines()) if r['split']=='DEV_MODEL'}
    pools={};cases={};sources={str(args.generation/'summary.json'):digest(args.generation/'summary.json'),
        str(args.analysis/'summary.json'):digest(args.analysis/'summary.json')}
    for method in ('independent4','whole4'):
        path=args.generation/f'repeat{args.repeat}'/method/'predictions.npz'
        relative=str(path.relative_to(args.generation)).replace('\\','/')
        if digest(path)!=generated['artifacts'][relative]['sha256']:raise ValueError('Saved candidate pool changed')
        with np.load(path,allow_pickle=False) as archive:
            ids=archive['scene_ids'].astype(str).tolist();paths=archive['paths'].copy()
        if len(ids)!=24 or len(set(ids))!=24 or set(ids)!=set(labels) or set(ids)!=set(observations):
            raise ValueError('All 24 original DEV instructions required')
        pools[method]=dict(zip(ids,paths));sources[str(path)]=digest(path)
        report=args.analysis/f'repeat{args.repeat}'/method/'per_scene.json'
        cases[method]={row['scene_id']:row for row in read(report)}
        if set(cases[method])!=set(ids):raise ValueError('Analysis omitted an instruction')
        sources[str(report)]=digest(report)
    args.output.mkdir(parents=True)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7'];panels=[];figures=[]
    ids=sorted(observations)
    for page in range(8):
        figure,axes=plt.subplots(3,2,figsize=(10,12),layout='constrained',squeeze=False)
        for row,identifier in enumerate(ids[page*3:page*3+3]):
            label=labels[identifier];obs=observations[identifier]
            current_path=(data/label['observation']).resolve();geometry_path=(data/label['verification_only']).resolve()
            for path in (current_path,geometry_path):
                if path.parent.name!=obs['parent_id'] or digest(path)!=analyzed['evaluation_input_sha256'][str(path)]:
                    raise ValueError('Evaluation artifact changed or parent differs')
                sources[str(path)]=digest(path)
            with np.load(current_path,allow_pickle=False) as archive:current=archive['gripper_pose'][:3].copy()
            with np.load(geometry_path,allow_pickle=False) as archive:
                centers=archive['obstacle_centers'].copy();sizes=archive['obstacle_halfsizes'].copy()
            targets=np.asarray(label['semantic_targets']['centers']);target=int(label['semantic_targets']['target_index'])
            extents=[targets[:,:2],current[None,:2],centers[:,:2]-sizes[:,:2]-.02,centers[:,:2]+sizes[:,:2]+.02]
            for method in pools:
                finite=pools[method][identifier][np.isfinite(pools[method][identifier]).all(axis=(1,2))]
                if len(finite):extents.append(finite[...,:2].reshape(-1,2))
            extent=np.concatenate(extents);lo=extent.min(0);hi=extent.max(0)
            # Use a common square extent for both methods, preserving all outliers
            # while avoiding needle-shaped panels when one coordinate diverges.
            bounds_center=(lo+hi)/2
            radius=max(float(np.max(hi-lo))*.57,.08)
            for column,method in enumerate(('independent4','whole4')):
                ax=axes[row,column];prediction=pools[method][identifier];case=cases[method][identifier]
                for center,size in zip(centers,sizes):
                    ax.add_patch(Rectangle(center[:2]-size[:2],2*size[0],2*size[1],facecolor='.80',edgecolor='.35'))
                    ax.add_patch(Rectangle(center[:2]-size[:2]-.02,2*size[0]+.04,2*size[1]+.04,fill=False,ls=':',edgecolor='.35'))
                ax.scatter(*targets[:,:2].T,c='.65',marker='o',s=25)
                ax.scatter(*targets[target,:2],c='black',marker='*',s=90,label='Instructed target')
                ax.scatter(*current[:2],c='black',marker='s',s=30,label='Current start')
                finite_count=0
                for slot,path in enumerate(prediction):
                    if not np.isfinite(path).all():continue
                    finite_count+=1
                    ax.plot(*path[:,:2].T,color=colors[slot],lw=1.3,alpha=.9,label='Slot '+str(slot+1))
                    ax.scatter(*path[-1,:2],color=colors[slot],marker='x',s=30)
                ax.set_xlim(bounds_center[0]-radius,bounds_center[0]+radius);ax.set_ylim(bounds_center[1]-radius,bounds_center[1]+radius)
                ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18)
                ax.set_xlabel('world x (m)');ax.set_ylabel('world y (m)')
                ax.set_title(identifier+' / '+method+'\nfinite '+str(finite_count)+'/4; 3D TipValid '+str(round(case['TipValidAtK']*4))+'/4',fontsize=10)
                panels.append(dict(scene_id=identifier,method=method,finite_candidates=finite_count,TipValidAtK=case['TipValidAtK']))
        figure.suptitle('Saved Qwen SFT candidates: all 24 DEV instructions, page '+str(page+1)+'/8\nXY projection only; 3D tip checks exclude arm, IK and execution',fontsize=12)
        for suffix in ('png','pdf'):
            path=args.output/('all_dev_page_'+str(page+1)+'.'+suffix)
            figure.savefig(path,dpi=130,bbox_inches='tight');figures.append(dict(path=str(path),sha256=digest(path)))
        plt.close(figure)
    (args.output/'index.json').write_text(json.dumps(dict(source_sha256=sources,plot_script_sha256=digest(__file__),
        checkpoint_sha256=generated['checkpoint_sha256'],sampling_repeat=args.repeat,panels=panels,figures=figures,
        scope='All original DEV slots, no prediction repair or extra inference; NaN failures are labeled, same axes across methods include every finite prediction. XY obstacle footprint alone does not show 3D collision.'),indent=2)+'\n')
    print(json.dumps(dict(panels=len(panels),files=len(figures))))


if __name__=='__main__':main()
