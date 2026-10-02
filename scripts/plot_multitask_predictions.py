"""Plot saved predictions for the first DEV parent/language in every task.

No model forward or outcome-based selection. Grey raw references are displayed
only after prediction. Figures do not certify collision or task success.
"""
import argparse
import hashlib
import json
from pathlib import Path
import textwrap

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def project(xyz, intrinsics, extrinsics):
    camera=(np.asarray(xyz)-extrinsics[:3,3])@extrinsics[:3,:3]
    pix=camera@intrinsics.T
    valid=camera[:,2]>0
    result=np.full((len(camera),2),np.nan)
    result[valid]=pix[valid,:2]/pix[valid,2:]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--all-dev-parents',action='store_true',help='First sorted language from every DEV parent')
    parser.add_argument('--comparison-run',type=Path,help='Saved paired model used only to share 3D axis limits')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve existing figures')
    config=json.loads((args.run/'config.json').read_text())
    summary=json.loads((args.run/'summary.json').read_text())
    if config.get('endpoint_mode')!='free_offset':raise ValueError('Expected six-task ordinary free endpoint run')
    from routeset.observed_multitask import check_multitask_model_gate
    gate=check_multitask_model_gate(args.snapshot/'observations.jsonl',args.snapshot/'supervision.jsonl')
    if gate is None:raise ValueError('Sealed multitask snapshot required')
    prediction=args.run/'dev_model/predictions.npz'
    if sha(prediction)!=summary['prediction_sha256']:raise ValueError('Saved prediction changed')
    observations={row['id']:row for row in map(json.loads,(args.snapshot/'observations.jsonl').read_text().splitlines())}
    supervision={row['id']:row for row in map(json.loads,(args.snapshot/'supervision.jsonl').read_text().splitlines())}
    with np.load(prediction,allow_pickle=False) as archive:
        ids=archive['scene_ids'].astype(str);paths=archive['paths'].copy();parents=archive['parent_ids'].astype(str)
    if paths.shape!=(len(ids),config['candidates'],config['horizon'],3) or not np.isfinite(paths).all():
        raise ValueError('Exact finite saved candidate budget required')
    comparison=None;comparison_sources={}
    if args.comparison_run:
        other_config=json.loads((args.comparison_run/'config.json').read_text())
        other_summary=json.loads((args.comparison_run/'summary.json').read_text())
        if any(config[k]!=other_config[k] for k in ('dataset_fingerprint','horizon','candidates','seed')):
            raise ValueError('Comparison must share original data, K, horizon and seed')
        other_path=args.comparison_run/'dev_model/predictions.npz'
        if sha(other_path)!=other_summary['prediction_sha256']:raise ValueError('Paired prediction changed')
        with np.load(other_path,allow_pickle=False) as archive:
            if not np.array_equal(archive['scene_ids'],ids) or archive['paths'].shape!=paths.shape:
                raise ValueError('Exact paired instruction order/shape required')
            comparison=archive['paths'].copy()
        if not np.isfinite(comparison).all():raise ValueError('Nonfinite paired paths cannot be hidden')
        comparison_sources={str(other_path):sha(other_path),str(args.comparison_run/'summary.json'):sha(args.comparison_run/'summary.json')}
    if any(observations[item]['split']!='DEV_MODEL' for item in ids):raise ValueError('Only DEV predictions allowed')
    if len(set(ids))!=len(ids) or set(ids)!={key for key,row in observations.items() if row['split']=='DEV_MODEL'}:
        raise ValueError('All DEV predictions must be retained')
    selected=[]
    for task in sorted({supervision[item]['task'] for item in ids}):
        eligible=[i for i,item in enumerate(ids) if supervision[item]['task']==task]
        selected.append(min(eligible,key=lambda i:(parents[i],ids[i])))
    if len(selected)!=6:raise ValueError('Six task panels required')
    if args.all_dev_parents:
        selected=[min(np.flatnonzero(parents==parent),key=lambda i:ids[i]) for parent in sorted(set(parents))]
        if len(selected)!=12:raise ValueError('Expected unchanged twelve-parent development set')
    args.output.mkdir(parents=True)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    sources={str(prediction):sha(prediction),str(args.run/'summary.json'):sha(args.run/'summary.json')}
    sources.update(comparison_sources)
    panels=[];figures=[]
    for page in range((len(selected)+2)//3):
        fig=plt.figure(figsize=(13,13),layout='constrained')
        for row,idx in enumerate(selected[page*3:(page+1)*3]):
            identifier=ids[idx];obs=observations[identifier];label=supervision[identifier]
            if label['semantic_targets'] is not None:raise ValueError('No invented goal labels')
            image=args.snapshot/obs['image'];current_file=args.snapshot/label['observation']
            with Image.open(image) as loaded:rgb=np.asarray(loaded.convert('RGB'))
            with np.load(current_file,allow_pickle=False) as current:
                intrinsics=current['camera_intrinsics'];extrinsics=current['camera_extrinsics'];start=current['gripper_pose'][:3]
            refs=[]
            for value in label['routes']:
                filename=args.snapshot/value
                with np.load(filename,allow_pickle=False) as archive:refs.append(archive['gripper_pose'][:,:3].copy())
                sources[str(filename)]=sha(filename)
            sources[str(image)]=sha(image);sources[str(current_file)]=sha(current_file)
            left=fig.add_subplot(3,2,row*2+1);right=fig.add_subplot(3,2,row*2+2,projection='3d')
            left.imshow(rgb)
            for reference in refs:
                uv=project(reference,intrinsics,extrinsics)
                left.plot(*uv.T,c='#B0B0B0',lw=1,alpha=.85)
                right.plot(*reference.T,c='#B0B0B0',lw=1,alpha=.85)
            for number,path in enumerate(paths[idx]):
                uv=project(path,intrinsics,extrinsics)
                left.plot(*uv.T,c=colors[number],lw=1.4)
                left.scatter(*uv[-1],c=colors[number],marker='x',s=30)
                right.plot(*path.T,c=colors[number],lw=1.5,label='Candidate '+str(number+1))
                right.scatter(*path[-1],c=colors[number],marker='x',s=20)
            right.scatter(*start,c='black',s=30,label='Current start')
            if comparison is not None:
                bounds=np.concatenate([paths[idx].reshape(-1,3),comparison[idx].reshape(-1,3),start[None]]+refs)
                lo,hi=bounds.min(0),bounds.max(0);center=(lo+hi)/2;radius=max(float((hi-lo).max())*.55,.04)
                right.set_xlim(center[0]-radius,center[0]+radius)
                right.set_ylim(center[1]-radius,center[1]+radius)
                right.set_zlim(center[2]-radius,center[2]+radius);right.set_box_aspect((1,1,1))
            left.set_xlim(-.5,rgb.shape[1]-.5);left.set_ylim(rgb.shape[0]-.5,-.5);left.axis('off')
            left.set_title(label['task']+' / '+identifier+'\n'+textwrap.fill(obs['instruction'],65),fontsize=10)
            right.set_xlabel('world x (m)');right.set_ylabel('world y (m)')
            # Matplotlib 3D tight boxes can clip the rotated z label on export.
            right.text2D(.88,.94,'world z (m)',transform=right.transAxes,fontsize=9,ha='center')
            for axis in (right.xaxis,right.yaxis,right.zaxis):axis.set_major_locator(MaxNLocator(3))
            right.tick_params(labelsize=8);right.view_init(elev=25,azim=-65)
            right.set_title('All K4 paths; grey = '+str(len(refs))+' recorded references',fontsize=10)
            if row==2:right.legend(loc='upper center',bbox_to_anchor=(.5,-.07),ncol=3,fontsize=8)
            panels.append(dict(task=label['task'],id=identifier,parent_id=parents[idx],reference_count=len(refs)))
        variant='event-supported auxiliary' if config.get('grounding_target')=='event_supported' else 'ordinary'
        selection='All 12 DEV parents, first language' if args.all_dev_parents else 'First DEV parent/language per task'
        fig.suptitle('Six-task frozen-Qwen + RGB-D '+variant+' — best step '+str(summary['best_step'])+
            '\n'+selection+'; no collision or execution certification',fontsize=13)
        for suffix in ('png','pdf'):
            output=args.output/('tasks_'+str(page+1)+'.'+suffix)
            fig.savefig(output,dpi=150,bbox_inches='tight');figures.append(dict(path=str(output),sha256=sha(output)))
        plt.close(fig)
    index=dict(selection=('all twelve DEV parents, first sorted language each' if args.all_dev_parents else
                         'first sorted DEV parent, then first sorted language, independently in each of six tasks'),
        shared_axis_comparison_run=str(args.comparison_run) if args.comparison_run else None,
        best_step=summary['best_step'],checkpoint_sha256=summary['best_checkpoint_sha256'],panels=panels,
        source_sha256=sources,plot_script_sha256=sha(__file__),figures=figures,current_mechanical_gate=gate,
        limitation='Visualization of saved task-level predictions and evaluation references; no new candidates, scoring, repair or robot execution')
    (args.output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(panels=len(panels),figures=figures)))


if __name__=='__main__':main()
