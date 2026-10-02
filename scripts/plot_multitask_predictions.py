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
    if any(observations[item]['split']!='DEV_MODEL' for item in ids):raise ValueError('Only DEV predictions allowed')
    if len(set(ids))!=len(ids) or set(ids)!={key for key,row in observations.items() if row['split']=='DEV_MODEL'}:
        raise ValueError('All DEV predictions must be retained')
    selected=[]
    for task in sorted({supervision[item]['task'] for item in ids}):
        eligible=[i for i,item in enumerate(ids) if supervision[item]['task']==task]
        selected.append(min(eligible,key=lambda i:(parents[i],ids[i])))
    if len(selected)!=6:raise ValueError('Six task panels required')
    args.output.mkdir(parents=True)
    colors=['#0072B2','#D55E00','#009E73','#CC79A7']
    sources={str(prediction):sha(prediction),str(args.run/'summary.json'):sha(args.run/'summary.json')}
    panels=[];figures=[]
    for page in range(2):
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
            left.set_xlim(-.5,rgb.shape[1]-.5);left.set_ylim(rgb.shape[0]-.5,-.5);left.axis('off')
            left.set_title(label['task']+' / '+identifier+'\n'+textwrap.fill(obs['instruction'],65),fontsize=10)
            right.set_xlabel('world x (m)');right.set_ylabel('world y (m)');right.set_zlabel('world z (m)')
            right.tick_params(labelsize=8);right.view_init(elev=25,azim=-65)
            right.set_title('All K4 paths; grey = '+str(len(refs))+' recorded references',fontsize=10)
            if row==2:right.legend(loc='upper center',bbox_to_anchor=(.5,-.07),ncol=3,fontsize=8)
            panels.append(dict(task=label['task'],id=identifier,parent_id=parents[idx],reference_count=len(refs)))
        fig.suptitle('Six-task ordinary frozen-Qwen + RGB-D baseline — best step '+str(summary['best_step'])+
            '\nFixed first DEV parent/language per task; no collision or execution certification',fontsize=13)
        for suffix in ('png','pdf'):
            output=args.output/('tasks_'+str(page+1)+'.'+suffix)
            fig.savefig(output,dpi=150,bbox_inches='tight');figures.append(dict(path=str(output),sha256=sha(output)))
        plt.close(fig)
    index=dict(selection='first sorted DEV parent, then first sorted language, independently in each of six tasks',
        best_step=summary['best_step'],checkpoint_sha256=summary['best_checkpoint_sha256'],panels=panels,
        source_sha256=sources,plot_script_sha256=sha(__file__),figures=figures,current_mechanical_gate=gate,
        limitation='Visualization of saved task-level predictions and evaluation references; no new candidates, scoring, repair or robot execution')
    (args.output/'index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(json.dumps(dict(panels=len(panels),figures=figures)))


if __name__=='__main__':main()
