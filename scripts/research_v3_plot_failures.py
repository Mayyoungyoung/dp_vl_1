"""First qualifying DEV examples, all candidates shown, no model changes."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np


def main(snapshot,closure,inputs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from PIL import Image
    from scripts.observed_layout_variation import geometry
    snapshot,closure,inputs,output=map(Path,(snapshot,closure,inputs,output));output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((inputs/'metadata.json').read_text(encoding='utf-8-sig'))
    rowfile=snapshot/'safety_mean/evaluation_fixed_q_v2/rows.json';rows=json.loads(rowfile.read_text())
    selection=[next(r for r in rows if any(c['semantic_goal_correct'] and not c['tip_segments_clear'] for c in r['candidates'])),
               next(r for r in rows if all(not c['semantic_goal_correct'] for c in r['candidates']))]
    poolfile=closure/'matched_q_paired_v1/evaluation/paired_dev/mean_seed0/pool.npz'
    qfile=closure/'matched_q_paired_v1/reliability/mean_seed0/calibration_seed0/predictions.npz'
    with np.load(poolfile) as z:pool={k:z[k] for k in ('paths','ids','labels')}
    with np.load(qfile) as z:q=z['q']
    files=[rowfile,poolfile,qfile,inputs/'metadata.json'];records=[]
    fig=plt.figure(figsize=(15,8.8),layout='constrained')
    grid=fig.add_gridspec(2,3,width_ratios=[1,1.45,.7])
    for i,r in enumerate(selection):
        m=meta[i];ident=r['id'];assert m['observation']['id']==ident
        image=inputs/(ident+'.png');cfgfile=inputs/(ident+'.json');files.extend([image,cfgfile])
        for key,p in [('image',image),('config',cfgfile)]:assert hashlib.sha256(p.read_bytes()).hexdigest()==m['sha256'][key]
        cfg=json.loads(cfgfile.read_text());j=list(pool['ids']).index(ident);paths=pool['paths'][j]
        valid=pool['labels'][j];np.testing.assert_array_equal(valid,r['valid'])
        colors=['#23867C' if v else '#C54747' for v in valid]
        ax=fig.add_subplot(grid[i,0]);ax.imshow(Image.open(image));ax.axis('off')
        target='cyan' if i==0 else 'purple'
        ax.set_title(f'{chr(65+i)}. Observed RGB: touch the {target} sphere',fontsize=11)
        ax.text(.5,-.08,ident,transform=ax.transAxes,ha='center',fontsize=8)
        ax=fig.add_subplot(grid[i,1],projection='3d')
        centers,halves=geometry(cfg)
        for center,half in zip(centers,halves):
            vertices=np.array(list(itertools.product((-1,1),repeat=3)))*half+center
            faces=[[vertices[k] for k in ids] for ids in ([0,1,3,2],[4,5,7,6],[0,1,5,4],[2,3,7,6],[0,2,6,4],[1,3,7,5])]
            ax.add_collection3d(Poly3DCollection(faces,facecolor='#89949F',edgecolor='#63717E',alpha=.22,lw=.5))
        for p,col in zip(paths,colors):ax.plot(*p.T,color=col,lw=1.4,alpha=.9);ax.scatter(*p[-1],color=col,s=14)
        goal=np.array(cfg['goal_xyz'][int(ident.rsplit('target',1)[1])]);start=paths[0,0]
        ax.scatter(*goal,marker='*',s=110,color='#222222',label='Requested goal (label)')
        ax.scatter(*start,marker='s',s=25,color='#222222')
        ax.set(xlim=(-.03,.53),ylim=(-.27,.27),zlim=(.75,1.05),xlabel='x (m)',ylabel='y (m)',zlabel='z (m)')
        ax.set_box_aspect((1.5,1.4,.8));ax.view_init(elev=28,azim=-55)
        ax.set_title('All eight generated paths; oracle boxes for evaluation only',fontsize=10)
        ax.tick_params(labelsize=7);ax.legend(loc='upper right',fontsize=7)
        ax=fig.add_subplot(grid[i,2]);ax.barh(np.arange(8),q[j],color=colors)
        ax.set(yticks=np.arange(8),yticklabels=[str(k) for k in range(8)],xlim=(0,1),xlabel='Independent q',ylabel='Candidate')
        ax.invert_yaxis();ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        ax.set_title(f'{int(valid.sum())}/8 valid\nGreen: valid; red: invalid',fontsize=10)
        records.append(dict(id=ident,selection='first request in stored order with goal-correct collision' if i==0 else 'first request in stored order with all semantic endpoints wrong',
            instruction=m['observation']['instruction'],valid=valid.tolist(),q=q[j].tolist()))
    fig.suptitle('Residual failures of the strong ordinary baseline\nFixed safety_mean generator + preselected paired-domain scorer seed0; DEV_MODEL only',fontsize=13)
    for ext in ('png','svg'):fig.savefig(output/('failure_examples.'+ext),dpi=180)
    plt.close(fig)
    source=dict(examples=records,input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='Mechanically selected first qualifying examples, not prevalence estimates. All M8 paths and calibrated q shown; labelled boxes/goals are evaluation overlays, never inference inputs; no robot execution.')
    (output/'SOURCES.json').write_text(json.dumps(source,indent=2)+'\n')
    (output/'CAPTION.md').write_text('Two residual failure examples, selected as the first qualifying stored DEV request for each stated category. All eight candidates are retained. RGB is the actual rendered observation; labelled geometry is shown only for evaluating the output. Box surfaces omit the20mm axis-expanded clearance boundary, so an invalid path may appear outside a solid. q comes from the preselected new-domain seed0 scorer, not an oracle. These illustrations do not estimate failure prevalence or demonstrate execution.\n')


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('snapshot','closure','inputs','output'):p.add_argument('--'+k,required=True)
    a=p.parse_args();main(a.snapshot,a.closure,a.inputs,a.output)
