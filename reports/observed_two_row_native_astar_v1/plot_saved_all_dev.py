"""All12 sealed DEV parents: original RGB and both arms, no forward/search."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image,ImageOps,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
BASE=Path(__file__).parent
BINARY=ROOT/'runs/observed_two_row_native_astar_v1'
COLORS=['#1565c0','#e68a00','#218c51','#a943b7']


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())


def main():
    index=read(BASE/'SYNC_SHA256_INDEX.json');lookup={r['relative']:r for r in index['entries']}
    report=read(BASE/'dev/report.json');output=BASE/'figures';output.mkdir(exist_ok=False)
    rows={arm:{r['id']:r for r in report['per_request'][arm]} for arm in ('edge','spatial')}
    parents=['two_row_reach_%d'%p for p in range(283264,283276)]
    manifest=[];nodecaps=[]
    for parent in parents:
        rgb=BASE/'visualization_inputs'/parent/'front.png'
        config_file=BASE/'visualization_inputs'/parent/(parent+'.json')
        geometry_file=BINARY/'visualization_inputs'/parent/'verification_only.npz'
        for path,relative in [(rgb,'visualization_inputs/'+parent+'/front.png'),
            (config_file,'visualization_inputs/'+parent+'/'+parent+'.json'),
            (geometry_file,'visualization_inputs/'+parent+'/verification_only.npz')]:
            assert sha(path)==lookup[relative]['sha256']
        cfg=read(config_file);goals=np.asarray(cfg['goal_xyz'])
        with np.load(geometry_file,allow_pickle=False) as archive:
            centers=archive['obstacle_centers'];halves=archive['obstacle_halfsizes']
        fig=plt.figure(figsize=(17,12));grid=fig.add_gridspec(4,4,height_ratios=[.85,1,1,1],hspace=.43,wspace=.25)
        ax=fig.add_subplot(grid[0,0]);ax.imshow(Image.open(rgb));ax.axis('off');ax.set_title('Original RGB (posthoc display)',fontsize=9)
        info=fig.add_subplot(grid[0,1:]);info.axis('off')
        info.text(0,.96,parent+' | all3 instructions ×2 arms ×4 slots',fontsize=15,va='top')
        info.text(0,.72,'Solid: returned H24; dotted: complete raw; dashed: finite but invalid.\n'
            'Gray: physical posts; dashed margin: evaluation-only 2cm inflation. Black star: requested target.\n'
            'NaN slots are written explicitly, never omitted from the budget. No arm/IK/execution claim.\n'
            'Both arms: same native kernel, K4, 20k nodes /2s. Frozen source b184060.',fontsize=10,va='top')
        parent_rows=[]
        data={};points=[goals]
        for target in range(3):
            identifier=parent+'_target%d'%target
            for arm in ('edge','spatial'):
                row=rows[arm][identifier];folder=BINARY/'dev'/arm/identifier
                assert sha(folder/'predictions.npz')==row['prediction_sha256']
                assert sha(folder/'raw_paths.npz')==row['raw_paths_sha256']
                with np.load(folder/'predictions.npz',allow_pickle=False) as archive:paths=archive['paths'][0].copy()
                with np.load(folder/'raw_paths.npz',allow_pickle=False) as archive:raw=[archive['candidate_%d'%k].copy() for k in range(4)]
                data[(target,arm)]=(row,paths,raw)
                points += [x for x in raw if len(x)]
                for attempt in row['generation']['attempts']:
                    if attempt['status']=='expanded_node_budget_exhausted':
                        nodecaps.append(dict(id=identifier,arm=arm,**attempt))
        allpoints=np.vstack(points);lower=allpoints.min(0)-.045;upper=allpoints.max(0)+.045
        lower=np.minimum(lower,(centers-halves-.02).min(0));upper=np.maximum(upper,(centers+halves+.02).max(0))
        for target in range(3):
            for col,(arm,axes) in enumerate([('edge',(0,1)),('spatial',(0,1)),('edge',(0,2)),('spatial',(0,2))]):
                row,paths,raw=data[(target,arm)];a,b=axes;ax=fig.add_subplot(grid[target+1,col])
                for center,half in zip(centers,halves):
                    ax.add_patch(Rectangle((center[a]-half[a],center[b]-half[b]),2*half[a],2*half[b],facecolor='#dedede',edgecolor='#888',lw=.7))
                    ax.add_patch(Rectangle((center[a]-half[a]-.02,center[b]-half[b]-.02),2*(half[a]+.02),2*(half[b]+.02),fill=False,edgecolor='#a66',lw=.65,ls='--'))
                ax.scatter(goals[:,a],goals[:,b],marker='*',s=50,color='#aaa',zorder=4)
                ax.scatter(goals[target,a],goals[target,b],marker='*',s=100,color='black',zorder=7)
                states=[]
                for k,(candidate,attempt) in enumerate(zip(row['candidates'],row['generation']['attempts'])):
                    status=attempt['status'];mode=candidate['declared_passage_type']
                    label=('unknown' if mode is None else '/'.join(x.replace('_y','') for x in mode)) if candidate['TipValid'] else status.replace('expanded_node_budget_exhausted','NODE CAP').replace('no_admissible_virtual_goal_attachment','NO ATTACH')
                    if candidate['finite_xyz'] and not candidate['TipValid']:label='WRONG GOAL' if not candidate['semantic_goal_correct'] else 'INVALID'
                    states.append('%d: %s'%(k,label))
                    if len(raw[k]):ax.plot(raw[k][:,a],raw[k][:,b],color=COLORS[k],lw=.8,ls=':',alpha=.6)
                    if np.isfinite(paths[k]).all():
                        ax.plot(paths[k,:,a],paths[k,:,b],color=COLORS[k],lw=1.5,ls='-' if candidate['TipValid'] else '--',alpha=.9)
                        ax.scatter(paths[k,0,a],paths[k,0,b],s=10,color='black',zorder=8)
                        ax.scatter(paths[k,-1,a],paths[k,-1,b],s=18,color=COLORS[k],zorder=6)
                metric=row['metrics'];ax.set_title('T%d %s %s | valid%d/4, known%d'%(target,arm,'XY' if b==1 else 'XZ',int(metric['TipValidAtK']*4),metric['UniqueClassifiedTipValidAtK']),fontsize=9)
                ax.text(.02,.02,'\n'.join(states),transform=ax.transAxes,fontsize=6.8,va='bottom',bbox=dict(facecolor='white',alpha=.82,edgecolor='none'))
                ax.set(xlim=(lower[a],upper[a]),ylim=(lower[b],upper[b]),xlabel='world x (m)',ylabel='world '+('y' if b==1 else 'z')+' (m)')
                ax.tick_params(labelsize=7);ax.set_aspect('equal',adjustable='box');ax.grid(alpha=.18)
                if col<2:parent_rows.append(dict(id=row['id'],arm=arm,valid=int(metric['TipValidAtK']*4),known_unique=metric['UniqueClassifiedTipValidAtK']))
        path=output/(parent+'.png');fig.savefig(path,dpi=150,bbox_inches='tight');plt.close(fig)
        manifest.append(dict(parent_id=parent,path=path.relative_to(BASE).as_posix(),sha256=sha(path),all_conditions=parent_rows,
            visualization_input_hashes={str(p.relative_to(ROOT)):sha(p) for p in (rgb,config_file,geometry_file)}))
    for group in range(3):
        sheet=Image.new('RGB',(1800,1340),'white');draw=ImageDraw.Draw(sheet)
        for j,item in enumerate(manifest[group*4:group*4+4]):
            picture=Image.open(BASE/item['path']).convert('RGB');picture.thumbnail((890,645))
            x=(j%2)*900+(900-picture.width)//2;y=(j//2)*670+20
            sheet.paste(picture,(x,y));draw.text((x,y-15),item['parent_id'],fill='black')
        sheet.save(output/('contact_sheet_%d.jpg'%group),quality=92)
    (BASE/'FIGURE_MANIFEST.json').write_text(json.dumps(dict(parents=manifest,node_cap_attempts=nodecaps,
        scope='All12 DEV parents/all36 instructions, both saved native arms. Labels opened only for posthoc plotting; no generated candidates.',
        source_report_sha256=sha(BASE/'dev/report.json'),plot_source_sha256=sha(__file__)),indent=2)+'\n')
    (BASE/'FIGURES.md').write_text('# All12 DEV parents, both native arms\n\n'+
        '\n'.join('- [%s](%s)'%(r['parent_id'],r['path']) for r in manifest)+'\n')
    print(json.dumps(dict(parents=len(manifest),conditions=36,arms=2,shown_reserved_slots=288,node_caps=len(nodecaps))))


if __name__=='__main__':main()
