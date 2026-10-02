"""Offline TRAIN-only analysis of sealed pools/graphs and already-read references."""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from scripts import observation_multiroute_astar as planner

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
write=lambda p,x:Path(p).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def original_resampler():
    source=ROOT/'routeset/observed_route_head.py'
    tree=ast.parse(source.read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resample_event_segments')
    namespace={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),namespace)
    return namespace['resample_event_segments'],dict(path=str(source),actual_sha256=sha(source),
        function_ast_sha256=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest(),
        scope='Exact existing pure function extracted without importing Torch or any model')


def cell_reason(cell,free,lower,observed):
    if any(cell[i]<0 or cell[i]>=free.shape[i] for i in range(3)):return 'outside_workspace'
    if free[cell]:return None
    world=lower+np.asarray(cell)*planner.CONFIG['voxel_m']
    camera=(world-observed['camera_extrinsics'][:3,3])@observed['camera_extrinsics'][:3,:3]
    z=float(camera[2]);projection=camera@observed['camera_intrinsics'].T
    if z<=0:return 'unknown_not_visible'
    u,v=np.rint(projection[:2]/z).astype(int)
    depth=observed['depth']
    if not(0<=u<depth.shape[1] and 0<=v<depth.shape[0]):return 'unknown_not_visible'
    measured=float(depth[v,u])
    if not np.isfinite(measured) or not 0<measured<=planner.prototype.CONFIG['maximum_depth_m']:
        return 'unknown_not_visible'
    if z>measured:return 'unknown_behind_observed_depth'
    if z>=measured-planner.CONFIG['visible_surface_clearance_m']:return 'observed_surface_depth_band'
    return 'observed_depth_free_but_conservative_grid_blocked'


def route_grid_diagnostic(path,grid,observed):
    free,lower=grid['free'],grid['lower']
    arc=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(path,axis=0),axis=-1))]
    fraction=arc/arc[-1] if arc[-1] else np.zeros_like(arc)
    segments=[];all_cells=set();reason_counts=Counter();regions=Counter();contact_regions=Counter()
    for i,(first,second) in enumerate(zip(path,path[1:])):
        bad=[]
        for cell in sorted(planner.segment_supercover(first,second,lower,planner.CONFIG['voxel_m'])):
            reason=cell_reason(cell,free,lower,observed)
            if reason is None:continue
            bad.append((cell,reason));all_cells.add(cell);reason_counts[reason]+=1
            center=lower+np.asarray(cell)*planner.CONFIG['voxel_m']
            contact=('predicted_goal_contact_ball' if np.linalg.norm(center-grid['predicted_goal'])<=grid['target_radius']
                else ('current_start_contact_ball' if np.linalg.norm(center-observed['gripper_pose'][:3])<=planner.CONFIG['current_tip_contact_radius_m']
                else 'outside_both_original_contact_balls'))
            contact_regions[contact]+=1
            region='first_segment' if i==0 else ('last_segment' if i==len(path)-2 else 'interior_segment')
            regions[region]+=1
        if bad:
            segments.append(dict(segment_index=i,arc_fraction_start_lower_bound=float(fraction[i]),
                arc_fraction_end_upper_bound=float(fraction[i+1]),blocked_cell_count=len(bad),
                reasons=dict(Counter(reason for _,reason in bad)),first_eight_blocked_cells=[list(c) for c,_ in bad[:8]]))
    def point_blocked(point):
        return any(cell_reason(c,free,lower,observed) is not None
            for c in planner.segment_supercover(point,point,lower,planner.CONFIG['voxel_m']))
    clear,_=planner.path_observed_clear(path,free,lower)
    assert bool(segments)==(not clear)
    return dict(grid_proxy_clear=clear,start_point_blocked=point_blocked(path[0]),end_point_blocked=point_blocked(path[-1]),
        endpoint_blocked_cell_reasons=dict(Counter(reason for c in planner.segment_supercover(path[-1],path[-1],lower,planner.CONFIG['voxel_m'])
            for reason in [cell_reason(c,free,lower,observed)] if reason is not None)),
        first_segment_blocked=any(s['segment_index']==0 for s in segments),
        last_segment_blocked=any(s['segment_index']==len(path)-2 for s in segments),
        interior_segment_blocked=any(0<s['segment_index']<len(path)-2 for s in segments),
        prefix25_segment_blocked=any(s['arc_fraction_start_lower_bound']<.25 for s in segments),
        middle50_segment_blocked=any(s['arc_fraction_start_lower_bound']<.75 and s['arc_fraction_end_upper_bound']>.25 for s in segments),
        first_blocked_segment=segments[0] if segments else None,
        all_blocked_segments=segments,blocked_unique_cells=len(all_cells),
        blocked_cell_segment_occurrences_by_reason=dict(reason_counts),
        any_blocked_cell_center_outside_original_contact_balls=contact_regions['outside_both_original_contact_balls']>0,
        blocked_cell_segment_occurrences_by_contact_region=dict(contact_regions),
        blocked_cell_segment_occurrences_by_index_region=dict(regions),
        distinction='Arc bounds describe a segment intersecting blocked discrete voxels; not an exact physical collision contact')


def main():
    began=time.perf_counter();report=json.loads((HERE/'analysis/report.json').read_text())
    inputs=json.loads((HERE/'TRAIN_INPUTS_INDEX.json').read_text())
    source=json.loads((HERE/'analysis/input_source_seal.json').read_text())
    expected=source['source_files_sha256']['scripts/observation_multiroute_astar.py']
    # The module's code is unchanged; archived runtime bytes may use CRLF.
    actual_file=ROOT/'scripts/observation_multiroute_astar.py'
    actual=sha(actual_file)
    assert actual==expected or hashlib.sha256(actual_file.read_bytes().replace(b'\r\n',b'\n')).hexdigest()=='cf487a91b5825c50ace087f5ebd29c1a7ca2b91607ccc8ff98af0e6a21991c93'
    resample,resampler_receipt=original_resampler()
    def local(original):
        item=inputs['files'][original];file=HERE/'TRAIN_INPUTS_IGNORED'/item['archived_name']
        assert sha(file)==item['sha256'];return file
    condition_rows=[];reference_rows=[]
    for row in report['reference_graph_support']:
        identifier=row['id'];parent=identifier.rsplit('_target',1)[0]
        current_name=next(n for n,item in inputs['files'].items() if item['kind']=='current_observation' and '/'+parent+'/' in n)
        with np.load(local(current_name),allow_pickle=False) as a:observed={k:a[k].copy() for k in a.files}
        gridpath=HERE/'analysis'/(identifier+'_observed_graph.npz')
        assert sha(gridpath)==row['graph']['artifact_sha256']
        with np.load(gridpath,allow_pickle=False) as a:grid={k:a[k].copy() for k in a.files}
        grid['target_radius']=row['graph']['grid']['contact_allowances']['target_radius_m']
        per_ref=[]
        for ref in row['references']:
            with np.load(local(ref['source']),allow_pickle=False) as a:
                raw=a['gripper_pose'][:,:3];h24,_=resample(a['gripper_pose'],a['gripper_open'],24)
            rec=dict(id=identifier,reference_index=ref['reference_index'],known_type=ref['known_type'],source_sha256=ref['sha256'])
            for kind,path in [('raw',raw),('h24',h24)]:
                rec[kind]=route_grid_diagnostic(path,grid,observed)
                assert rec[kind]['grid_proxy_clear']==ref[kind]['original_grid_proxy_clear']
                rec[kind]['visible_proxy_passed']=ref[kind]['original_visible_proxy']['passed']
                rec[kind]['visible_point_segment_clear']=ref[kind]['original_visible_proxy']['finite_point_cloud_segment_clear']
                rec[kind]['ray_proxy_passed']=ref[kind]['original_visible_proxy']['ray_check']['passed']
                rec[kind]['reference_end_to_predicted_goal_m']=ref[kind]['reference_end_to_predicted_goal_m']
            per_ref.append(rec);reference_rows.append(rec)
        swap=next(r for r in report['records'] if r['id']==identifier and r['arm']=='cyclic_direct_swap')
        candidates=[r for r in report['collision_transitions'] if r['id']==identifier]
        condition_rows.append(dict(id=identifier,parent_id=parent,target_index=int(identifier[-1]),reference_count=len(per_ref),
            body_mean_m=float(np.mean([r['full_body_mean_m'] for r in swap['displacement']])),
            body_max_m=max(r['full_body_max_m'] for r in swap['displacement']),
            prefix_mean_m=float(np.mean([r['prefix_mean_m'] for r in swap['displacement']])),
            normal_clear=sum(r['arms']['normal']['tip_segments_clear'] for r in candidates),
            swap_clear=sum(r['arms']['cyclic_direct_swap']['tip_segments_clear'] for r in candidates),
            clear_to_collision=sum(r['clear_to_collision'] for r in candidates),collision_to_clear=sum(r['collision_to_clear'] for r in candidates),
            start_attachments=row['graph']['start_attachments']['accepted_connections'],goal_attachments=row['graph']['goal_attachments']['accepted_connections'],
            raw_visible_proxy_pass=sum(r['raw']['visible_proxy_passed'] for r in per_ref),h24_visible_proxy_pass=sum(r['h24']['visible_proxy_passed'] for r in per_ref),
            raw_start_point_blocked=sum(r['raw']['start_point_blocked'] for r in per_ref),raw_end_point_blocked=sum(r['raw']['end_point_blocked'] for r in per_ref),
            raw_first_segment_blocked=sum(r['raw']['first_segment_blocked'] for r in per_ref),raw_last_segment_blocked=sum(r['raw']['last_segment_blocked'] for r in per_ref),
            raw_interior_segment_blocked=sum(r['raw']['interior_segment_blocked'] for r in per_ref)))
    aggregate={}
    for kind in ('raw','h24'):
        values=[r[kind] for r in reference_rows]
        counts={key:sum(v[key] for v in values) for key in ('grid_proxy_clear','start_point_blocked','end_point_blocked',
            'first_segment_blocked','last_segment_blocked','interior_segment_blocked','prefix25_segment_blocked',
            'middle50_segment_blocked','visible_proxy_passed','visible_point_segment_clear','ray_proxy_passed',
            'any_blocked_cell_center_outside_original_contact_balls')}
        reasons=Counter();locations=Counter();contacts=Counter();endpoint_reasons=Counter()
        for v in values:
            reasons.update(v['blocked_cell_segment_occurrences_by_reason']);locations.update(v['blocked_cell_segment_occurrences_by_index_region'])
            contacts.update(v['blocked_cell_segment_occurrences_by_contact_region'])
            endpoint_reasons.update(v['endpoint_blocked_cell_reasons'])
        first=[v['first_blocked_segment']['arc_fraction_start_lower_bound'] for v in values if v['first_blocked_segment']]
        counts.update(blocked_cell_segment_occurrences_by_reason=dict(reasons),blocked_occurrences_by_index_region=dict(locations),
            blocked_occurrences_by_original_contact_region=dict(contacts),
            endpoint_blocked_cell_occurrences_by_reason=dict(endpoint_reasons),
            first_blocked_segment_start_arc_lower_bound=dict(min=min(first),median=float(np.median(first)),max=max(first)),
            ref_end_to_predicted_goal_m=dict(min=min(v['reference_end_to_predicted_goal_m'] for v in values),
                mean=float(np.mean([v['reference_end_to_predicted_goal_m'] for v in values])),max=max(v['reference_end_to_predicted_goal_m'] for v in values)))
        aggregate[kind]=counts
    swaps=[d for r in report['records'] if r['arm']=='cyclic_direct_swap' for d in r['displacement']]
    result=dict(protocol='saved_train12_direct_conditioning_graph_localization_v1',new_forward_requests=0,new_searches=0,
        parent_count=4,input_count=12,positive_reference_count=len(reference_rows),candidate_pairs=48,
        direct_swap=dict(normal_clear=sum(c['normal_clear'] for c in condition_rows),swap_clear=sum(c['swap_clear'] for c in condition_rows),
            clear_to_collision=sum(c['clear_to_collision'] for c in condition_rows),collision_to_clear=sum(c['collision_to_clear'] for c in condition_rows),
            full_body_mean_m=float(np.mean([d['full_body_mean_m'] for d in swaps])),full_body_max_m=max(d['full_body_max_m'] for d in swaps),
            prefix_mean_m=float(np.mean([d['prefix_mean_m'] for d in swaps])),prefix_median_m=float(np.median([d['prefix_mean_m'] for d in swaps]))),
        conditions=condition_rows,graph_support=aggregate,reference_records=reference_rows,
        offline_analysis_seconds=time.perf_counter()-began,original_report_sha256=sha(HERE/'analysis/report.json'),
        original_grid_module_sha256=actual,original_grid_runtime_sha256=expected,resampler=resampler_receipt,
        scope='Blocked reference voxels are not an impossibility proof for alternative paths; no thresholds or pool changed')
    write(HERE/'SAVED_DIAGNOSIS.json',result)
    figures=HERE/'figures';figures.mkdir(exist_ok=True)
    labels=[r['id'].replace('two_row_reach_','').replace('_target','/t') for r in condition_rows]
    x=np.arange(12);fig,axes=plt.subplots(3,1,figsize=(13,10),sharex=True)
    axes[0].bar(x-.18,[100*c['body_mean_m'] for c in condition_rows],.36,label='Full body mean')
    axes[0].bar(x+.18,[100*c['prefix_mean_m'] for c in condition_rows],.36,label='First 50% arc mean')
    axes[0].set_ylabel('Displacement (cm)');axes[0].legend();axes[0].set_title('Fixed endpoint direct-feature intervention: counterfactual TRAIN diagnostic')
    axes[1].bar(x-.18,[c['normal_clear'] for c in condition_rows],.36,label='Normal tip-clear')
    axes[1].bar(x+.18,[c['swap_clear'] for c in condition_rows],.36,label='Swapped tip-clear')
    axes[1].set_ylim(0,4.5);axes[1].set_ylabel('Clear candidates / 4');axes[1].legend()
    axes[2].bar(x-.25,[c['reference_count'] for c in condition_rows],.25,label='Known positive references')
    axes[2].bar(x,[c['raw_visible_proxy_pass'] for c in condition_rows],.25,label='Raw visible proxy passes')
    axes[2].bar(x+.25,[c['h24_visible_proxy_pass'] for c in condition_rows],.25,label='H24 visible proxy passes')
    axes[2].set_ylabel('References');axes[2].set_title('All 69 references fail full original grid proxy; this is NOT a no-path theorem')
    axes[2].legend();axes[2].set_xticks(x,labels,rotation=45,ha='right');fig.tight_layout();fig.savefig(figures/'condition_summary.png',dpi=170);plt.close(fig)
    for parent in sorted({r['parent_id'] for r in condition_rows}):
        fig,axes=plt.subplots(2,3,figsize=(13,7))
        for t in range(3):
            identifier=parent+'_target%d'%t
            with np.load(HERE/'analysis'/('normal_'+identifier+'.npz')) as a:normal=a['paths']
            with np.load(HERE/'analysis'/('cyclic_direct_swap_'+identifier+'.npz')) as a:changed=a['paths']
            for k in range(4):
                color='C%d'%k
                for row,(vertical,name) in enumerate(((1,'Y'),(2,'Z'))):
                    ax=axes[row,t];ax.plot(normal[k,:,0],normal[k,:,vertical],color=color,label='Normal %d'%k)
                    ax.plot(changed[k,:,0],changed[k,:,vertical],color=color,linestyle='--',label='Swap %d'%k)
                    ax.scatter(normal[k,-1,0],normal[k,-1,vertical],c=color,s=12)
                    ax.set_xlabel('X (m)');ax.set_ylabel(name+' (m)');ax.grid(alpha=.2);ax.set_aspect('equal',adjustable='datalim')
            axes[0,t].set_title('Target%d: donor target%d'%(t,(t+1)%3))
        axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle(parent+' | solid=normal; dashed=intervention; same endpoint; no obstacle overlay')
        fig.tight_layout();fig.savefig(figures/(parent+'.png'),dpi=160);plt.close(fig)
    from PIL import Image,ImageOps,ImageDraw
    originals=[figures/(p+'.png') for p in sorted({r['parent_id'] for r in condition_rows})]
    sheet=Image.new('RGB',(1400,820),'white')
    for i,file in enumerate(originals):
        im=Image.open(file).convert('RGB');im.thumbnail((700,410));sheet.paste(im,((i%2)*700,(i//2)*410))
    sheet.save(figures/'all4parents_contact_sheet.png')
    print(json.dumps(dict(direct=result['direct_swap'],reference_count=len(reference_rows),graph=aggregate),indent=2))


if __name__=='__main__':main()
