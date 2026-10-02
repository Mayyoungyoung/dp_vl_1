"""Read-only audit of every v6 initialization and registered route slot."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import collect_two_row_canonical_repair as repair
import analyze_observed_two_row_pilot as previous
batch=repair.batch;pilot=repair.pilot


def analyze(data,source_v5,output):
    manifest=json.loads((data/'manifest.json').read_text());summary=json.loads((data/'summary.json').read_text())
    repair.validate_registration(manifest['config']);output.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((data/'artifact_hashes.json').read_text())
    if not all(batch.digest(data/name)==digest for name,digest in hashes.items()):raise ValueError('artifact hash mismatch')
    reg=json.loads((source_v5/'registration.json').read_text())
    if batch.digest(source_v5/'registration.json')!=manifest['config']['registration_sha256']:raise ValueError('source registration differs')
    selected=next(p for p in reg['parent_plan'] if p['config']['seed']==283102);c=selected['config']
    layouts=batch.read_rows(data/'layout_audits.jsonl');records=batch.read_rows(data/'attempts.jsonl')
    observations=batch.read_rows(data/'observations.jsonl');supervision=batch.read_rows(data/'supervision.jsonl')
    if len(layouts)!=4 or {r['parent_id'] for r in layouts}!={p['parent_id'] for p in reg['parent_plan']}:raise ValueError('incomplete layout denominator')
    if any(not k.startswith('route:') for k in summary['planning_api_entry_counts']):raise ValueError('initialization/restore attempted planning')
    if len(records)>27 or any(r['parent_id']!=selected['parent_id'] for r in records):raise ValueError('route budget/parent mismatch')
    if len({(r['input_id'],r['attempt']) for r in records})!=len(records):raise ValueError('duplicate slot')
    if len(observations)>3 or any(set(r)!=set(('id','parent_id','split','image','instruction')) for r in observations):raise ValueError('observation whitelist/join mismatch')
    if observations and (len({r['image'] for r in observations})!=1 or len({r['instruction'] for r in observations})!=len(observations)):raise ValueError('same-image distinct-language contract violated')
    if any(r['split']!='DEV_COLLECTION' for r in observations+supervision):raise ValueError('role changed')
    goals=np.asarray(c['goal_xyz']);paths={};details=[];per_target=[]
    verified=data/selected['parent_id']/'verification_only.npz'
    if verified.exists():
        with np.load(verified,allow_pickle=False) as a:goals=a['target_centers'].copy()
    for r in records:
        target=int(r['input_id'].rsplit('target',1)[1]);trace=r.get('trajectory') or r.get('failed_partial_trajectory')
        detail=dict(input_id=r['input_id'],attempt=r['attempt'],success=r['success'],error=r.get('error'),collision_label=r.get('collision_pair'))
        if trace:
            path=data/r['parent_id']/trace['file']
            if batch.digest(path)!=trace['sha256']:raise ValueError('trace SHA mismatch')
            with np.load(path,allow_pickle=False) as a:xyz=a['gripper_pose'][:,:3].copy()
            if len(xyz)>1:
                fields,_,valid=repair.route_acceptance(xyz,goals,target,c)
                detail.update(recomputed_tip_and_endpoint_fields=fields,actual_crossings=previous.row_crossings(xyz,c),
                    segment_diagnostics=previous.segment_diagnostics(xyz,r.get('planning_segments',[]),c))
                if r['success'] and not valid:raise ValueError('accepted route fails unchanged checks')
                if r['success'] and json.dumps(fields['actual_route_type'])!=json.dumps(r['actual_route_type']):raise ValueError('actual type mismatch')
            paths[(target,r['attempt'])]=xyz
        details.append(detail)
    for target in range(3):
        rows=[r for r in records if r['input_id'].endswith('target%d'%target)]
        types=[r['actual_route_type'] for r in rows if r['success']]
        row=pilot.summarize_reference_types(selected['parent_id']+'_target%d'%target,types,c)
        row.update(requested=9,attempted=len(rows),failed=sum(not r['success'] for r in rows),unattempted=9-len(rows))
        per_target.append(row)
    result=dict(source=str(data),analyzer_sha256=batch.digest(__file__),source_collector=manifest['sources_sha256'],
        original_summary=summary,artifact_hashes_verified=len(hashes),layout_audits=layouts,per_target=per_target,
        same_image_three_targets=len(observations)==3,initialization_ik_or_path_entries=0,
        strict_route_restores=sum(bool(r.get('strict_restore',{}).get('passed')) for r in records),
        failures=dict(Counter(r.get('error') for r in records if not r['success'])),
        accepted_unknown=sum(r['success'] and r['actual_route_type'] is None for r in records),
        accepted_over=sum(r['success'] and r['actual_route_type'] is not None and 'over' in r['actual_route_type'] for r in records),
        explicitly_counted_get_path=sum(sum(s['get_path_calls'] for s in r.get('planning_segments',[])) for r in records),
        old_dynamic_initial_state_pairing=False,all_solution_count=None)
    batch.write(output/'analysis.json',result);pilot.write_json(output/'all_attempt_diagnostics.json',details)
    if paths:plot(paths,records,c,output/'all_27_attempts.png')
    return result


def plot(paths,records,c,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    centers,halves=pilot.geometry(c);goals=np.asarray(c['goal_xyz']);all_xyz=np.concatenate(list(paths.values()))
    figure,axes=plt.subplots(3,2,figsize=(13,15));palette=plt.get_cmap('tab10')
    for target in range(3):
        for column,(a,b) in enumerate(((0,1),(0,2))):
            axis=axes[target,column]
            for center,half in zip(centers,halves):axis.add_patch(Rectangle((center[a]-half[a],center[b]-half[b]),2*half[a],2*half[b],color='gray',alpha=.35))
            for r in records:
                key=(target,r['attempt'])
                if not r['input_id'].endswith('target%d'%target) or key not in paths:continue
                xyz=paths[key];color=palette(r['attempt'])
                axis.plot(xyz[:,a],xyz[:,b],color=color,lw=1.3,alpha=.85,ls='-' if r['success'] else '--',label='%d %s'%(r['attempt'],'PASS' if r['success'] else 'FAIL'))
                axis.scatter(xyz[-1,a],xyz[-1,b],color=color,s=20,marker='x')
            axis.scatter(goals[target,a],goals[target,b],marker='*',s=100,c='black',label='goal label')
            axis.set_xlim(min(all_xyz[:,a].min(),centers[:,a].min())-.04,max(all_xyz[:,a].max(),goals[:,a].max())+.04)
            axis.set_ylim(min(all_xyz[:,b].min(),(centers-halves)[:,b].min())-.04,max(all_xyz[:,b].max(),(centers+halves)[:,b].max())+.04)
            axis.set_xlabel('xyz'[a]+' (m)');axis.set_ylabel('xyz'[b]+' (m)');axis.set_aspect('equal',adjustable='box');axis.grid(alpha=.2)
            axis.set_title('Target %d — %s projection, all 9 registered slots'%(target,'XY' if column==0 else 'XZ'))
            if column==0:axis.legend(fontsize=7,ncol=2)
    figure.suptitle('Canonical static-start development repair: solid=accepted, dashed=failed partial/full route\nGray boxes are projections; endpoint markers are not robot contact locations',fontsize=12)
    figure.tight_layout(rect=(0,0,1,.96));figure.savefig(path,dpi=150);plt.close(figure)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,required=True);parser.add_argument('--source-v5',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();result=analyze(a.data,a.source_v5,a.output)
    print(json.dumps({k:result[k] for k in ('per_target','accepted_unknown','accepted_over','strict_route_restores','explicitly_counted_get_path')}))


if __name__=='__main__':main()
