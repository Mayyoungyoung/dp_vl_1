"""Audit all four closed development layouts and render every registered slot."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import collect_two_row_layout4 as batch
import analyze_observed_two_row_pilot as single
from diagnose_two_row_endpoint_ik import rotation


def obb_overlap(first,second):
    def unpack(item):
        pose=np.asarray(item['pose']);bounds=np.asarray(item['bounding_box']).reshape(3,2);basis=rotation(pose[3:])
        return pose[:3]+basis@bounds.mean(1),basis,(bounds[:,1]-bounds[:,0])/2
    ac,ar,ah=unpack(first);bc,br,bh=unpack(second)
    axes=[ar[:,i] for i in range(3)]+[br[:,i] for i in range(3)]
    axes.extend(np.cross(ar[:,i],br[:,j]) for i in range(3) for j in range(3))
    return all(np.linalg.norm(axis)<1e-10 or abs((ac-bc)@axis)<=np.abs(ar.T@axis)@ah+np.abs(br.T@axis)@bh for axis in axes)


def failed_setup_evidence(source,planned,shape_provenance=None):
    parent=source/planned['parent_id'];setup=json.loads((parent/'setup.json').read_text())
    result=dict(error=setup.get('error'),collision_label=setup.get('collision_pair'),new_ik_queries=0,new_routes=0,
        implication='Executed preparation collision is not endpoint impossibility; one run does not isolate stochastic planning from physical layout.')
    path=parent/'failed_setup.json'
    if path.exists():
        world=json.loads(path.read_text())
        posts=[world['state']['_extra_derived_two_row_post_%d'%i] for i in range(4)]
        centers=[p['pose'][:3] for p in posts]
        halves=[np.diff(np.asarray(p['bounding_box']).reshape(3,2),axis=1).ravel()/2 for p in posts]
        goals=[world['state'][name]['pose'][:3] for name in ('target','distractor0','distractor1')]
        result.update(actual_geometry_1mm_sha256=batch.scene_geometry_hash(centers,halves,goals,.001),
            actual_geometry_exact_sha256=batch.scene_geometry_hash(centers,halves,goals),
            geometry_evidence_scope='Post/goal geometry at failed setup capture, not a valid route-start observation',world_capture_sha256=batch.digest(path))
        if shape_provenance:
            hits=[]
            for shape in json.loads(shape_provenance.read_text()):
                name=shape['name']
                if not name.startswith('Panda') or not shape['collidable']:continue
                robot=dict(pose=world['state']['_global_pose_'+name]['pose'],bounding_box=shape['bounding_box'])
                hits.extend(dict(robot_shape=name,post_index=i) for i,post in enumerate(posts) if obb_overlap(robot,post))
            result.update(obb_overlaps_diagnostic_only=hits,shape_provenance_sha256=batch.digest(shape_provenance),
                obb_scope='Saved bbox proxy from separate same-pinned-model static diagnostic; not a runtime mesh contact query or proven original collision body.')
    if setup.get('trajectory'):
        path=parent/setup['trajectory']['file']
        if batch.digest(path)!=setup['trajectory']['sha256']:raise ValueError('setup trace SHA mismatch')
        xyz=np.load(path,allow_pickle=False)['gripper_pose'][:,:3];config=planned['config'];centers,halves=batch.collector.geometry(config)
        failed=[i for i,(a,b) in enumerate(zip(xyz[:-1],xyz[1:])) if not batch.collector.legacy.tip_polyline_clear(np.stack([a,b]),centers,halves,config['tip_clearance_m'])]
        result.update(last_tip_xyz=xyz[-1].tolist(),requested_entry_xyz=config['entry_xyz'],
            distance_to_requested_entry_m=float(np.linalg.norm(xyz[-1]-config['entry_xyz'])),first_setup_tip_2cm_failure=failed[0] if failed else None,
            trajectory_sha256=batch.digest(path),planning_segments=setup.get('planning_segments',[]))
    return result


def analyze(data,output,shape_provenance=None):
    summary=json.loads((data/'summary.json').read_text())
    registration=json.loads((data/'registration.json').read_text())
    if summary['closed_parents']!=4 or summary['requested_routes']!=108 or summary['role']!='DEV_COLLECTION':
        raise ValueError('complete fixed four-parent development corpus required')
    output.mkdir(parents=True,exist_ok=False)
    all_observations=batch.read_rows(data/'observations.jsonl')
    all_supervision=batch.read_rows(data/'supervision.jsonl')
    parents=[];targets=[];failures=Counter()
    for planned in registration['parent_plan']:
        parent=planned['parent_id'];seed=parent.rsplit('_',1)[1];source=data/'raw'/seed
        closure=json.loads((data/'closures'/(seed+'.json')).read_text())
        checked=batch.parent_evidence(source,planned,closure.get('worker_seconds'),closure.get('worker_returncode'),.001)
        if closure!=checked:raise ValueError('closed parent evidence changed')
        if not (source/'summary.json').exists():
            parents.append(dict(parent_id=parent,status=closure['status'],closed_evidence=closure,analysis_available=False))
            continue
        analysis=single.analyze(source,output/parent)
        observations=[r for r in all_observations if r['parent_id']==parent]
        supervision=[r for r in all_supervision if r['parent_id']==parent]
        if analysis['setup_success']:
            if len(observations)!=3 or len(supervision)!=3:raise ValueError('three target observation/supervision rows required')
            if any(set(r)!=set(('id','parent_id','split','image','instruction')) for r in observations):
                raise ValueError('observation input whitelist changed')
            if any(r['split']!='DEV_COLLECTION' for r in observations+supervision):raise ValueError('unexpected role')
            hashes={r['id']:batch.digest(data/r['image']) for r in observations}
            same_image=len(set(hashes.values()))==1 and len({r['image'] for r in observations})==1
            if not same_image or len({r['instruction'] for r in observations})!=3:
                raise ValueError('same image / distinct target language audit failed')
            if sorted(r['semantic_targets']['target_index'] for r in supervision)!=[0,1,2]:
                raise ValueError('target label join incomplete')
            if len({r['observation'] for r in supervision})!=1:raise ValueError('targets use different state/depth observations')
            for row in supervision:
                if len(row['routes'])!=len(row['route_types']):raise ValueError('route/type join mismatch')
                for route in row['routes']:
                    if not (data/route).is_file():raise ValueError('reference route missing')
        else:
            same_image=None;hashes={}
            if observations or supervision:raise ValueError('setup failure must not create valid observations')
        setup_failure=None if analysis['setup_success'] else failed_setup_evidence(source,planned,shape_provenance)
        for row in analysis['collection_summary']['per_target']:
            target=dict(row,parent_id=parent)
            targets.append(target)
        failures.update(analysis['failure_counts'])
        parents.append(dict(parent_id=parent,status=closure['status'],setup_success=analysis['setup_success'],
            attempted=analysis['collection_summary']['route_attempts'],accepted=analysis['success_count'],
            unknown=analysis['accepted_unknown_types'],over=analysis['accepted_with_over_type'],
            strict_restore_passes=analysis['strict_restore_passes'],same_rgb_for_three_goals=same_image,
            input_image_hashes=hashes,
            actual_geometry_1mm_sha256=closure.get('actual_geometry_1mm_sha256') or (setup_failure or {}).get('actual_geometry_1mm_sha256'),
            actual_geometry_exact_sha256=closure.get('actual_geometry_exact_sha256') or (setup_failure or {}).get('actual_geometry_exact_sha256'),
            setup_failure_diagnostic=setup_failure,
            closed_evidence=closure,analysis_available=True))
    result=dict(source=str(data),source_registration_sha256=batch.digest(data/'registration.json'),
        analyzer_source_sha256=batch.digest(__file__),single_analyzer_sha256=batch.digest(single.__file__),
        actual_requested_parents=4,actual_requested_goal_instructions=12,actual_requested_routes=108,
        closed_parent_count=4,parents=parents,targets=targets,failure_counts=dict(failures),
        accepted_routes=sum(r.get('accepted',0) for r in parents),
        accepted_unknown_routes=sum(r.get('unknown',0) for r in parents),
        targets_with_at_least_five_known_valid_types=sum(r['distinct_classified']>=5 for r in targets),
        targets_with_at_least_five_lateral_types=sum(r['distinct_lateral_sequences']>=5 for r in targets),
        actual_geometry_duplicate_groups=batch.duplicate_groups(parents,'actual_geometry_1mm_sha256'),
        actual_geometry_measured_parent_count=sum(r.get('actual_geometry_1mm_sha256') is not None for r in parents),
        same_image_three_goal_parent_count=sum(r.get('same_rgb_for_three_goals') is True for r in parents),
        closed_collection_summary=summary,
        interpretation='Four fixed local layout perturbations, all DEV_COLLECTION. No method advantage, complete solution set, continuous full-robot certificate or broad generalization claim.')
    batch.write(output/'analysis.json',result)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--shape-provenance',type=Path)
    args=parser.parse_args();result=analyze(args.data,args.output,args.shape_provenance)
    print(json.dumps({k:result[k] for k in ('accepted_routes','accepted_unknown_routes','targets_with_at_least_five_known_valid_types',
        'targets_with_at_least_five_lateral_types','actual_geometry_duplicate_groups','same_image_three_goal_parent_count')}))


if __name__=='__main__':main()
