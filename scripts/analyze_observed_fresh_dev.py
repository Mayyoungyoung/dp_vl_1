"""Post-hoc exhaustive fresh-DEV endpoint diagnostics, without model changes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs',type=Path,required=True)
    parser.add_argument('--observations',type=Path,required=True)
    parser.add_argument('--supervision',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    observations={r['id']:r for r in map(json.loads,args.observations.read_text().splitlines())}
    labels={r['id']:r for r in map(json.loads,args.supervision.read_text().splitlines())}
    index=json.loads((args.runs/'index.json').read_text())
    arms={};input_ids=None
    for entry in index['runs']:
        key=entry['method']+'_seed'+str(entry['seed']);folder=args.runs/key
        rows={r['scene_id']:r for r in json.loads((folder/'per_scene.json').read_text())}
        counts=Counter();details=[]
        with np.load(folder/'predictions.npz',allow_pickle=False) as archive:
            identifiers=list(map(str,archive['scene_ids']));endpoints=archive['paths'][:,:,-1]
            if input_ids is None:input_ids=identifiers
            if identifiers!=input_ids:raise ValueError('all arms must evaluate exact same ordered fresh inputs')
            for identifier,xyz in zip(identifiers,endpoints):
                specification=labels[identifier]['semantic_targets']
                centers=np.asarray(specification['centers']);target=int(specification['target_index']);tol=float(specification['tolerance'])
                if tol!=.03:raise ValueError('do not change endpoint tolerance in post-hoc diagnostic')
                distance=np.linalg.norm(xyz[:,None]-centers[None],axis=-1)
                nearest=distance.argmin(axis=1);success=(nearest==target)&(distance[:,target]<=tol)
                categories=['semantic_success' if ok else ('nearest_identity_wrong' if predicted!=target else 'nearest_identity_correct_beyond_3cm')
                            for ok,predicted in zip(success,nearest)]
                counts.update(categories)
                if float(success.mean())!=rows[identifier]['semantic_goal_accuracy']:raise ValueError('post-hoc criterion differs from original v2')
                details.append(dict(scene_id=identifier,parent_id=observations[identifier]['parent_id'],instruction=observations[identifier]['instruction'],
                    semantic_success_candidates=int(success.sum()),candidate_count=len(xyz),candidate_categories=categories,
                    target_center_error_m=distance[:,target].tolist(),nearest_target_indices=nearest.tolist(),expected_target_index=target,
                    reference_ADE_m=rows[identifier]['candidate_matched_ADE_m'],reference_endpoint_error_m=rows[identifier]['candidate_endpoint_error_m']))
        if len(details)!=48:raise ValueError('keep all 48 new development inputs')
        arms[key]=dict(candidate_category_counts=dict(counts),examples_with_no_semantic_candidate=sum(r['semantic_success_candidates']==0 for r in details),
            examples_with_any_semantic_failure=sum(r['semantic_success_candidates']<4 for r in details),all_scene_diagnostics=details,
            prediction_sha256=sha(folder/'predictions.npz'),per_scene_sha256=sha(folder/'per_scene.json'))
    pair_differences=[]
    for seed in (0,1,2):
        soft={r['scene_id']:r for r in arms['soft_seed'+str(seed)]['all_scene_diagnostics']}
        peak={r['scene_id']:r for r in arms['peak_seed'+str(seed)]['all_scene_diagnostics']}
        for identifier in input_ids:
            a,b=soft[identifier],peak[identifier]
            pair_differences.append(dict(seed=seed,scene_id=identifier,instruction=a['instruction'],
                semantic_gain=(b['semantic_success_candidates']-a['semantic_success_candidates'])/4,
                reference_ADE_change_m=b['reference_ADE_m']-a['reference_ADE_m'],
                target_center_error_change_m=float(np.mean(b['target_center_error_m'])-np.mean(a['target_center_error_m']))))
    persistent=[]
    for position,identifier in enumerate(input_ids):
        selected=[arms['peak_seed'+str(seed)]['all_scene_diagnostics'][position] for seed in (0,1,2)]
        if all(row['semantic_success_candidates']==0 for row in selected):
            persistent.append(dict(scene_id=identifier,instruction=selected[0]['instruction'],
                mean_center_errors_by_seed_m=[float(np.mean(row['target_center_error_m'])) for row in selected],
                candidate_categories_by_seed=[row['candidate_categories'] for row in selected]))
    result=dict(evaluation_protocol='observation_eval_v2',checkpoint_selection_protocol=index['checkpoint_selection_protocol'],
        posthoc_only=True,model_or_threshold_changes=False,source_observations_sha256=sha(args.observations),source_supervision_sha256=sha(args.supervision),
        source_index_sha256=sha(args.runs/'index.json'),script_sha256=sha(__file__),aggregate=index['aggregate'],
        arms=arms,all_144_scene_pair_differences=pair_differences,all_three_peak_seeds_have_zero_success=persistent,
        interpretation='nearest-identity categories are geometric endpoint diagnostics only; no image-cause or robot-execution inference')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(aggregate=result['aggregate'],categories={key:value['candidate_category_counts'] for key,value in arms.items()},persistent_failures=persistent),indent=2))


if __name__=='__main__':main()
