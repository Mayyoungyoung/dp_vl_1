"""Weak instruction-mean endpoint control; no image or current-state input."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from routeset.common import sha256, write_json
from routeset.observed_route_head import semantic_endpoint_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    observations = [json.loads(line) for line in (args.data/'observations.jsonl').read_text().splitlines() if line.strip()]
    labels = {row['id']: row for row in [json.loads(line) for line in (args.data/'supervision.jsonl').read_text().splitlines() if line.strip()]}
    examples = defaultdict(list)
    train_ids, train_parents = [], set()
    source_hashes = {str(args.data/name):sha256(args.data/name) for name in ('observations.jsonl', 'supervision.jsonl')}
    for row in observations:
        if row['split'] != 'TRAIN':
            continue
        label = labels[row['id']]
        if not label.get('routes'):
            continue
        endpoints = []
        for name in label['routes']:
            route = args.data/name
            with np.load(route, allow_pickle=False) as archive:
                endpoints.append(archive['gripper_pose'][-1, :3].astype(np.float64))
            source_hashes[str(route)] = sha256(route)
        # Each parent/instruction contributes one mean, independent of how
        # many successful demonstrations the collector found for that item.
        examples[row['instruction']].append(np.mean(endpoints, axis=0))
        train_ids.append(row['id']); train_parents.add(row['parent_id'])
    means = {instruction:np.mean(values, axis=0) for instruction,values in examples.items()}
    global_train_mean = np.mean([value for values in examples.values() for value in values], axis=0)
    rows = []
    for row in observations:
        if row['split'] != 'DEV_MODEL':
            continue
        if row['parent_id'] in train_parents:
            raise ValueError('parent split leak')
        label = labels[row['id']]
        endpoint = means.get(row['instruction'], global_train_mean)
        specification = label['semantic_targets']
        success = semantic_endpoint_accuracy(endpoint[None], specification)
        goal = np.asarray(specification['centers'][specification['target_index']])
        reference_endpoints = []
        for filename in label.get('routes', []):
            with np.load(args.data/filename, allow_pickle=False) as archive:
                reference_endpoints.append(archive['gripper_pose'][-1, :3])
        rows.append(dict(id=row['id'], parent_id=row['parent_id'], instruction=row['instruction'],
                         prediction_endpoint=endpoint.tolist(), train_instruction_examples=len(examples.get(row['instruction'], [])),
                         unseen_instruction_fallback=row['instruction'] not in means,
                         semantic_goal_accuracy=float(success[0]), goal_error_m=float(np.linalg.norm(endpoint-goal)),
                         reference_endpoint_error_m=float(np.linalg.norm(np.asarray(reference_endpoints)-endpoint, axis=-1).min()) if reference_endpoints else None,
                         reference_count=len(reference_endpoints)))
    result = dict(evaluation_protocol='observation_eval_v2', model='TRAIN instruction-mean endpoint control',
                  scope='weak diagnostic of spatial priors; no path generation, strong baseline or robot-validity claim',
                  inputs=['instruction string only'], candidate_count=1, train_supervised_examples=len(train_ids),
                  train_parents=len(train_parents), examples=len(rows), semantic_evaluation_examples=len(rows),
                  reference_evaluation_examples=sum(row['reference_count'] > 0 for row in rows),
                  semantic_goal_accuracy=float(np.mean([row['semantic_goal_accuracy'] for row in rows])),
                  goal_error_m=float(np.mean([row['goal_error_m'] for row in rows])),
                  reference_endpoint_error_m=float(np.mean([row['reference_endpoint_error_m'] for row in rows if row['reference_endpoint_error_m'] is not None])),
                  unseen_instruction_count=sum(row['unseen_instruction_fallback'] for row in rows),
                  estimator='mean recorded terminal xyz per TRAIN observation, then mean across exact instruction; unseen instruction uses global TRAIN observation mean',
                  evaluation_rule='original nearest-target identity AND distance <= 0.03m; no threshold change',
                  training_instruction_means={key:dict(mean=value.tolist(), examples=len(examples[key])) for key,value in means.items()},
                  source_sha256=source_hashes, script_sha256=sha256(Path(__file__)), per_scene=rows)
    write_json(args.output, result)
    print(json.dumps({key:value for key,value in result.items() if key not in ('source_sha256', 'training_instruction_means', 'per_scene')}), flush=True)


if __name__ == '__main__':
    main()
