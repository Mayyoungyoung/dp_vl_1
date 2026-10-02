"""Endpoint-only diagnostic of complete raw JSON; never repair/score routes.

Wrong-horizon/missing-candidate outputs retain their original failed status.
This separate descriptive measurement does not replace any main metric.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json


def endpoint_records(requests, labels):
    records, failures = [], []
    for request in requests:
        try: decoded = json.loads(request['text'])
        except (ValueError, RecursionError):
            failures.append(dict(scene_id=request['scene_id'], request=request['request'], reason='complete_text_json_invalid'))
            continue
        if not isinstance(decoded, list): raise ValueError('Unexpected complete JSON type')
        label = labels[request['scene_id']]
        if label['split'] != 'DEV_MODEL' or label['parent_id'] != request['parent_id']:
            raise ValueError('Original DEV identity join required')
        specification = label['semantic_targets']
        if specification['tolerance'] != .03: raise ValueError('Original semantic tolerance required')
        centers = np.asarray(specification['centers'], dtype=np.float64)
        target = specification['target_index']
        for slot, route in enumerate(decoded):
            # Do not complete malformed points, parse JSON fragments or resample.
            if (not isinstance(route, list) or not route or any(not isinstance(point, list) or len(point) != 4
                    or any(type(value) is not int for value in point) for point in route)):
                failures.append(dict(scene_id=request['scene_id'], request=request['request'], slot=slot, reason='malformed_points'))
                continue
            endpoint = np.asarray(route[-1][:3], dtype=np.float64)/1000
            distances = np.linalg.norm(centers-endpoint, axis=-1)
            records.append(dict(scene_id=request['scene_id'], parent_id=request['parent_id'], request=request['request'],
                slot_in_raw_json=slot, decoded_route_count=len(decoded), requested_routes=request['k'],
                raw_horizon=len(route), exact_horizon=len(route)==24, endpoint_m=endpoint.tolist(),
                requested_target_distance_m=float(distances[target]), nearest_target_is_requested=bool(distances.argmin()==target),
                within_original_3cm=bool(distances[target]<=.03),
                original_request_format_valid_slots=request['parse']['format_valid_candidates'],
                original_main_scores_unchanged=True))
    return records, failures


def describe(records):
    values = np.asarray([r['requested_target_distance_m'] for r in records])
    if not len(values): return dict(count=0)
    per_scene = {}
    for scene in sorted({r['scene_id'] for r in records}):
        per_scene[scene] = float(np.mean([r['requested_target_distance_m'] for r in records if r['scene_id']==scene]))
    return dict(count=len(values), scenes_with_parseable_endpoints=len(per_scene), min_m=float(values.min()),
        median_m=float(np.median(values)), candidate_mean_m=float(values.mean()), max_m=float(values.max()),
        mean_over_evaluable_scene_means_m=float(np.mean(list(per_scene.values()))),
        within_3cm_count=sum(r['within_original_3cm'] for r in records),
        nearest_requested_and_within_3cm_count=sum(r['within_original_3cm'] and r['nearest_target_is_requested'] for r in records))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--supervision-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh endpoint diagnostic required')
    generated=json.loads((args.generation/'summary.json').read_text())
    if generated['status']!='completed' or generated['repeats']!=1 or generated['examples']!=24:
        raise ValueError('Original complete 120-request comparison required')
    journals={}
    for method, count in [('independent4',96),('whole4',24)]:
        relative=f'repeat0/{method}/requests.jsonl'; path=args.generation/relative
        if sha256(path)!=generated['artifacts'][relative]['sha256']: raise ValueError('Frozen journal changed')
        journals[method]=[json.loads(line) for line in path.read_text().splitlines()]
        if len(journals[method])!=count: raise ValueError('Complete original request count required')
    # Evaluation-only metadata is opened only after both complete pools verify.
    if sha256(args.supervision_metadata)!='37da38855af0357732ac48f125f140e8ed5d1a60008f1424fd5080c7a9421237':
        raise ValueError('Original reserved observation supervision metadata hash required')
    labels={row['id']:row for row in map(json.loads,args.supervision_metadata.read_text().splitlines()) if row['split']=='DEV_MODEL'}
    if len(labels)!=24 or {r['parent_id'] for r in labels.values()}!={'obstacle_reach_%d'%n for n in range(272096,272104)}:
        raise ValueError('Exact original eight DEV parents required')
    results={}
    for method,requests in journals.items():
        records,failures=endpoint_records(requests,labels)
        results[method]=dict(requests=len(requests),complete_json_requests=len(requests)-sum(f['reason']=='complete_text_json_invalid' for f in failures),
            endpoint_description_all_complete_json=describe(records),
            endpoint_description_exact_horizon=describe([r for r in records if r['exact_horizon']]),
            endpoint_description_wrong_horizon=describe([r for r in records if not r['exact_horizon']]),
            raw_route_horizon_counts=dict(Counter(str(r['raw_horizon']) for r in records)),
            records=records,unparsed_failures=failures)
    args.output.mkdir(parents=True)
    write_json(args.output/'summary.json',dict(protocol='vlm_raw_complete_json_endpoint_diagnostic_v1',
        generation_summary_sha256=sha256(args.generation/'summary.json'),supervision_metadata_sha256=sha256(args.supervision_metadata),
        analyzer_sha256=sha256(__file__),generation_source_commit=generated['code_commit'],checkpoint_sha256=generated['checkpoint_sha256'],
        results=results,model_calls=0,reference_route_files_opened=0,verification_geometry_files_opened=0,repaired_texts=0,
        scope='Separate endpoint-only description, including wrong-horizon and missing-candidate complete JSON. Those outputs remain main-format failures; no route validity/coverage is recalculated.'))
    print(json.dumps({key: value['endpoint_description_all_complete_json'] for key,value in results.items()},indent=2))


if __name__=='__main__': main()
