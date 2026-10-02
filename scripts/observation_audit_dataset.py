"""Summarize real observation collection without promoting near-duplicates to modes."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from observation_cache_qwen import read_manifest, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    observations = read_manifest(a.data / 'observations.jsonl')
    attempts = [json.loads(line) for line in (a.data / 'attempts.jsonl').read_text().splitlines() if line]
    supervision_path = a.data / 'supervision.jsonl'
    supervision = [json.loads(line) for line in supervision_path.read_text().splitlines() if line] if supervision_path.exists() else []
    parents = {}
    image_hashes = {}
    for row in observations:
        parents[row['parent_id']] = row['split']
        image_hashes[row['id']] = sha256(a.data / row['image'])
    by_parent = {}
    for row in observations:
        by_parent.setdefault(row['parent_id'], []).append(row)
    for parent_id, rows in by_parent.items():
        if len({image_hashes[row['id']] for row in rows}) != 1:
            raise ValueError('Within-parent goal instructions use different images: ' + parent_id)
    successes = [row for row in attempts if row.get('success')]
    final_errors = [row['endpoint_error_m'] for row in successes if 'endpoint_error_m' in row]
    route_files = [name for row in supervision for name in row['routes']]
    finite_routes, routes_matching_acceptance = 0, 0
    current_state_dims = set()
    for row in supervision:
        with np.load(a.data / row['observation']) as current:
            current_state_dims.add(tuple(current['gripper_pose'].shape))
            if 'task_low_dim_state' in current.files:
                raise ValueError('Forbidden task state in observation archive')
        centers = row['semantic_targets']['centers']
        goal = np.asarray(centers[row['semantic_targets']['target_index']])
        for filename in row['routes']:
            with np.load(a.data / filename) as route:
                xyz = route['gripper_pose'][:, :3]
                finite_routes += int(np.isfinite(xyz).all())
                error = float(np.linalg.norm(xyz[-1] - goal))
                routes_matching_acceptance += int(error <= row['semantic_targets']['tolerance'])
    if finite_routes != len(route_files) or routes_matching_acceptance != len(route_files):
        raise ValueError('Saved route violates finite/endpoint acceptance audit')
    file_records = []
    for path in sorted(a.data.rglob('*')):
        if path.is_file() and 'qwen_cache' not in path.parts:
            file_records.append({'path': str(path.relative_to(a.data)), 'bytes': path.stat().st_size, 'sha256': sha256(path)})
    a.output.mkdir(parents=True, exist_ok=True)
    (a.output / 'dataset_file_hashes.json').write_text(json.dumps(file_records, indent=2), encoding='utf-8')
    report = {'dataset': str(a.data), 'parents': len(parents), 'parents_by_split': dict(Counter(parents.values())),
        'observation_instruction_pairs': len(observations), 'unique_initial_image_hashes': len(set(image_hashes.values())),
        'same_image_distinct_instructions_per_parent': sorted(set(len({r['instruction'] for r in rows}) for rows in by_parent.values())),
        'attempts': len(attempts), 'successful_attempts': len(successes),
        'failure_reasons': dict(Counter(row.get('error', 'unknown') for row in attempts if not row.get('success'))),
        'near_duplicate_successes': sum(bool(row.get('near_duplicate')) for row in successes),
        'restore_max_abs': max((row.get('restore', {}).get('max_abs', 0) for row in attempts), default=0),
        'restore_rgb_max_difference': max((row.get('restore', {}).get('rgb_max_difference', 0) for row in attempts), default=0),
        'saved_routes': len(route_files), 'finite_saved_routes': finite_routes,
        'endpoint_checked_saved_routes': routes_matching_acceptance,
        'endpoint_error_mean_m': float(np.mean(final_errors)) if final_errors else None,
        'endpoint_error_max_m': max(final_errors) if final_errors else None,
        'all_current_pose_shapes': [list(shape) for shape in sorted(current_state_dims)],
        'unique_valid_route_types': None,
        'scope': 'collection and input-isolation audit; not learned model performance or continuous collision certification',
        'hash_manifest_sha256': sha256(a.output / 'dataset_file_hashes.json')}
    cache = a.data / 'qwen_cache'
    if cache.exists():
        samples = [json.loads(line) for line in (cache / 'samples.jsonl').read_text().splitlines() if line]
        if len(samples) != len(observations) or {row['id'] for row in samples} != set(image_hashes):
            raise ValueError('Qwen cache ids do not exactly match observation manifest')
        cached_by_id, cache_hashes = {}, []
        for row in samples:
            path = cache / row['file']
            if sha256(path) != row['sha256']:
                raise ValueError('Qwen sample hash mismatch')
            with np.load(path) as features:
                if str(features['image_sha256']) != image_hashes[row['id']]:
                    raise ValueError('Qwen feature input image differs from observation')
                if not np.isfinite(features['mean_hidden']).all() or not np.isfinite(features['last_hidden']).all():
                    raise ValueError('Nonfinite Qwen cache features')
                cached_by_id[row['id']] = (features['mean_hidden'].copy(), features['last_hidden'].copy())
                cache_hashes.append({'id': row['id'], 'file': row['file'], 'sha256': row['sha256']})
        (a.output / 'qwen_cache_hashes.json').write_text(json.dumps(cache_hashes, indent=2), encoding='utf-8')
        latencies = [row['single_request_seconds'] for row in samples]
        pair_l2 = []
        for rows in by_parent.values():
            anchor = cached_by_id[rows[0]['id']][1]
            pair_l2.extend(float(np.linalg.norm(anchor-cached_by_id[row['id']][1])) for row in rows[1:])
        report['qwen_cache'] = {'samples': len(samples), 'finite_and_hash_checked': len(samples),
            'hidden_shapes': [list(x.shape) for x in next(iter(cached_by_id.values()))],
            'single_request_seconds_median': float(np.median(latencies)),
            'single_request_seconds_p95': float(np.quantile(latencies, .95)),
            'first_single_request_seconds': latencies[0],
            'same_image_language_last_hidden_l2_min': min(pair_l2),
            'status': json.loads((cache / 'status.json').read_text()),
            'config': json.loads((cache / 'cache_config.json').read_text()),
            'hash_manifest_sha256': sha256(a.output / 'qwen_cache_hashes.json')}
    (a.output / 'audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
