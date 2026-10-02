"""Eight observation-only TRAIN K1 requests, distinct from DEV comparison."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .common import sha256
from .vlm_sft_data import resolve
from .vlm_sft_evaluation import GENERATION_SAMPLE_KEYS, request_seed, safe_parse_paths

PARENTS = ['obstacle_reach_%d' % n for n in range(272000, 272008)]


def validate_probe_samples(samples):
    if len(samples) != 8 or [s['parent_id'] for s in samples] != PARENTS:
        raise ValueError('Exactly eight predeclared TRAIN parents required')
    for sample in samples:
        if (set(sample) != GENERATION_SAMPLE_KEYS or sample['split'] != 'TRAIN'
                or sample['id'] != sample['parent_id']+'_target0'):
            raise ValueError('Only predeclared observation-only TRAIN target0 inputs allowed')


def train_probe_inputs(config, index, plan):
    if hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest() != plan['training_data_fingerprint']:
        raise ValueError('Original SFT index changed')
    if [r['parent_id'] for r in plan['samples']] != PARENTS:
        raise ValueError('Fixed TRAIN8 sampling plan required')
    manifest = Path(config['observations'])
    if sha256(manifest) != plan['observations_sha256'] or index.get(str(manifest.resolve())) != sha256(manifest):
        raise ValueError('Original observation manifest changed')
    metadata = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    samples, hashes = [], {}
    for declared in plan['samples']:
        rows = sorted([row for row in metadata if row['parent_id'] == declared['parent_id']], key=lambda row: row['id'])
        if not rows or rows[0]['id'] != declared['id'] or declared['id'] != declared['parent_id']+'_target0':
            raise ValueError('Fixed first observation missing; no substitute')
        if any(row['split'] != 'TRAIN' for row in rows) or len({row['id'] for row in rows}) != len(rows):
            raise ValueError('TRAIN parent role/identity changed')
        row = rows[0]
        if set(row) != {'id', 'parent_id', 'split', 'instruction', 'image'}:
            raise ValueError('Observation-only metadata required')
        image = resolve(manifest.parent, row['image']).resolve()
        current = image.with_name('observation.npz')
        if image.name != 'front.png' or image.parent.name != row['parent_id']:
            raise ValueError('Current input outside declared parent')
        for path, key in ((image, 'image_sha256'), (current, 'observation_sha256')):
            hashes[str(path)] = sha256(path)
            if hashes[str(path)] != declared[key] or index.get(str(path)) != hashes[str(path)]:
                raise ValueError('Original TRAIN input changed')
        samples.append(dict(id=row['id'], parent_id=row['parent_id'], split='TRAIN', instruction=row['instruction'],
                            image_path=str(image), observation_path=str(current)))
    validate_probe_samples(samples)
    return samples, hashes


def generate_train_probe(samples, generate_request, output, seed=0):
    validate_probe_samples(samples)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    paths, events, records = [], [], []
    for sample in samples:
        started = time.perf_counter(); text, token_metadata, failure = '', {}, None
        decoding_seed = request_seed(seed, 0, sample['id'], 'train8_constrained_k1', 0)
        try:
            result = generate_request(sample, 1, decoding_seed, 512)
            text, token_metadata = result['text'], result['tokens']
            if not isinstance(text, str): raise TypeError('Raw decoded text required')
        except Exception as error:
            failure = dict(type=type(error).__name__, message=str(error)); text = ''
        xyz, opened, receipt = safe_parse_paths(text, 1, 24)
        record = dict(scene_id=sample['id'], parent_id=sample['parent_id'], split='TRAIN',
            method='train8_constrained_k1', repeat=0, request=0, k=1, horizon=24, max_new_tokens=512,
            decoding_seed=decoding_seed, text=text, tokens=token_metadata, failure=failure, parse=receipt,
            observed_input_to_decoded_routes_seconds=time.perf_counter()-started)
        records.append(record); paths.append(xyz); events.append(opened)
        with (output/'requests.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, allow_nan=False)+'\n'); handle.flush()
    np.savez_compressed(output/'predictions.npz', scene_ids=np.asarray([s['id'] for s in samples]),
        parent_ids=np.asarray(PARENTS), paths=np.stack(paths), gripper_open=np.stack(events))
    result = dict(scope='TRAIN-only capacity/format probe; not independent4 versus whole4 or a DEV result',
        examples=8, parents=8, requested_autoregressive_calls=8, requested_candidate_slots=8,
        charged_candidate_slots=sum(r['parse']['charged_candidate_slots'] for r in records),
        strict_finite_slots=int(np.isfinite(np.stack(paths)).all(axis=(2, 3)).sum()),
        failed_requests=sum(r['failure'] is not None for r in records),
        reached_token_limit=sum(r['tokens'].get('reached_token_limit', False) for r in records),
        total_generation_seconds=sum(r['observed_input_to_decoded_routes_seconds'] for r in records),
        geometry_opened=False, candidate_pooling_between_repeats=False)
    (output/'probe_summary.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return [result]
