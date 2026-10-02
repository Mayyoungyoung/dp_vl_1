"""Observation-only direct-VLM SFT interfaces and deterministic budget rules.

Pure sampling/configuration helpers import no model runtime. Supervision stays
separate from the record accepted by the generation-prefix builder.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from .common import sha256
from .vlm_route_serialization import (PROTOCOL, assistant_only_labels, depth_image,
                                     parse_paths, prompt, serialize_paths)

TRAINING_PROTOCOL = 'vlm_route_sft_alternating_k1_k4_v1'
OBSERVATION_KEYS = {'id', 'parent_id', 'split', 'image', 'instruction'}
OBSERVED_INPUT_KEYS = {'instruction', 'current', 'camera_intrinsics', 'camera_extrinsics', 'rgb', 'depth'}


def validate_reserved_observation_manifest(path):
    """Inspect IDs/roles only, before opening any referenced image or route."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    expected = {f'obstacle_reach_{n}': 'TRAIN' if n < 272096 else 'DEV_MODEL' for n in range(272000,272104)}
    actual = {}
    ids = set()
    for row in rows:
        if set(row) != OBSERVATION_KEYS or row['id'] in ids:
            raise ValueError('Reserved observation manifest requires unique observation-only records')
        ids.add(row['id'])
        if expected.get(row['parent_id']) != row['split']:
            raise ValueError('Reserved parent identity/role mismatch; raw contents were not opened')
        actual[row['parent_id']] = row['split']
    if actual != expected:
        raise ValueError('Complete reserved96 TRAIN / 8 DEV parent manifest required before raw reads')
    return rows


@contextmanager
def exclusive_training_output(output, resume):
    """Block distinct job IDs from concurrently writing one training output."""
    output = Path(output)
    if resume:
        if not (output/'last.pt').is_file():
            raise FileNotFoundError('SFT resume requires original last.pt')
    else:
        output.mkdir(parents=True, exist_ok=False)
    lock = output/'train.lock'
    descriptor = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        os.write(descriptor, json.dumps(dict(pid=os.getpid(), code_commit=os.environ.get('CODE_COMMIT'))).encode())
        os.close(descriptor); descriptor = None
        yield output
    finally:
        if descriptor is not None:
            os.close(descriptor)
        lock.unlink(missing_ok=True)


def resolve(root, value):
    path = Path(value)
    return path if path.is_absolute() else Path(root) / path


def reference_indices(count, k, rng):
    if count < 1 or k not in (1, 4):
        raise ValueError('Positive references and training K1/K4 required')
    if k == 1:
        return np.asarray([rng.integers(count)], dtype=np.int64)
    if count > k:
        raise ValueError('K4 all-reference protocol requires at most four positive references')
    chosen = np.arange(count, dtype=np.int64)
    if count < k:
        chosen = np.r_[chosen, rng.integers(count, size=k-count)]
    return rng.permutation(chosen)


def alternating_k(step):
    if step < 1:
        raise ValueError('Optimizer steps are one based')
    return 1 if step % 2 else 4


def parent_language_groups(samples, split='TRAIN'):
    groups = {}
    for index, row in enumerate(samples):
        if row['split'] == split and len(row['paths']):
            groups.setdefault(row['parent_id'], []).append(index)
    return [sorted(groups[parent], key=lambda i: samples[i]['id']) for parent in sorted(groups)]


def sample_parent_language(groups, rng):
    if not groups:
        raise ValueError('No positive-reference TRAIN parents')
    choices = groups[int(rng.integers(len(groups)))]
    return int(choices[int(rng.integers(len(choices)))])


def fixed_development_plan(samples, seed):
    """One independently fixed K1 and K4 target per positive DEV instruction."""
    rng = np.random.default_rng(seed)
    plan = []
    for index in sorted(range(len(samples)), key=lambda i: samples[i]['id']):
        row = samples[index]
        if row['split'] != 'DEV_MODEL' or not len(row['paths']):
            continue
        for k in (1, 4):
            plan.append(dict(index=index, id=row['id'], parent_id=row['parent_id'], k=k,
                             reference_indices=reference_indices(len(row['paths']), k, rng).tolist()))
    if not plan:
        raise ValueError('At least one positive DEV instruction is required for token-NLL selection')
    return plan


def validate_resume_config(current, saved):
    # No continuation by silently changing planned steps, DEV cadence or input.
    if set(current) != set(saved):
        raise ValueError('SFT resume configuration keys differ')
    for key in current:
        if current[key] != saved[key]:
            raise ValueError('SFT resume config mismatch: ' + key)


def _array(value):
    return value.detach().cpu().numpy() if hasattr(value, 'detach') else np.asarray(value)


def user_messages(observed, k, horizon):
    if set(observed) != OBSERVED_INPUT_KEYS:
        raise ValueError('Generation prefix requires the exact observation-only whitelist')
    text = prompt(observed['instruction'], observed['current'], observed['camera_intrinsics'],
                  observed['camera_extrinsics'], k, horizon)
    messages = [dict(role='user', content=[dict(type='image'), dict(type='image'), dict(type='text', text=text)])]
    images = [observed['rgb'], Image.fromarray(depth_image(observed['depth']))]
    return messages, images


def prepare_prefix(processor, observed, k, horizon):
    """This function has no argument or access path for future routes/labels."""
    messages, images = user_messages(observed, k, horizon)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return processor(text=[text], images=images, return_tensors='pt')


def prepare_teacher_forcing(processor, observed, paths, events, k, horizon):
    paths, events = np.asarray(paths), np.asarray(events)
    if paths.shape != (k, horizon, 3) or events.shape != (k, horizon):
        raise ValueError('Exactly the requested positive target routes are required')
    messages, images = user_messages(observed, k, horizon)
    answer = serialize_paths(paths, events)
    prefix_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(messages + [dict(role='assistant', content=[dict(type='text', text=answer)])],
                                              tokenize=False, add_generation_prompt=False)
    prefix = processor(text=[prefix_text], images=images, return_tensors='pt')
    full = processor(text=[full_text], images=images, return_tensors='pt')
    prompt_ids = _array(prefix['input_ids'])[0]
    full_ids = _array(full['input_ids'])[0]
    labels = assistant_only_labels(full_ids, prompt_ids)
    parsed, _, receipt = parse_paths(answer, k, horizon)
    if receipt['format_valid_candidates'] != k:
        raise ValueError('Serialized positive target did not round-trip')
    metadata = dict(k=k, prompt_tokens=len(prompt_ids), sequence_tokens=len(labels),
                    supervised_tokens=int((labels != -100).sum()),
                    max_coordinate_quantization_error_m=float(np.abs(parsed-paths).max()),
                    prefix_exactly_matches=True, prompt_fully_masked=True,
                    generation_prefix_contains_no_answer=True)
    return full, labels[None], metadata


def read_observation(sample):
    with np.load(sample['observation_path'], allow_pickle=False) as source:
        current = np.r_[source['gripper_pose'].reshape(7), source['gripper_open'].reshape(1)]
        geometry = {key: source[key].copy() for key in ('depth', 'camera_intrinsics', 'camera_extrinsics')}
    with Image.open(sample['image_path']) as image:
        rgb = image.convert('RGB').copy()
    return dict(instruction=sample['instruction'], current=current, rgb=rgb, **geometry)


def load_sft_data(observations, supervision, horizon=24, resampler=None):
    """Read an isolated TRAIN/DEV export; raw geometry/semantic labels excluded.

    resampler injection supports pure interface fixtures; production always
    resolves the historical event-preserving reference resampler below.
    """
    if resampler is None:
        from .observed_route_head import resample_event_segments
        resampler = resample_event_segments
    observations, supervision = Path(observations).resolve(), Path(supervision).resolve()
    def read(path):
        return [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    rows, label_rows = read(observations), read(supervision)
    labels = {row['id']: row for row in label_rows}
    if len(labels) != len(label_rows) or len({row['id'] for row in rows}) != len(rows):
        raise ValueError('Duplicate observation/supervision identity')
    if set(labels) != {row['id'] for row in rows}:
        raise ValueError('Exact observation/supervision identity join required')
    hashes = {str(path): sha256(path) for path in (observations, supervision)}
    samples, parent_roles = [], {}
    for row in rows:
        if set(row) != OBSERVATION_KEYS or row['split'] not in ('TRAIN', 'DEV_MODEL'):
            raise ValueError('Use isolated observation-only TRAIN/DEV_MODEL export')
        label = labels[row['id']]
        if label['parent_id'] != row['parent_id'] or label['split'] != row['split']:
            raise ValueError('Observation/supervision parent or split mismatch')
        if parent_roles.setdefault(row['parent_id'], row['split']) != row['split']:
            raise ValueError('Parent leakage between TRAIN and DEV_MODEL')
        image = resolve(observations.parent, row['image']).resolve()
        current_file = resolve(supervision.parent, label['observation']).resolve()
        with np.load(current_file, allow_pickle=False) as archive:
            current = np.r_[archive['gripper_pose'].reshape(7), archive['gripper_open'].reshape(1)]
        if not np.isfinite(current).all():
            raise ValueError('Nonfinite current observation')
        paths, events = [], []
        for filename in label['routes']:
            route = resolve(supervision.parent, filename).resolve()
            with np.load(route, allow_pickle=False) as archive:
                xyz, opened = resampler(archive['gripper_pose'], archive['gripper_open'], horizon)
            if np.linalg.norm(xyz[0]-current[:3]) > .005:
                raise ValueError('Positive reference starts outside recorded current state')
            paths.append(xyz); events.append(opened); hashes[str(route)] = sha256(route)
        if len(paths) > 4:
            raise ValueError('K4 all-reference protocol does not support more than four positives')
        for path in (image, current_file):
            hashes[str(path)] = sha256(path)
        samples.append(dict(id=row['id'], parent_id=row['parent_id'], split=row['split'],
                            instruction=row['instruction'], image_path=str(image), observation_path=str(current_file),
                            paths=np.asarray(paths, dtype=np.float32).reshape(-1, horizon, 3),
                            events=np.asarray(events, dtype=np.float32).reshape(-1, horizon)))
    if not parent_language_groups(samples):
        raise ValueError('Positive-reference TRAIN data required')
    fingerprint = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    return dict(samples=samples, source_sha256=hashes, fingerprint=fingerprint,
                protocol=PROTOCOL, parents_by_split={split: len({s['parent_id'] for s in samples if s['split']==split})
                    for split in ('TRAIN', 'DEV_MODEL')},
                unreferenced=[{key:s[key] for key in ('id','parent_id','split')} for s in samples if not len(s['paths'])])
