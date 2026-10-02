"""Closed-instruction RGB-D prototype localization baseline, candidate budget one.

Training uses visible pixels near positive demonstration endpoints. Prediction
takes only RGB, depth, camera calibration and an exact instruction string.
Evaluation goal labels are never arguments to prediction or point selection.
This estimates one observed surface point, not a route or robot execution.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy import ndimage


INPUT_KEYS = {'id', 'parent_id', 'split', 'image', 'instruction'}
CONFIG = dict(endpoint_neighborhood_m=.04, color_scale_floor=.025,
              training_distance_quantile=.99, minimum_component_pixels=3,
              maximum_depth_m=10., connectivity=8,
              extent_log_scale=1., prototype='median RGB and smoothed chromaticity',
              component_rule='mean color likelihood times soft TRAIN metric-extent prior',
              prediction_rule='observed point nearest component world-coordinate median')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def feature_colors(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)
    return np.concatenate([rgb, (rgb+.02)/(rgb.sum(-1, keepdims=True)+.06)], axis=-1)


def observed_grid(rgb, depth, intrinsics, camera_to_world):
    """NumPy backprojection, with supplied focal signs and no image flip."""
    depth = np.asarray(depth, dtype=np.float64)
    if rgb.shape != depth.shape+(3,) or intrinsics.shape != (3, 3):
        raise ValueError('matching RGB-D and 3x3 camera intrinsics required')
    if camera_to_world.shape not in ((3, 4), (4, 4)):
        raise ValueError('camera-to-world must be 3x4 or 4x4')
    yy, xx = np.indices(depth.shape)
    rays = np.stack([xx, yy, np.ones_like(xx)], -1) @ np.linalg.inv(intrinsics).T
    valid = np.isfinite(depth) & (depth > 0) & (depth <= CONFIG['maximum_depth_m'])
    xyz = (rays*np.where(valid, depth, 0.)[..., None]) @ camera_to_world[:3, :3].T + camera_to_world[:3, 3]
    rgb = rgb.astype(np.float64)/255.
    return rgb, xyz, valid


def load_observation(dataset, observation, observation_pointer):
    if set(observation) != INPUT_KEYS:
        raise ValueError('observation input whitelist violated')
    image_path = dataset/observation['image']
    current_path = dataset/observation_pointer
    rgb = np.asarray(Image.open(image_path).convert('RGB'))
    with np.load(current_path, allow_pickle=False) as archive:
        # Access only the declared current RGB-D camera fields.
        grid = observed_grid(rgb, archive['depth'], archive['camera_intrinsics'], archive['camera_extrinsics'])
    return grid, [image_path, current_path]


def metric_extent(points):
    return float(np.linalg.norm(np.quantile(points, .95, axis=0)-np.quantile(points, .05, axis=0)))


def fit_prototypes(dataset, observations, labels):
    started = time.perf_counter()
    colors, extents, sample_records = defaultdict(list), defaultdict(list), []
    train_parents, source_hashes = set(), {}
    for row in observations:
        if row['split'] != 'TRAIN':
            continue
        label = labels[row['id']]
        train_parents.add(row['parent_id'])
        if not label.get('routes'):
            sample_records.append(dict(id=row['id'], used=False, reason='no_positive_reference'))
            continue
        endpoints = []
        for name in label['routes']:
            path = dataset/name
            with np.load(path, allow_pickle=False) as archive:
                endpoints.append(archive['gripper_pose'][-1, :3])
            source_hashes[str(path)] = digest(path)
        (rgb, xyz, valid), paths = load_observation(dataset, row, label['observation'])
        source_hashes.update({str(path):digest(path) for path in paths})
        # Positive mixture: no average endpoint between possibly distinct goals.
        distance = np.linalg.norm(xyz[..., None, :]-np.asarray(endpoints)[None, None], axis=-1).min(-1)
        selected = valid & (distance <= CONFIG['endpoint_neighborhood_m'])
        count = int(selected.sum())
        if count < CONFIG['minimum_component_pixels']:
            sample_records.append(dict(id=row['id'], used=False, reason='insufficient_visible_positive_pixels', pixels=count))
            continue
        colors[row['instruction']].append(feature_colors(rgb[selected]))
        extents[row['instruction']].append(metric_extent(xyz[selected]))
        sample_records.append(dict(id=row['id'], used=True, instruction=row['instruction'], pixels=count,
                                   visible_extent_m=extents[row['instruction']][-1], positive_references=len(endpoints)))
    prototypes = {}
    for instruction, values in colors.items():
        # Each independent parent contributes equal total weight to center and
        # dispersion summaries; many foreground pixels do not increase exposure.
        center = np.median(np.stack([np.median(value, axis=0) for value in values]), axis=0)
        scale = np.maximum(np.median(np.stack([np.quantile(np.abs(value-center), .68, axis=0) for value in values]), axis=0),
                           CONFIG['color_scale_floor'])
        distances = [(((value-center)/scale)**2).mean(-1) for value in values]
        threshold = max(float(np.median([np.quantile(value, CONFIG['training_distance_quantile']) for value in distances])), 1.)
        prototypes[instruction] = dict(center=center.tolist(), scale=scale.tolist(), threshold=threshold,
                                       extent_m=float(np.median(extents[instruction])),
                                       train_examples=len(values), train_pixels=sum(len(value) for value in values))
    return dict(prototypes=prototypes, config=CONFIG, training_examples=sum(row['used'] for row in sample_records),
                training_parents=len(train_parents), training_seconds=time.perf_counter()-started,
                training_rows=sample_records, training_source_sha256=source_hashes), train_parents


def predict(rgb, xyz, valid, instruction, model):
    """No labels, target centers, segmentation masks or future paths allowed."""
    if instruction not in model['prototypes']:
        return None, dict(status='unsupported_exact_instruction', components_considered=0, candidates_emitted=0)
    prototype = model['prototypes'][instruction]
    features = feature_colors(rgb)
    center, scale = np.asarray(prototype['center']), np.asarray(prototype['scale'])
    distance = (((features-center)/scale)**2).mean(-1)
    foreground = valid & (distance <= prototype['threshold'])
    components, count = ndimage.label(foreground, structure=np.ones((3, 3), dtype=np.uint8))
    choices = []
    for index, bounds in enumerate(ndimage.find_objects(components), 1):
        if bounds is None:
            continue
        region = components[bounds] == index
        size = int(region.sum())
        if size < CONFIG['minimum_component_pixels']:
            continue
        points = xyz[bounds][region]
        extent = metric_extent(points)
        extent_log_ratio = np.log(max(extent, 1e-6)/max(prototype['extent_m'], 1e-6))
        size_score = np.exp(-.5*(extent_log_ratio/CONFIG['extent_log_scale'])**2)
        color_score = float(np.exp(-.5*distance[bounds][region]).mean())
        score = color_score*size_score
        median = np.median(points, axis=0)
        endpoint = points[np.linalg.norm(points-median, axis=-1).argmin()]
        choices.append((score, index, endpoint, dict(pixels=size, observed_extent_m=extent,
                                                     color_score=color_score, extent_score=float(size_score))))
    if not choices:
        return None, dict(status='no_supported_component', components_considered=int(count), candidates_emitted=0)
    score, index, endpoint, information = max(choices, key=lambda item:(item[0], -item[1]))
    return endpoint, dict(status='predicted', components_considered=int(count), eligible_components=len(choices),
                         candidates_emitted=1, selected_component=index, selected_component_score=score, **information)


def evaluate_endpoint(endpoint, specification):
    if endpoint is None:
        return dict(semantic_goal_accuracy=0., goal_error_m=None, nearest_target_index=None)
    distances = np.linalg.norm(np.asarray(specification['centers'])-endpoint, axis=-1)
    index = specification['target_index']
    return dict(semantic_goal_accuracy=float(int(distances.argmin()) == index and distances[index] <= specification['tolerance']),
                goal_error_m=float(distances[index]), nearest_target_index=int(distances.argmin()))


def self_test():
    rgb = np.zeros((30, 40, 3), dtype=np.float64)
    xyz = np.stack(np.meshgrid(np.arange(40)*.002, np.arange(30)*.002), -1)
    xyz = np.concatenate([xyz, np.ones((30, 40, 1))], -1)
    rgb[5:12, 5:12] = [1., 0., 0.]
    rgb[15:22, 25:32] = [0., 0., 1.]
    prototypes = {}
    for instruction, color in [('reach red', [1., 0., 0.]), ('reach blue', [0., 0., 1.])]:
        prototypes[instruction] = dict(center=feature_colors(np.asarray(color)).tolist(), scale=[.025]*6, threshold=1., extent_m=.017)
    model = dict(prototypes=prototypes)
    for instruction, expected in [('reach red', [.016, .016, 1.]), ('reach blue', [.056, .036, 1.])]:
        endpoint, info = predict(rgb, xyz, np.ones((30, 40), bool), instruction, model)
        assert np.linalg.norm(endpoint-expected) < 1e-8 and info['candidates_emitted'] == 1
    endpoint, info = predict(rgb, xyz, np.ones((30, 40), bool), 'unseen', model)
    assert endpoint is None and info['status'] == 'unsupported_exact_instruction'
    # Camera sign/axis convention independently known for a single noncentral pixel.
    calibration = np.array([[-2., 0., 0.], [0., -2., 0.], [0., 0., 1.]])
    _, coordinates, _ = observed_grid(np.zeros((2, 2, 3), np.uint8), np.ones((2, 2)), calibration, np.eye(4))
    assert np.allclose(coordinates[1, 1], [-.5, -.5, 1.])
    assert evaluate_endpoint(None, {'centers':[[0, 0, 0]], 'target_index':0, 'tolerance':.03})['semantic_goal_accuracy'] == 0.
    print('prototype self-test passed: two language targets, rejection, signed camera, abstention denominator')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.data is None or args.output is None:
        parser.error('--data and --output are required')
    if args.output.exists():
        raise FileExistsError('preserve existing experiment: '+str(args.output))
    observations = read_rows(args.data/'observations.jsonl')
    labels = {row['id']:row for row in read_rows(args.data/'supervision.jsonl')}
    if len({row['id'] for row in observations}) != len(observations):
        raise ValueError('duplicate input IDs')
    for row in observations:
        if set(row) != INPUT_KEYS or labels[row['id']].get('split', row['split']) != row['split']:
            raise ValueError('input whitelist or split mismatch')
    model, train_parents = fit_prototypes(args.data, observations, labels)
    dev = [row for row in observations if row['split'] == 'DEV_MODEL']
    if train_parents & {row['parent_id'] for row in dev}:
        raise ValueError('parent split leak')
    output_rows, evaluation_hashes, predictions = [], {}, {}
    for row in dev:
        label = labels[row['id']]
        started = time.perf_counter()
        (rgb, xyz, valid), paths = load_observation(args.data, row, label['observation'])
        endpoint, details = predict(rgb, xyz, valid, row['instruction'], model)
        elapsed = time.perf_counter()-started
        predictions[row['id']] = endpoint
        evaluation_hashes.update({str(path):digest(path) for path in paths})
        # The first access to semantic labels occurs AFTER prediction and timer.
        metrics = evaluate_endpoint(endpoint, label['semantic_targets'])
        references = []
        for name in label.get('routes', []):
            path = args.data/name
            with np.load(path, allow_pickle=False) as archive:
                references.append(archive['gripper_pose'][-1, :3])
            evaluation_hashes[str(path)] = digest(path)
        output_rows.append(dict(id=row['id'], parent_id=row['parent_id'], instruction=row['instruction'],
                                prediction_endpoint=endpoint.tolist() if endpoint is not None else None,
                                latency_seconds=elapsed, reference_count=len(references),
                                reference_endpoint_error_m=float(np.linalg.norm(np.asarray(references)-endpoint, axis=-1).min()) if references and endpoint is not None else None,
                                **details, **metrics))
    by_parent = defaultdict(list)
    for row in dev:
        by_parent[row['parent_id']].append(row)
    swap_rows = []
    for parent, group in by_parent.items():
        if len({digest(args.data/row['image']) for row in group}) != 1:
            raise ValueError('paired language comparison requires identical image content')
        for before in group:
            for after in group:
                if before['id'] == after['id']:
                    continue
                first, second = predictions[before['id']], predictions[after['id']]
                swap_rows.append(dict(parent_id=parent, original_id=before['id'], changed_instruction_id=after['id'],
                                      endpoint_response_m=float(np.linalg.norm(first-second)) if first is not None and second is not None else None,
                                      changed_instruction_semantic_accuracy=evaluate_endpoint(second, labels[after['id']]['semantic_targets'])['semantic_goal_accuracy'],
                                      stale_instruction_semantic_accuracy=evaluate_endpoint(first, labels[after['id']]['semantic_targets'])['semantic_goal_accuracy']))
    latencies = np.asarray([row['latency_seconds'] for row in output_rows])
    errors = [row['goal_error_m'] for row in output_rows if row['goal_error_m'] is not None]
    reference_errors = [row['reference_endpoint_error_m'] for row in output_rows if row['reference_endpoint_error_m'] is not None]
    report = dict(evaluation_protocol='observation_eval_v2', baseline='TRAIN endpoint-supervised color prototype RGB-D localization',
                  scope='closed exact instructions, one observed endpoint; no path, robot validity or open-vocabulary claim',
                  data=str(args.data.resolve()), script_sha256=digest(__file__), config=CONFIG,
                  inputs=['current RGB', 'current optical-axis depth', 'camera intrinsics and camera-to-world', 'exact instruction'],
                  candidate_budget=1, proposal_budget_includes_abstentions=True,
                  internal_components_are='image segmentation hypotheses, not generated route candidates; all counts recorded',
                  goal_rule='original nearest-target identity AND <=0.03m; failed/unknown localization counted as zero',
                  examples=len(dev), semantic_evaluation_examples=len(dev),
                  reference_evaluation_examples=sum(row['reference_count'] > 0 for row in output_rows),
                  semantic_goal_accuracy=float(np.mean([row['semantic_goal_accuracy'] for row in output_rows])),
                  unknown_instruction_count=sum(row['status'] == 'unsupported_exact_instruction' for row in output_rows),
                  abstention_count=sum(row['candidates_emitted'] == 0 for row in output_rows),
                  goal_error_m_conditional_on_prediction=float(np.mean(errors)) if errors else None,
                  reference_endpoint_error_m_conditional_on_prediction=float(np.mean(reference_errors)) if reference_errors else None,
                  cpu_latency_ms=dict(median=float(np.median(latencies)*1000), p95=float(np.quantile(latencies, .95)*1000),
                                      first=float(latencies[0]*1000), includes='RGB/depth loading, camera backprojection, color segmentation, component scoring and endpoint; no Qwen/scoring/robot execution'),
                  language_swap=dict(directed_pairs=len(swap_rows), changed_instruction_semantic_accuracy=float(np.mean([row['changed_instruction_semantic_accuracy'] for row in swap_rows])),
                                     stale_instruction_semantic_accuracy=float(np.mean([row['stale_instruction_semantic_accuracy'] for row in swap_rows])), per_pair=swap_rows),
                  manifest_sha256={name:digest(args.data/name) for name in ('observations.jsonl', 'supervision.jsonl')},
                  model=model, evaluation_source_sha256=evaluation_hashes, per_scene=output_rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('model', 'per_scene', 'evaluation_source_sha256', 'language_swap')}), flush=True)


if __name__ == '__main__':
    main()
