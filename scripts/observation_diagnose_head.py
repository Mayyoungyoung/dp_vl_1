"""CPU-only diagnostics for the frozen observation head and RGB-D labels."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
from routeset.observed_route_head import ObservedRouteHead, load_observed_dataset
from scripts.train_observed_routes import evaluate


def world_points(depth, intrinsics, extrinsics):
    vv, uu = np.indices(depth.shape)
    pixels = np.stack([uu, vv, np.ones_like(uu)], -1)
    camera = (pixels @ np.linalg.inv(intrinsics).T) * depth[..., None]
    return camera @ extrinsics[:3, :3].T + extrinsics[:3, 3]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    config = json.loads((a.run / 'config.json').read_text())
    data = load_observed_dataset(config['observations'], config['supervision'], config['cache_dir'], config['horizon'], config['pooling'])
    a.output.mkdir(parents=True, exist_ok=True)
    train_ids = np.flatnonzero(data['splits'] == 'TRAIN')
    dev_ids = np.flatnonzero(data['splits'] == 'DEV_MODEL')
    model = ObservedRouteHead(config['feature_dim'], config['horizon'], config['candidates'], config['width'], config['depth'])
    results = {}
    for checkpoint_name in ['best', 'last']:
        checkpoint = torch.load(a.run / (checkpoint_name + '.pt'), map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        results[checkpoint_name] = {'step': checkpoint['step']}
        for split, ids in [('TRAIN', train_ids), ('DEV_MODEL', dev_ids)]:
            output = a.output / checkpoint_name / split
            metrics = evaluate(model, data, ids, 'cpu', output)
            with np.load(output / 'predictions.npz') as prediction:
                endpoints = prediction['paths'][:, :, -1]
            nearest_correct = []
            target_errors = []
            for endpoint, idx in zip(endpoints, ids):
                specification = data['semantic_targets'][idx]
                centers = np.asarray(specification['centers'])
                distances = np.linalg.norm(endpoint[:, None]-centers[None], axis=-1)
                nearest_correct.extend((distances.argmin(1) == specification['target_index']).tolist())
                target_errors.extend(distances[:, specification['target_index']].tolist())
            metrics['diagnostic_nearest_target_identity_only'] = float(np.mean(nearest_correct))
            metrics['diagnostic_goal_error_median_m'] = float(np.median(target_errors))
            metrics['diagnostic_candidate_endpoint_spread_m'] = float(np.linalg.norm(endpoints-endpoints.mean(1, keepdims=True), axis=-1).mean())
            results[checkpoint_name][split] = metrics
    records = [json.loads(x) for x in Path(config['supervision']).read_text().splitlines()]
    obs_rows = {x['id']: x for x in [json.loads(line) for line in Path(config['observations']).read_text().splitlines()]}
    base = Path(config['supervision']).parent
    visibility = []
    for row in records:
        with np.load(base / row['observation']) as current:
            depth, K, E = current['depth'], current['camera_intrinsics'], current['camera_extrinsics']
            world = world_points(depth, K, E)
        centers = np.asarray(row['semantic_targets']['centers'])
        goal = centers[row['semantic_targets']['target_index']]
        camera = (goal-E[:3, 3]) @ E[:3, :3]
        projected = K @ camera
        uv = projected[:2] / projected[2]
        row_diag = {'id': row['id'], 'split': row['split'], 'target_pixel_uv': uv.tolist(),
                    'target_camera_z': float(camera[2]),
                    'nearest_observed_surface_to_target_m': float(cKDTree(world.reshape(-1, 3)).query(goal)[0])}
        in_frame = camera[2] > 0 and 0 <= uv[0] < depth.shape[1] and 0 <= uv[1] < depth.shape[0]
        row_diag['in_frame'] = bool(in_frame)
        if in_frame:
            x, y = np.rint(uv).astype(int)
            x, y = np.clip(x, 0, depth.shape[1]-1), np.clip(y, 0, depth.shape[0]-1)
            row_diag['center_pixel_depth_residual_m'] = float(depth[y, x]-camera[2])
            row_diag['center_pixel_rgb'] = np.asarray(Image.open(base / obs_rows[row['id']]['image']))[y, x].tolist()
            row_diag['center_ray_observed_world_to_target_m'] = float(np.linalg.norm(world[y, x]-goal))
        reference_error = []
        for name in row['routes']:
            with np.load(base / name) as trajectory:
                reference_error.append(float(np.linalg.norm(trajectory['gripper_pose'][-1,:3]-goal)))
        row_diag['reference_endpoint_error_max_m'] = max(reference_error) if reference_error else None
        visibility.append(row_diag)
    results['rgbd_label_audit'] = {'examples': len(visibility), 'in_frame': sum(x['in_frame'] for x in visibility),
        'nearest_surface_distance_mean_m': float(np.mean([x['nearest_observed_surface_to_target_m'] for x in visibility])),
        'nearest_surface_distance_max_m': max(x['nearest_observed_surface_to_target_m'] for x in visibility),
        'center_ray_to_target_max_m': max(x.get('center_ray_observed_world_to_target_m', 0) for x in visibility),
        'negative_focal_lengths_expected': True,
        'coordinate_convention': 'camera_to_world extrinsic and PyRep signed intrinsics; no vertical flip',
        'note': 'Evaluation labels used only for diagnosis; not a model input or changed test standard'}
    (a.output / 'visibility_per_scene.json').write_text(json.dumps(visibility, indent=2), encoding='utf-8')
    (a.output / 'diagnostic.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print(json.dumps(results, indent=2), flush=True)


if __name__ == '__main__':
    main()
