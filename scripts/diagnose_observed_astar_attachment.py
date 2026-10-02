"""Read-only first-TRAIN diagnosis of the frozen observed A* endpoint rejection."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial import cKDTree

from scripts import observation_multiroute_astar as planner
from scripts import observation_prototype_grounding as prototype


def camera_record(point, depth, valid, component, intrinsics, camera_to_world):
    camera = (np.asarray(point)-camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    projection = camera @ intrinsics.T
    if camera[2] <= 0:
        return dict(in_view=False, camera_z=float(camera[2]))
    pixel_float = projection[:2]/camera[2]
    pixel = np.rint(pixel_float).astype(int)
    u, v = pixel
    inside = bool(0 <= u < depth.shape[1] and 0 <= v < depth.shape[0])
    item = dict(in_view=inside, pixel_float=pixel_float.tolist(), pixel=pixel.tolist(), camera_z=float(camera[2]))
    if inside:
        delta = float(depth[v, u]-camera[2])
        item.update(valid_depth=bool(valid[v, u]), measured_depth=float(depth[v, u]),
            measured_surface_minus_point_depth_m=delta, predicted_target_component=bool(component[v, u]),
            visibly_free_by_original_depth_rule=bool(valid[v, u] and delta > planner.CONFIG['visible_surface_clearance_m']),
            behind_measured_surface=bool(valid[v, u] and delta < 0))
    return item


def exact_visible_point_segment_clearance(first, second, points):
    """Complete segment-to-finite-observed-point minimum; no hidden-solid claim."""
    delta = second-first
    norm = float(delta @ delta)
    parameter = np.clip(((points-first) @ delta)/norm, 0., 1.) if norm > 0 else np.zeros(len(points))
    distance = np.linalg.norm(points-(first+parameter[:, None]*delta), axis=-1)
    index = int(distance.argmin())
    return float(distance[index]), points[index].tolist()


def diagnose(data, probe, output):
    if output.exists():
        raise FileExistsError('preserve earlier diagnostics: '+str(output))
    original = json.loads((probe/'report.json').read_text())
    if not original['subset_training_cost_diagnostic'] or original['split'] != 'TRAIN' or original['examples'] != 1:
        raise ValueError('requires the predeclared first-TRAIN probe, not DEV analysis')
    rows = prototype.read_rows(data/'observations.jsonl')
    first = sorted([row for row in rows if row['split'] == 'TRAIN'], key=lambda row: row['id'])[0]
    if original['records'][0]['id'] != first['id']:
        raise ValueError('probe was not the predeclared first TRAIN observation')
    labels = {row['id']: row for row in prototype.read_rows(data/'supervision.jsonl')}
    # Only the current observation pointer is accessed; no semantic target,
    # reference route, passage type, or verification geometry is inspected.
    observation_file = data/labels[first['id']]['observation']
    (rgb, xyz, valid), files = prototype.load_observation(data, first, labels[first['id']]['observation'])
    with np.load(observation_file, allow_pickle=False) as archive:
        current = np.r_[archive['gripper_pose'], np.asarray(archive['gripper_open']).reshape(1)]
        depth, intrinsics, camera_to_world = archive['depth'], archive['camera_intrinsics'], archive['camera_extrinsics']
    model, prior = original['prototype_model'], original['train_workspace_prior']
    endpoint, localization = prototype.predict(rgb, xyz, valid, first['instruction'], model)
    if endpoint is None:
        raise ValueError('the original first-TRAIN probe had a predicted endpoint')
    component = planner.selected_target_component(rgb, valid, first['instruction'], model, localization)
    free, lower, start_permission, target_permission, grid = planner.build_grid(rgb, xyz, valid, depth,
        intrinsics, camera_to_world, current, endpoint, first['instruction'], model, localization, prior)
    if grid != original['records'][0]['grid']:
        # Timing is naturally different; all geometric and permission fields
        # must match the already-emitted v1 record exactly.
        ignored = {'grid_construction_seconds'}
        if {k:v for k,v in grid.items() if k not in ignored} != {k:v for k,v in original['records'][0]['grid'].items() if k not in ignored}:
            raise RuntimeError('reconstructed grid differs from the frozen original probe')
    voxel = planner.CONFIG['voxel_m']
    other_pixels = np.argwhere(valid & ~component)
    other_points = xyz[valid & ~component]
    point_tree = cKDTree(other_points)
    occupied_indices = planner.grid_index(other_points, lower, voxel)
    inside = np.all((occupied_indices >= 0) & (occupied_indices < np.asarray(free.shape)), axis=-1)
    occupied_centers = lower+np.unique(occupied_indices[inside], axis=0)*voxel
    occupied_tree = cKDTree(occupied_centers)
    audits, reasons = {}, {}
    for name, point in [('current_tip', current[:3]), ('predicted_endpoint', endpoint)]:
        index = planner.grid_index(point, lower, voxel)
        cell_inside = bool(np.all(index >= 0) and np.all(index < free.shape))
        center = lower+index*voxel
        raw_distance, raw_index = point_tree.query(point)
        center_distance, center_raw_index = point_tree.query(center)
        occupied_distance, occupied_id = occupied_tree.query(center)
        exact = camera_record(point, depth, valid, component, intrinsics, camera_to_world)
        projected_center = camera_record(center, depth, valid, component, intrinsics, camera_to_world)
        item = dict(point=point.tolist(), rounded_grid_index=index.tolist(), rounded_grid_center=center.tolist(),
            point_to_grid_center_m=float(np.linalg.norm(point-center)), workspace_contains_cell=cell_inside,
            original_grid_traversable=bool(free[tuple(index)]) if cell_inside else False,
            start_permission=bool(start_permission[tuple(index)]) if cell_inside else False,
            predicted_target_permission=bool(target_permission[tuple(index)]) if cell_inside else False,
            exact_point_projection=exact, cell_center_projection=projected_center,
            exact_point_nearest_nonselected_surface_distance_m=float(raw_distance),
            exact_point_nearest_nonselected_surface_xyz=other_points[raw_index].tolist(),
            exact_point_nearest_nonselected_surface_pixel_vu=other_pixels[raw_index].tolist(),
            cell_center_nearest_nonselected_surface_distance_m=float(center_distance),
            cell_center_nearest_nonselected_surface_xyz=other_points[center_raw_index].tolist(),
            nearest_occupied_voxel_center_distance_m=float(occupied_distance),
            nearest_occupied_voxel_center=occupied_centers[occupied_id].tolist(),
            rejected_by_original_voxel_inflation=bool(occupied_distance <= grid['conservative_surface_inflation_m']))
        audits[name] = item
        reasons[name] = dict(outside_workspace=int(not cell_inside),
            not_in_view=int(not projected_center.get('in_view', False)),
            invalid_depth=int(projected_center.get('in_view', False) and not projected_center.get('valid_depth', False)),
            center_not_in_front_free=int(not projected_center.get('visibly_free_by_original_depth_rule', False)),
            center_not_predicted_target_component=int(not projected_center.get('predicted_target_component', False)),
            other_visible_surface_inflation=int(item['rejected_by_original_voxel_inflation']),
            original_not_traversable=int(not item['original_grid_traversable']))
    goal_index = planner.grid_index(endpoint, lower, voxel)
    neighbors = []
    for delta in np.ndindex(3, 3, 3):
        offset = np.asarray(delta)-1
        index = goal_index+offset
        if not np.all(index >= 0) or not np.all(index < free.shape):
            continue
        point = lower+index*voxel
        clear, checked = planner.path_observed_clear(np.stack([endpoint, point]), free, lower)
        distance, nearest = exact_visible_point_segment_clearance(endpoint, point, other_points)
        neighbors.append(dict(index=index.tolist(), grid_traversable=bool(free[tuple(index)]),
            distance_from_endpoint_m=float(np.linalg.norm(point-endpoint)),
            inside_unchanged_target_radius=bool(np.linalg.norm(point-endpoint) <= grid['contact_allowances']['target_radius_m']),
            original_grid_complete_connector_clear=bool(clear), connector_voxels_checked=checked,
            exact_complete_connector_to_nonselected_observed_points_distance_m=distance,
            nearest_nonselected_observed_point=nearest))
    output.mkdir(parents=True)
    result = dict(scope='read-only diagnosis of predeclared first TRAIN sample; no planning/output/threshold change',
        id=first['id'], split='TRAIN', config=planner.CONFIG, localization=localization,
        original_probe_report_sha256=prototype.digest(probe/'report.json'),
        original_grid_geometry_and_permissions_reproduced_exactly=True,
        no_dev_labels_or_geometry_read=True, no_model_or_planner_update=True,
        endpoint_audits=audits, overlapping_reason_indicators=reasons, adjacent_grid_attachment_audit=neighbors,
        traversable_local_neighbors=sum(item['grid_traversable'] for item in neighbors),
        original_grid_clear_connectors_to_free_neighbors=sum(item['grid_traversable'] and item['original_grid_complete_connector_clear'] for item in neighbors),
        point_cloud_clearance_limitation='Exact segment minimum is only against finite nonselected visible points, not solids or unknown-space visibility; it is not an executable connector certificate.',
        grid_metadata=grid, source_hashes={str(file):prototype.digest(file) for file in files},
        script_sha256=prototype.digest(__file__), planner_sha256=prototype.digest(planner.__file__))
    (output/'diagnostic.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    np.savez_compressed(output/'observed_permission_audit.npz', rgb=rgb, predicted_component=component,
        world_xyz=xyz, valid=valid, current=current, predicted_endpoint=endpoint,
        current_permission_indices=np.argwhere(start_permission), target_permission_indices=np.argwhere(target_permission),
        lower=lower, grid_shape=free.shape, local_indices=np.array([x['index'] for x in neighbors]),
        local_traversable=np.array([x['grid_traversable'] for x in neighbors]))
    draw(result, rgb, component, output)
    return result


def draw(result, rgb, component, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(1, 2, figsize=(12, 6))
    overlay = rgb.copy()
    overlay[component] = .5*overlay[component]+.5*np.array([0., 1., 0.])
    for axis in axes:
        axis.imshow(overlay)
        endpoint = result['endpoint_audits']['predicted_endpoint']
        for key, label, color in [('exact_point_projection', 'predicted observed endpoint', '#ee00dd'),
                                  ('cell_center_projection', 'rounded voxel center', '#dd3300')]:
            point = endpoint[key]
            if point.get('in_view'):
                u, v = point['pixel_float']
                axis.scatter(u, v, c=color, s=55, marker='x', linewidths=2, label=label)
        v, u = endpoint['exact_point_nearest_nonselected_surface_pixel_vu']
        axis.scatter(u, v, c='#ffdd00', s=50, marker='+', label='nearest nonselected visible point')
        axis.legend(fontsize=7)
    pixels = np.argwhere(component)
    y0, x0 = pixels.min(0)-10; y1, x1 = pixels.max(0)+10
    axes[1].set_xlim(x0, x1); axes[1].set_ylim(y1, y0)
    axes[0].set_title('Current RGB; green = predicted component only')
    axes[1].set_title('Endpoint versus rounded grid center (TRAIN only)')
    figure.suptitle(result['id']+' — observation and fixed contact-grid audit; no true goal/box input')
    figure.tight_layout()
    figure.savefig(output/'training_endpoint_attachment.png', dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True, type=Path)
    parser.add_argument('--probe', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    result = diagnose(args.data, args.probe, args.output)
    print(json.dumps(dict(id=result['id'],reasons=result['overlapping_reason_indicators'],
        local_free=result['traversable_local_neighbors'], complete_grid_connectors=result['original_grid_clear_connectors_to_free_neighbors'],
        seconds=time.perf_counter()-started)), flush=True)


if __name__ == '__main__':
    main()
