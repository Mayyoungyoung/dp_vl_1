"""Versioned virtual endpoint attachment; v1 grid, thresholds and failures remain.

Connectors use exact full-segment distance to finite nonselected visible points
and a fixed 3.5mm ray/depth/contact sample spacing. Neither is a hidden-solid or
robot certificate. Exactly four bounded graph searches are emitted without
post-generation replacement, repair or filtering.
"""
import argparse
from collections import Counter
import heapq
import itertools
import json
from pathlib import Path
import time

import numpy as np

from scripts import observation_multiroute_astar as v1
from scripts import observation_prototype_grounding as prototype


CONFIG = dict(v1.CONFIG, version='v2_virtual_exact_endpoints', attachment_neighbor_extent_voxels=1,
    attachment_ray_step_m=.0025,
    attachment_scope='original neighboring free cells within unchanged start/target contact radii; target component unchanged',
    connector_collision_scope='complete segment distance to finite nonselected visible points >= original 20mm plus fixed-step observed-ray/contact checks; not continuous solid/visibility certification')


def segment_point_distance(first, second, points):
    if not len(points):
        return None
    delta = second-first
    squared = float(delta @ delta)
    parameter = np.clip(((points-first) @ delta)/squared, 0., 1.) if squared > 0 else np.zeros(len(points))
    return float(np.linalg.norm(points-(first+parameter[:, None]*delta), axis=-1).min())


def sample_polyline(path):
    samples = []
    for first, second in zip(path, path[1:]):
        segments = max(1, int(np.ceil(np.linalg.norm(second-first)/CONFIG['attachment_ray_step_m'])))
        samples.append(first+np.linspace(0, 1, segments+1)[:, None]*(second-first))
    return np.concatenate(samples) if samples else path.copy()


def check_rays(samples, depth, valid, component, intrinsics, camera_to_world, current, endpoint, target_radius):
    camera = (samples-camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    z = camera[:, 2]
    projected = camera @ intrinsics.T
    pixels = np.rint(projected[:, :2]/np.where(z > 0, z, 1.)[:, None]).astype(int)
    height, width = depth.shape
    in_view = (z > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width) & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    u, vv = np.clip(pixels[:, 0], 0, width-1), np.clip(pixels[:, 1], 0, height-1)
    observed = in_view & valid[vv, u]
    measured = depth[vv, u]
    ordinary = observed & (z < measured-CONFIG['visible_surface_clearance_m'])
    start_allowed = np.linalg.norm(samples-current[:3], axis=-1) <= CONFIG['current_tip_contact_radius_m']
    target_allowed = (observed & component[vv, u] & (np.abs(z-measured) <= CONFIG['target_contact_depth_m']) &
                      (np.linalg.norm(samples-endpoint, axis=-1) <= target_radius))
    allowed = ordinary | start_allowed | target_allowed
    unknown = ~observed | (z > measured)
    return dict(passed=bool(allowed.all()), samples=len(samples), failed_samples=int((~allowed).sum()),
        ordinary_visible_free_samples=int(ordinary.sum()), original_start_permission_samples=int(start_allowed.sum()),
        original_target_permission_samples=int(target_allowed.sum()), unknown_samples=int(unknown.sum()),
        unknown_samples_admitted_only_by_original_contact=int((unknown & allowed).sum()),
        first_failed_sample_coordinates=samples[~allowed][:8].tolist(),
        step_m=CONFIG['attachment_ray_step_m'], continuous_visibility_certificate=False)


def visible_proxy(path, nonselected_points, ray_inputs):
    begin = time.perf_counter()
    distances = [segment_point_distance(first, second, nonselected_points) for first, second in zip(path, path[1:])]
    finite = [x for x in distances if x is not None]
    minimum = min(finite) if finite else None
    point_clear = minimum is None or minimum >= CONFIG['visible_surface_clearance_m']
    point_seconds = time.perf_counter()-begin
    begin_ray = time.perf_counter()
    ray = check_rays(sample_polyline(path), **ray_inputs)
    return dict(passed=bool(point_clear and ray['passed']), finite_point_cloud_segment_clear=bool(point_clear),
        complete_polyline_minimum_nonselected_observed_point_distance_m=minimum,
        observed_nontarget_points=len(nonselected_points), exact_point_distance_seconds=point_seconds,
        ray_check=ray, ray_check_seconds=time.perf_counter()-begin_ray,
        limitation='finite visible points plus fixed-step depth proxy, not continuous hidden-solid or robot certificate')


def attachments(point, name, free, lower, radius, nonselected_points, ray_inputs):
    began = time.perf_counter()
    center = v1.grid_index(point, lower, CONFIG['voxel_m'])
    accepted, records = {}, []
    for offset in itertools.product((-1, 0, 1), repeat=3):
        index = center+np.asarray(offset)
        location = lower+index*CONFIG['voxel_m']
        length = float(np.linalg.norm(location-point))
        inside = bool(np.all(index >= 0) and np.all(index < free.shape))
        record = dict(index=index.tolist(), center_xyz=location.tolist(), within_grid=inside,
            within_unchanged_contact_radius=bool(length <= radius), radius_m=float(radius), length_m=length,
            original_grid_free=bool(free[tuple(index)]) if inside else False)
        if not inside or length > radius or not free[tuple(index)]:
            record.update(accepted=False, rejection='outside_original_local_free_attachment_set')
            records.append(record)
            continue
        connector = np.stack([point, location] if name == 'start' else [location, point])
        checks = visible_proxy(connector, nonselected_points, ray_inputs)
        record.update(accepted=checks['passed'], checks=checks,
            rejection=None if checks['passed'] else 'exact_visible_point_distance_or_fixed_ray_contact_check_failed')
        if checks['passed']:
            accepted[tuple(index)] = dict(length_m=length, connector=connector, record_index=len(records))
        records.append(record)
    return accepted, dict(kind=name, exact_endpoint=point.tolist(), rounded_index=center.tolist(),
        original_contact_radius_m=float(radius), neighboring_cells_considered=len(records), accepted_connections=len(accepted),
        accepted_voxel_indices=[[int(value) for value in index] for index in accepted], candidates=records,
        attachment_seconds=time.perf_counter()-began, complete_route_proposals_emitted=0)


def astar_virtual(free, lower, exact_goal, starts, goals, used_edges, start_permission, target_permission):
    began = time.perf_counter()
    statistics = dict(expanded_nodes=0, neighbor_edges_examined=0, free_edges_examined=0,
        diagonal_supercover_voxels_examined=0, virtual_start_edges=len(starts), virtual_goal_edges=len(goals),
        virtual_goal_edges_examined=0, permission_start_edge_checks=0, permission_target_edge_checks=0,
        maximum_expanded_nodes=CONFIG['maximum_expanded_nodes_per_candidate'], search_deadline_seconds=CONFIG['search_deadline_seconds'])
    def finish(path, status):
        statistics.update(status=status, search_seconds=time.perf_counter()-began, complete_raw_paths_emitted=int(path is not None))
        return path, statistics
    if not starts:
        return finish(None, 'no_admissible_virtual_start_attachment')
    if not goals:
        return finish(None, 'no_admissible_virtual_goal_attachment')
    start_virtual, goal_virtual = (-2, -2, -2), (-1, -1, -1)
    queue, scores, previous, closed = [], {}, {}, set()
    counter = itertools.count()
    for index, attachment in starts.items():
        cost = attachment['length_m']*(1+CONFIG['used_edge_penalty_multiplier']*used_edges[('start', index)])
        scores[index], previous[index] = cost, start_virtual
        heuristic = float(np.linalg.norm(lower+np.asarray(index)*CONFIG['voxel_m']-exact_goal))
        heapq.heappush(queue, (cost+heuristic, next(counter), index))
    while queue:
        _, _, current = heapq.heappop(queue)
        if current in closed:
            continue
        if current == goal_virtual:
            path = [previous[goal_virtual]]
            while previous[path[-1]] != start_virtual:
                path.append(previous[path[-1]])
            path.reverse()
            statistics.update(selected_start_attachment=[int(value) for value in path[0]], selected_goal_attachment=[int(value) for value in path[-1]],
                successful_route_grid_edges=len(path)-1, successful_route_virtual_edges=2,
                successful_route_edges_in_start_permission=sum(bool(start_permission[a] or start_permission[b]) for a,b in zip(path,path[1:])),
                successful_route_edges_in_target_permission=sum(bool(target_permission[a] or target_permission[b]) for a,b in zip(path,path[1:])))
            return finish(path, 'path_found')
        closed.add(current)
        statistics['expanded_nodes'] += 1
        if statistics['expanded_nodes'] >= CONFIG['maximum_expanded_nodes_per_candidate']:
            return finish(None, 'expanded_node_budget_exhausted')
        if statistics['expanded_nodes'] % 64 == 0 and time.perf_counter()-began >= CONFIG['search_deadline_seconds']:
            return finish(None, 'search_time_budget_exhausted')
        if current in goals:
            statistics['virtual_goal_edges_examined'] += 1
            cost = goals[current]['length_m']*(1+CONFIG['used_edge_penalty_multiplier']*used_edges[('goal', current)])
            proposed = scores[current]+cost
            if proposed < scores.get(goal_virtual, np.inf):
                scores[goal_virtual], previous[goal_virtual] = proposed, current
                heapq.heappush(queue, (proposed, next(counter), goal_virtual))
        for delta, touched, length in v1.NEIGHBORS:
            statistics['neighbor_edges_examined'] += 1
            neighbor = tuple(current[axis]+delta[axis] for axis in range(3))
            if any(neighbor[axis] < 0 or neighbor[axis] >= free.shape[axis] for axis in range(3)):
                continue
            blocked = False
            for offset in touched:
                statistics['diagonal_supercover_voxels_examined'] += 1
                cell = tuple(current[axis]+offset[axis] for axis in range(3))
                if not free[cell]:
                    blocked = True
                    break
            if blocked:
                continue
            statistics['free_edges_examined'] += 1
            statistics['permission_start_edge_checks'] += int(start_permission[current] or start_permission[neighbor])
            statistics['permission_target_edge_checks'] += int(target_permission[current] or target_permission[neighbor])
            edge = ('grid',)+tuple(sorted((current, neighbor)))
            cost = length*CONFIG['voxel_m']*(1+CONFIG['used_edge_penalty_multiplier']*used_edges[edge])
            proposed = scores[current]+cost
            if proposed < scores.get(neighbor, np.inf):
                scores[neighbor], previous[neighbor] = proposed, current
                heuristic = float(np.linalg.norm(lower+np.asarray(neighbor)*CONFIG['voxel_m']-exact_goal))
                heapq.heappush(queue, (proposed+heuristic, next(counter), neighbor))
    return finish(None, 'open_set_exhausted')


def plan_observation(rgb, xyz, valid, depth, intrinsics, camera_to_world, current, instruction, model, prior):
    began = time.perf_counter()
    endpoint, localization = prototype.predict(rgb, xyz, valid, instruction, model)
    localization_seconds = time.perf_counter()-began
    paths = np.full((CONFIG['candidates'], CONFIG['horizon'], 3), np.nan)
    if endpoint is None:
        return paths, [None]*CONFIG['candidates'], dict(localization=localization, localization_seconds=localization_seconds,
            attempts=[dict(slot=i,status='localization_failure',complete_raw_paths_emitted=0) for i in range(CONFIG['candidates'])],
            failed_slots=CONFIG['candidates'], raw_complete_paths=0, submitted_candidate_budget=CONFIG['candidates'],
            request_generation_seconds=time.perf_counter()-began, filtering_or_repair=False)
    try:
        free, lower, start_permission, target_permission, grid = v1.build_grid(rgb,xyz,valid,depth,intrinsics,
            camera_to_world,current,endpoint,instruction,model,localization,prior)
    except ValueError as error:
        return paths, [None]*CONFIG['candidates'], dict(localization=localization,localization_seconds=localization_seconds,
            attempts=[dict(slot=i,status='grid_construction_rejected',error=str(error),complete_raw_paths_emitted=0) for i in range(CONFIG['candidates'])],
            failed_slots=CONFIG['candidates'],raw_complete_paths=0,submitted_candidate_budget=CONFIG['candidates'],request_generation_seconds=time.perf_counter()-began)
    component = v1.selected_target_component(rgb,valid,instruction,model,localization)
    nonselected_points = xyz[valid & ~component]
    target_radius = grid['contact_allowances']['target_radius_m']
    ray_inputs = dict(depth=depth,valid=valid,component=component,intrinsics=intrinsics,camera_to_world=camera_to_world,
        current=current,endpoint=endpoint,target_radius=target_radius)
    starts, start_audit = attachments(current[:3],'start',free,lower,CONFIG['current_tip_contact_radius_m'],nonselected_points,ray_inputs)
    goals, goal_audit = attachments(endpoint,'goal',free,lower,target_radius,nonselected_points,ray_inputs)
    used_edges, earlier, raw_paths, attempts = Counter(), set(), [], []
    for slot in range(CONFIG['candidates']):
        cells, record = astar_virtual(free,lower,endpoint,starts,goals,used_edges,start_permission,target_permission)
        record['slot'] = slot
        if cells is None:
            raw_paths.append(None);attempts.append(record)
            continue
        emission = time.perf_counter()
        raw = np.vstack([current[:3],lower+np.asarray(cells)*CONFIG['voxel_m'],endpoint])
        sampled = v1.resample(raw,CONFIG['horizon'])
        record['path_materialization_resampling_seconds'] = time.perf_counter()-emission
        raw_grid, raw_grid_count = v1.path_observed_clear(raw,free,lower)
        h24_grid, h24_grid_count = v1.path_observed_clear(sampled,free,lower)
        raw_proxy = visible_proxy(raw,nonselected_points,ray_inputs)
        sampled_proxy = visible_proxy(sampled,nonselected_points,ray_inputs)
        signature = tuple(cells)
        record.update(raw_point_count=len(raw),raw_original_grid_proxy_clear=bool(raw_grid),h24_original_grid_proxy_clear=bool(h24_grid),
            raw_original_grid_voxels_checked=raw_grid_count,h24_original_grid_voxels_checked=h24_grid_count,
            raw_visible_point_ray_proxy=raw_proxy,h24_visible_point_ray_proxy=sampled_proxy,
            resampling_changed_visible_point_ray_proxy=bool(raw_proxy['passed'] != sampled_proxy['passed']),
            exact_grid_path_duplicate=signature in earlier,raw_length_m=float(np.linalg.norm(np.diff(raw,axis=0),axis=-1).sum()),
            h24_length_m=float(np.linalg.norm(np.diff(sampled,axis=0),axis=-1).sum()),post_generation_checks_change_output=False)
        paths[slot]=sampled;raw_paths.append(raw);attempts.append(record);earlier.add(signature)
        used_edges[('start',cells[0])]+=1;used_edges[('goal',cells[-1])]+=1
        for first,second in zip(cells,cells[1:]):
            used_edges[('grid',)+tuple(sorted((first,second)))]+=1
    return paths,raw_paths,dict(version=CONFIG['version'],localization=localization,localization_seconds=localization_seconds,
        grid=grid,virtual_start_attachments=start_audit,virtual_goal_attachments=goal_audit,attempts=attempts,
        submitted_candidate_budget=CONFIG['candidates'],raw_complete_paths=sum(path is not None for path in raw_paths),
        failed_slots=sum(path is None for path in raw_paths),request_generation_seconds=time.perf_counter()-began,
        filtering_or_repair=False,original_mask_grid_clearance_and_contact_radii_unchanged=True)


def self_test():
    # Endpoints are far from a point obstacle, but the complete segment crosses it.
    assert segment_point_distance(np.array([-1.,0,0]),np.array([1.,0,0]),np.array([[0.,0,0]]))==0.
    assert segment_point_distance(np.array([-1.,.03,0]),np.array([1.,.03,0]),np.array([[0.,0,0]]))==.03
    depth=np.ones((5,5));valid=np.ones((5,5),bool);component=np.zeros((5,5),bool)
    intrinsics=np.array([[10.,0,2],[0,10.,2],[0,0,1]])
    current=np.array([0.,0,.925,0,0,0,1,1]);goal=np.array([0.,0,1.])
    rays=dict(depth=depth,valid=valid,component=component,intrinsics=intrinsics,camera_to_world=np.eye(4),current=current,endpoint=goal,target_radius=.06)
    assert not check_rays(np.array([[0.,0,1.02]]),**rays)['passed']
    component[:]=True
    assert check_rays(np.array([[0.,0,1.02]]),**rays)['passed']
    assert not check_rays(np.array([[0.,0,1.04]]),**rays)['passed']
    lower=np.array([-.05,-.05,.9]);free=np.ones((5,5,5),bool);free[:,:,4]=False
    nonselected=np.array([[2.,0,1.]])
    starts,sa=attachments(current[:3],'start',free,lower,.04,nonselected,rays)
    goals,ga=attachments(goal,'goal',free,lower,.06,nonselected,rays)
    assert len(starts) and len(goals) and not free[tuple(v1.grid_index(goal,lower,CONFIG['voxel_m']))]
    path,record=astar_virtual(free,lower,goal,starts,goals,Counter(),np.zeros_like(free),np.zeros_like(free))
    assert path is not None and record['complete_raw_paths_emitted']==1 and record['successful_route_virtual_edges']==2
    raw=np.vstack([current[:3],lower+np.asarray(path)*CONFIG['voxel_m'],goal])
    proxy=visible_proxy(raw,nonselected,rays)
    assert proxy['passed']
    # Actual grid indices originate in NumPy. Every successful nested record
    # must serialize without a permissive fallback that hides nonfinite values.
    json.dumps(dict(start=sa,goal=ga,search=record,proxy=proxy),allow_nan=False)
    missing,record=astar_virtual(free,lower,goal,starts,{},Counter(),np.zeros_like(free),np.zeros_like(free))
    assert missing is None and record['status']=='no_admissible_virtual_goal_attachment'
    # Goal connectors have costs: the first reached attachment need not be the
    # cheapest virtual goal, and used endpoint edges participate in penalties.
    corridor=np.ones((4,1,1),bool);zero=np.zeros_like(corridor)
    simple_start={(0,0,0):dict(length_m=0.)}
    simple_goals={(1,0,0):dict(length_m=.05),(3,0,0):dict(length_m=0.)}
    end=np.array([.075,0.,0.])
    path,_=astar_virtual(corridor,np.zeros(3),end,simple_start,simple_goals,Counter({('goal',(1,0,0)):1}),zero,zero)
    assert path[-1]==(3,0,0)
    failed,raw_failed,record=plan_observation(None,None,None,None,None,None,current,'unseen',{'prototypes':{}},None)
    assert failed.shape==(4,24,3) and np.isnan(failed).all() and len(raw_failed)==len(record['attempts'])==4
    print('v2 self-test passed: complete connector distance, blocked unknown, unchanged target depth shell, virtual goal attachment, failed attachment and K4 NaN preservation')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--train-benchmark-only',action='store_true');parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        self_test()
    elif args.data is None or args.output is None:
        parser.error('--data and --output required')
    else:
        v1.run(args.data,args.output,args.train_benchmark_only,plan_function=plan_observation,planner_config=CONFIG,
            baseline_name='TRAIN prototype + observed grid A* v2 virtual exact endpoint attachment',generation_script=__file__)


if __name__=='__main__':
    main()
