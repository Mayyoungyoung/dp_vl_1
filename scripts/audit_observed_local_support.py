"""Fixed first-eight TRAIN parents: observed support at references/collisions.

No training or model modification. True boxes only locate evaluation queries;
the support function accepts only query coordinates and current observations.
"""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree

from scripts.audit_observed_departure_regions import PARENTS, selected_rows
from scripts.observation_prototype_grounding import digest
from routeset.geometry import segment_aabb_intersection


CONFIG = dict(radii_m=[.05, .10, .20], reference_prefix_arc_fraction=.25,
    prefix_samples_per_reference=17, reference_sampling='17 equal normalized arc positions from0 to0.25 inclusive',
    vertex_fraction_definition='raw positive-record vertex index for references; H24 predicted vertex index for collision queries; not interchangeable',
    visibility_depth_tolerance_m=.02, maximum_depth_m=10., collision_clearance_m=.02,
    collision_query='exact first closed-AABB contact at original2cm expansion; label-only query selection',
    clouds=['all_valid_depth_pixels', 'exact_original_head_pixel_stride'],
    visibility='projection into current depth: front, near_surface, occluded, outside, invalid_depth, behind_camera',
    absence='no nearby visible point does not imply free space',
    scope='predeclared old32 TRAIN parents272000-272007/all24 instructions; no DEV or locked inputs')


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def prefix_samples(path):
    path = np.asarray(path, np.float64)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    total = float(cumulative[-1])
    fractions = np.linspace(0., CONFIG['reference_prefix_arc_fraction'], CONFIG['prefix_samples_per_reference'])
    samples, vertices = [], []
    for fraction in fractions:
        distance = fraction*total
        segment = min(int(np.searchsorted(cumulative, distance, side='right')-1), len(path)-2)
        offset = (distance-cumulative[segment])/lengths[segment] if lengths[segment] > 0 else 0.
        samples.append(path[segment]+offset*(path[segment+1]-path[segment]))
        vertices.append((segment+offset)/(len(path)-1))
    return np.asarray(samples), fractions, np.asarray(vertices), total


def first_contact(path, centers, halfsizes):
    """Exact slab entry, checked against the shared closed-AABB predicate."""
    path = np.asarray(path, np.float64)
    lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    cumulative = np.r_[0., np.cumsum(lengths)]
    total = float(cumulative[-1])
    for segment, (first, second) in enumerate(zip(path, path[1:])):
        contacts = []
        for obstacle, (center, halfsize) in enumerate(zip(centers, halfsizes)):
            lower, upper = center-halfsize-CONFIG['collision_clearance_m'], center+halfsize+CONFIG['collision_clearance_m']
            if not bool(segment_aabb_intersection(first, second, lower, upper)):
                continue
            direction = second-first
            moving = np.abs(direction) > 1e-12
            low, high = np.full(3, -np.inf), np.full(3, np.inf)
            for axis in np.flatnonzero(moving):
                pair = [(lower[axis]-first[axis])/direction[axis], (upper[axis]-first[axis])/direction[axis]]
                low[axis], high[axis] = min(pair), max(pair)
            fraction = float(np.clip(max(0., low.max()), 0., 1.))
            contacts.append((fraction, obstacle, first+fraction*direction))
        if contacts:
            fraction, obstacle, point = min(contacts, key=lambda item:(item[0], item[1]))
            return dict(xyz=point.tolist(), segment_index=segment, segment_fraction=fraction, obstacle_index=obstacle,
                normalized_vertex_fraction=(segment+fraction)/(len(path)-1),
                normalized_arc_fraction=float((cumulative[segment]+fraction*lengths[segment])/total) if total else 0.,
                arc_before_segment=float(cumulative[segment]/total) if total else 0.,
                path_length_m=total, distance_from_current_m=float(np.linalg.norm(point-path[0])))
    return None


def visibility(queries, depth, intrinsics, camera_to_world):
    camera = (queries-camera_to_world[:3, 3]) @ camera_to_world[:3, :3]
    projected = camera @ intrinsics.T
    z = camera[:, 2]
    uv = np.rint(projected[:, :2]/np.where(z > 0, z, 1.)[:, None]).astype(int)
    height, width = depth.shape
    output = []
    for point, pixel, optical_depth in zip(queries, uv, z):
        u, v = pixel
        if optical_depth <= 0:
            output.append(dict(category='behind_camera', optical_depth_m=float(optical_depth)))
        elif not (0 <= u < width and 0 <= v < height):
            output.append(dict(category='outside_image', pixel_xy=pixel.tolist(), optical_depth_m=float(optical_depth)))
        else:
            measured = float(depth[v, u])
            if not np.isfinite(measured) or not 0 < measured <= CONFIG['maximum_depth_m']:
                output.append(dict(category='invalid_depth', pixel_xy=pixel.tolist(), optical_depth_m=float(optical_depth)))
                continue
            delta = float(optical_depth-measured)
            category = ('occluded' if delta > CONFIG['visibility_depth_tolerance_m'] else
                        'in_front_of_measured_surface' if delta < -CONFIG['visibility_depth_tolerance_m'] else 'near_visible_surface')
            output.append(dict(category=category, pixel_xy=pixel.tolist(), optical_depth_m=float(optical_depth),
                measured_depth_m=measured, optical_depth_minus_measurement_m=delta))
    return output


def support(queries, visible_points):
    """No labels, obstacle boxes, target coordinates or route types allowed."""
    if not len(visible_points):
        return dict(nearest_distance_m=[None]*len(queries), counts={str(radius):[0]*len(queries) for radius in CONFIG['radii_m']})
    tree = cKDTree(visible_points)
    distances, _ = tree.query(queries, k=1)
    return dict(nearest_distance_m=distances.tolist(),
        counts={str(radius):[len(indices) for indices in tree.query_ball_point(queries, radius)] for radius in CONFIG['radii_m']})


def distribution(values):
    if not len(values):
        return None
    return dict(min=float(np.min(values)), p10=float(np.quantile(values, .1)), median=float(np.median(values)),
        p90=float(np.quantile(values, .9)), max=float(np.max(values)))


def aggregate(rows):
    output = dict(queries=len(rows), observations=len({r['scene_id'] for r in rows}),
        parents=len({r['parent_id'] for r in rows}), visibility=dict(Counter(r['visibility']['category'] for r in rows)), clouds={})
    for cloud in CONFIG['clouds']:
        values = [r['support'][cloud] for r in rows]
        distances = [r['nearest_distance_m'] for r in values if r['nearest_distance_m'] is not None]
        output['clouds'][cloud] = dict(nearest_distance_m=distribution(distances), radii={})
        for radius in map(str, CONFIG['radii_m']):
            counts = [r['counts'][radius] for r in values]
            output['clouds'][cloud]['radii'][radius] = dict(point_count=distribution(counts),
                supported_queries=sum(count > 0 for count in counts), unsupported_queries=sum(count == 0 for count in counts),
                coverage=float(np.mean(np.asarray(counts) > 0)) if counts else None)
    return output


def run(data, output, diagnostic_path, diagnostic_hash):
    if output.exists():
        raise FileExistsError('fresh output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('CPU audit requires hidden CUDA')
    if digest(diagnostic_path) != diagnostic_hash:
        raise ValueError('prior TRAIN diagnostic hash mismatch')
    diagnostic = json.loads(diagnostic_path.read_text())
    original = diagnostic['prediction_results']['peak_seed0_last1000']
    provenance = original['provenance']
    config_path = Path(provenance['source_run'])/'config.json'
    if digest(config_path) != provenance['source_config_sha256']:
        raise ValueError('original head config hash mismatch')
    config = json.loads(config_path.read_text())
    if config['seed'] != 0 or config['anchor_mode'] != 'straight_through_peak' or Path(config['observations']).resolve() != (data/'observations.jsonl').resolve():
        raise ValueError('original old32 TRAIN peak seed0 required')
    prediction = Path(provenance['prediction_source'])
    if digest(prediction) != provenance['prediction_sha256']:
        raise ValueError('saved TRAIN prediction hash mismatch')
    observations = sorted(selected_rows(data/'observations.jsonl'), key=lambda row:row['id'])
    labels = {row['id']:row for row in selected_rows(data/'supervision.jsonl')}
    if len(observations) != 24 or {row['parent_id'] for row in observations} != PARENTS:
        raise ValueError('all first8 TRAIN parents/24 instructions required')
    prior_rows = {(r['scene_id'], r['candidate']):r for r in original['per_candidate']}
    with np.load(prediction, allow_pickle=False) as archive:
        ids = list(map(str, archive['scene_ids']))
        if len(ids) != len(set(ids)) or len(ids) != 96 or any(not value.startswith('obstacle_reach_2720') for value in ids):
            raise ValueError('expected unique old32 TRAIN predictions')
        paths = {row['id']:archive['paths'][ids.index(row['id'])] for row in observations}
    # Use the exact point extraction used by the head, including stride offset,
    # float32 storage, camera signs and valid mask. No model/checkpoint is loaded.
    import torch
    from routeset.observed_geometry import backproject_rgbd
    torch.set_num_threads(1)
    started = time.perf_counter()
    output.mkdir(parents=True)
    records, cloud_records, hashes = [], {}, {}
    expected_sources = diagnostic['raw_source_hashes']
    def checked(name, parent):
        path = (data/name).resolve()
        if path.parent.name != parent or parent not in PARENTS:
            raise ValueError('raw file outside selected TRAIN parent')
        actual = digest(path)
        if expected_sources.get(str(path)) != actual:
            raise ValueError('raw selected TRAIN file changed: '+str(path))
        hashes[str(path)] = actual
        return path
    for row in observations:
        identifier, parent = row['id'], row['parent_id']
        label = labels[identifier]
        if set(row) != {'id','parent_id','split','image','instruction'} or label['split'] != 'TRAIN':
            raise ValueError('strict TRAIN observation whitelist required')
        image = checked(row['image'], parent)
        observation = checked(label['observation'], parent)
        with Image.open(image) as source:
            rgb = np.asarray(source.convert('RGB'))
        with np.load(observation, allow_pickle=False) as archive:
            depth, intrinsics, camera = archive['depth'], archive['camera_intrinsics'], archive['camera_extrinsics']
            current = archive['gripper_pose'][:3]
        clouds = {}
        for name, stride in zip(CONFIG['clouds'], (1, config['pixel_stride'])):
            projected = backproject_rgbd(rgb, depth, intrinsics, camera, pixel_stride=stride)
            clouds[name] = projected['world_xyz'][projected['valid_mask']]
        metadata = dict(valid_points={name:len(value) for name,value in clouds.items()},
            pixel_stride=config['pixel_stride'], observation_sha256=hashes[str(observation)], image_sha256=hashes[str(image)])
        if parent in cloud_records and cloud_records[parent] != metadata:
            raise ValueError('same-parent observation changed across instructions')
        cloud_records[parent] = metadata
        queries = []
        for reference, name in enumerate(label['routes']):
            path = checked(name, parent)
            with np.load(path, allow_pickle=False) as archive:
                positive = archive['gripper_pose'][:, :3]
            samples, arc, vertices, total = prefix_samples(positive)
            for sample, (point, fraction, vertex) in enumerate(zip(samples, arc, vertices)):
                queries.append(dict(kind='positive_reference_prefix', reference=reference, sample=sample, xyz=point.tolist(),
                    normalized_arc_fraction=float(fraction), normalized_vertex_fraction=float(vertex), path_length_m=total,
                    distance_from_current_m=float(np.linalg.norm(point-current))))
        # Verification-only box data are used exclusively to place diagnostic
        # first-contact queries. They never enter clouds/support/visibility.
        verification = checked(label['verification_only'], parent)
        with np.load(verification, allow_pickle=False) as archive:
            centers, halfsizes = archive['obstacle_centers'], archive['obstacle_halfsizes']
        for candidate, path in enumerate(paths[identifier]):
            contact = first_contact(path, centers, halfsizes)
            prior = prior_rows[(identifier, candidate)]
            if (contact is None) != (prior['first_collision'] is None):
                raise ValueError('collision existence differs from frozen original diagnostic')
            if contact is not None:
                if contact['segment_index'] != prior['first_collision']['segment_index']:
                    raise ValueError('collision segment differs from original fixed checker')
                queries.append(dict(kind='prediction_first_contact', candidate=candidate,
                    semantic_goal_correct=prior['semantic_goal_correct'], **contact))
        xyz = np.asarray([q['xyz'] for q in queries])
        if len(queries):
            vis = visibility(xyz, depth, intrinsics, camera)
            supports = {name:support(xyz, cloud) for name,cloud in clouds.items()}
            for index, query in enumerate(queries):
                query.update(scene_id=identifier, parent_id=parent, visibility=vis[index], support={})
                for name, values in supports.items():
                    query['support'][name] = dict(nearest_distance_m=values['nearest_distance_m'][index],
                        counts={radius:counts[index] for radius,counts in values['counts'].items()})
                records.append(query)
        print(json.dumps(dict(id=identifier, query_count=len(queries), candidates=4)), flush=True)
    if sum(r['kind'] == 'positive_reference_prefix' for r in records) != 40*CONFIG['prefix_samples_per_reference']:
        raise ValueError('all40 fixed TRAIN positive references required')
    np.savez_compressed(output/'queries.npz', xyz=np.asarray([r['xyz'] for r in records]),
        kind=np.asarray([r['kind'] for r in records]), scene_id=np.asarray([r['scene_id'] for r in records]),
        reference=np.asarray([r.get('reference', -1) for r in records]), candidate=np.asarray([r.get('candidate', -1) for r in records]),
        normalized_arc_fraction=np.asarray([r['normalized_arc_fraction'] for r in records]))
    write(output/'per_query.json', records)
    groups = {}
    for kind in ('positive_reference_prefix', 'prediction_first_contact'):
        subset = [r for r in records if r['kind'] == kind]
        groups[kind] = aggregate(subset)
        groups[kind]['by_visibility'] = {name:aggregate([r for r in subset if r['visibility']['category'] == name])
            for name in sorted({r['visibility']['category'] for r in subset})}
        groups[kind]['arc_fraction'] = distribution([r['normalized_arc_fraction'] for r in subset])
        groups[kind]['vertex_fraction'] = distribution([r['normalized_vertex_fraction'] for r in subset])
        groups[kind]['physical_distance_from_current_m'] = distribution([r['distance_from_current_m'] for r in subset])
        groups[kind]['per_parent'] = {parent:aggregate([r for r in subset if r['parent_id'] == parent]) for parent in sorted(PARENTS)}
    prediction_rows = [r for r in records if r['kind'] == 'prediction_first_contact']
    groups['prediction_first_contact']['semantic_correct'] = aggregate([r for r in prediction_rows if r['semantic_goal_correct']])
    groups['prediction_first_contact']['earliest_quarter_count'] = sum(r['normalized_arc_fraction'] <= .25 for r in prediction_rows)
    report = dict(config=CONFIG, code_commit=os.environ.get('CODE_COMMIT'), total_seconds=time.perf_counter()-started,
        parents=8, instructions=24, positive_references=40, prediction_candidates=96, collision_candidates=len(prediction_rows),
        noncollision_candidates=96-len(prediction_rows), all_queries=groups, point_clouds=cloud_records,
        original_prediction=provenance, diagnostic_sha256=diagnostic_hash, source_sha256={
            'script':digest(__file__), 'point_extraction':digest(Path(__file__).resolve().parents[1]/'routeset/observed_geometry.py'),
            'selected_rows':digest(Path(__file__).with_name('audit_observed_departure_regions.py')),
            'box_predicate':digest(Path(__file__).resolve().parents[1]/'routeset/geometry.py')},
        raw_train_source_sha256=hashes, observation_manifest_sha256=digest(data/'observations.jsonl'),
        supervision_manifest_sha256=digest(data/'supervision.jsonl'),
        artifact_sha256={name:digest(output/name) for name in ('queries.npz','per_query.json')},
        limitation='Query-level statistics are correlated within route/parent; no nearby visible point is not free; projected front depth is not collision certification; scale evidence only on TRAIN')
    write(output/'summary.json', report)
    print(json.dumps(dict(collision_candidates=len(prediction_rows), total_seconds=report['total_seconds'])), flush=True)


def self_test():
    samples, arc, vertex, total = prefix_samples(np.asarray([[0.,0,0],[.1,0,0],[.1,.9,0]]))
    np.testing.assert_allclose(samples[-1], [.1,.15,0], atol=1e-12)
    assert total == 1. and arc[-1] == .25 and vertex[-1] > .5
    centers, halfsizes = np.zeros((1,3)), np.full((1,3), .1)
    contact = first_contact(np.array([[-.4,0,0],[.4,0,0]]), centers, halfsizes)
    np.testing.assert_allclose(contact['xyz'], [-.12,0,0], atol=1e-12)
    assert np.isclose(contact['normalized_arc_fraction'], .35)
    assert first_contact(np.array([[-.4,.5,0],[.4,.5,0]]), centers, halfsizes) is None
    result = support(np.asarray([[0.,0,0],[1.,0,0]]), np.asarray([[.04,0,0],[.08,0,0]]))
    assert result['counts']['0.05'] == [1,0] and result['counts']['0.1'] == [2,0]
    assert support(np.zeros((1,3)), np.empty((0,3)))['nearest_distance_m'] == [None]
    depth = np.ones((5,5)); intrinsics=np.array([[-2.,0,2],[0,-2.,2],[0,0,1]])
    rows=visibility(np.asarray([[0,0,.5],[0,0,1],[0,0,1.2],[0,0,-1],[10,0,1]]),depth,intrinsics,np.eye(4))
    assert [r['category'] for r in rows] == ['in_front_of_measured_surface','near_visible_surface','occluded','behind_camera','outside_image']
    print('support self-test passed: arc sampling, exact original-margin collision, support counts, missing support, signed-camera depth visibility')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('data','output','diagnostic'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--diagnostic-sha256')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    elif None in (args.data,args.output,args.diagnostic,args.diagnostic_sha256):
        parser.error('data/output/diagnostic/hash required')
    else:
        run(args.data,args.output,args.diagnostic,args.diagnostic_sha256)


if __name__ == '__main__':
    main()
