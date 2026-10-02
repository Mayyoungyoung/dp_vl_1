"""Saved-prediction two-row metrics, using the unchanged geometric validity rules.

The route classifier is the same pure function used to accept/label collected
references. Geometry and target coordinates are evaluation labels only. This
module neither launches a simulator nor calls a model, planner, or path repair.
"""
import numpy as np

from scripts.collect_observed_two_row_pilot import crossing_signature, geometry as registered_geometry
from scripts.evaluate_observed_obstacles import scene_metrics as base_scene_metrics

PROTOCOL = 'observed_two_row_tip_eval_v1'
TYPE_VALUES = {'negative_y', 'middle', 'positive_y', 'over'}
TYPE_FIELDS = ('row_x', 'post_y', 'post_size_xyz', 'post_base_z', 'tip_clearance_m')


def validate_labels(geometry, specification, route_config, reference_types):
    """Do not silently apply the single-obstacle type definition to two rows."""
    if route_config.get('tip_clearance_m') != .02 or route_config.get('endpoint_tolerance_m') != .03:
        raise ValueError('Original two-row 2cm clearance and 3cm endpoint thresholds required')
    if specification.get('tolerance') != .03:
        raise ValueError('Semantic threshold differs from registered collection')
    if any(name not in route_config for name in TYPE_FIELDS):
        raise ValueError('Explicit registered two-row type geometry required')
    rows = np.asarray(route_config['row_x'], dtype=float)
    columns = np.asarray(route_config['post_y'], dtype=float)
    sizes = np.asarray(route_config['post_size_xyz'], dtype=float)
    if (rows.shape != (2,) or columns.shape != (2, 2) or sizes.shape != (3,)
            or not np.isfinite(np.r_[rows, columns.ravel(), sizes, route_config['post_base_z']]).all()
            or rows[0] >= rows[1] or np.any(columns[:, 0] >= columns[:, 1]) or np.any(sizes <= 0)):
        raise ValueError('Two ordered rows of positive finite physical boxes required')
    expected_centers, expected_halves = registered_geometry(route_config)
    actual_centers = np.asarray(geometry['obstacle_centers'])
    actual_halves = np.asarray(geometry['obstacle_halfsizes'])
    # This is the existing collection geometry readback tolerance, not a
    # relaxed collision/classification threshold. Classification uses the
    # original registered coordinates, just as the reference collector does.
    for actual, expected in ((actual_centers, expected_centers), (actual_halves, expected_halves)):
        if actual.shape != (4, 3) or not np.allclose(actual, expected, atol=1e-6, rtol=0):
            raise ValueError('Measured boxes differ from registered two-row geometry')
    goals = np.asarray(specification['centers'])
    expected_goals = np.asarray(route_config['goal_xyz'])
    if goals.shape != (3, 3) or expected_goals.shape != (3, 3) or not np.allclose(goals, expected_goals, atol=1e-6, rtol=0):
        raise ValueError('Evaluation targets differ from registered physical goals')
    for mode in reference_types:
        if mode is not None and (not isinstance(mode, (list, tuple)) or len(mode) != 2
                                 or any(value not in TYPE_VALUES for value in mode)):
            raise ValueError('Reference types must be known two-row relations or unknown')


def scene_metrics(paths, opened, current, geometry, specification, reference_types, route_config, scores=None):
    """Count every submitted slot; keep valid unknown types valid and uncounted."""
    validate_labels(geometry, specification, route_config, reference_types)
    metrics, candidates = base_scene_metrics(
        paths, opened, current, geometry, specification, reference_types,
        clearance=.02, scores=scores,
        type_classifier=lambda path: crossing_signature(path, route_config))
    metrics.update(evaluation_protocol=PROTOCOL,
        type_definition='Ordered, consistent actual row passage relations; not guide IDs or homotopy equivalence',
        reference_policy='Coverage of known positive types only; unobserved types are not negatives',
        unknown_type_policy='Valid unknown routes count in TipValid, not in distinct classified types',
        validity_scope='Correct target within3cm, start within5mm, unchanged reach events, physical-box tip segments with2cm clearance; no arm/IK/execution certificate',
        paths_repaired_or_filtered=False)
    return metrics, candidates
