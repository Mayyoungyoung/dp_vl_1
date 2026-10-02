import copy
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.collect_observed_two_row_pilot import geometry, proposed_waypoints
from scripts.evaluate_observed_two_row import scene_metrics
from scripts.evaluate_observed_obstacles import scene_metrics as original_metrics


@pytest.fixture
def case():
    config = json.loads((Path(__file__).resolve().parents[1]/'configs/observed_two_row_pilot_v4.json').read_text())
    centers, halves = geometry(config)
    boxes = dict(obstacle_centers=centers, obstacle_halfsizes=halves)
    current = dict(gripper_pose=np.r_[config['entry_xyz'], 0., 0., 0., 1.], gripper_open=1.)
    target = dict(centers=copy.deepcopy(config['goal_xyz']), target_index=1, tolerance=.03)
    sequences = [('negative_y','negative_y'), ('positive_y','positive_y')]
    paths = np.stack([np.vstack([config['entry_xyz'], proposed_waypoints(config['goal_xyz'][1], mode, config)]) for mode in sequences])
    return config, boxes, current, target, sequences, paths


def test_actual_two_row_types_duplicates_and_invalid_slots(case):
    config, boxes, current, target, modes, paths = case
    collision = paths[0].copy(); collision[1] = boxes['obstacle_centers'][0]
    four = np.stack([paths[0], paths[1], paths[0], collision])
    opened = np.ones(four.shape[:2]);scores = np.array([0., 0., 0., 1.])
    metrics, candidates = scene_metrics(four, opened, current, boxes, target, modes, config, scores)
    assert metrics['TipValidAtK'] == .75 and metrics['AnyTipValidAtK'] == 1
    assert metrics['UniqueClassifiedTipValidAtK'] == 2 and metrics['DuplicateClassifiedTipValidCount'] == 1
    assert metrics['KnownReferenceTypeCoverageAtK'] == 1 and metrics['SelectedTipValidAtK'] == 0
    assert metrics['ValidAtK'] is None and len(candidates) == 4
    assert [tuple(r['declared_passage_type']) for r in candidates[:2]] == modes
    # Extending classification must not alter the old validity predicates.
    original, old_rows = original_metrics(four, opened, current, boxes, target, [], scores=scores)
    for key in ('TipValidAtK','AnyTipValidAtK','semantic_goal_accuracy','TipClearAtK','StartCorrectAtK','EventSequenceCorrectAtK'):
        assert metrics[key] == original[key]
    assert [r['TipValid'] for r in candidates] == [r['TipValid'] for r in old_rows]


def test_valid_unknown_is_not_made_invalid_or_a_new_type(case):
    config, boxes, current, target, _, _ = case
    # Mid-gap at a z outside the finite classifier's lateral/over bands:
    # clear of every physical box but intentionally unclassified.
    path = np.array([config['entry_xyz'], [.14, 0., .90], [.34, 0., .90], config['goal_xyz'][1]])[None]
    metrics, rows = scene_metrics(path, np.ones(path.shape[:2]), current, boxes, target, [None], config)
    assert metrics['TipValidAtK'] == 1 and metrics['UniqueClassifiedTipValidAtK'] == 0
    assert metrics['UnknownTypeTipValidCount'] == 1 and metrics['KnownReferenceTypeCoverageAtK'] is None
    assert rows[0]['declared_passage_type'] is None


def test_wrong_goal_start_events_nonfinite_and_absent_scores_stay_in_budget(case):
    config, boxes, current, target, modes, paths = case
    broken = np.repeat(paths[:1], 4, axis=0)
    broken[0, -1] = config['goal_xyz'][0]
    broken[1, 0, 0] += .006
    broken[3, 2, 1] = np.nan
    opened = np.ones(broken.shape[:2]);opened[2, 2] = 0.
    metrics, rows = scene_metrics(broken, opened, current, boxes, target, modes, config)
    assert metrics['candidates'] == 4 and metrics['TipValidAtK'] == 0
    assert metrics['SelectedTipValidAtK'] is None and metrics['format_failure_count'] == 1
    assert not rows[0]['semantic_goal_correct'] and not rows[1]['starts_at_current_state']
    assert not rows[2]['event_state_sequence_correct'] and not rows[3]['finite_xyz']


def test_no_unknown_negative_labels_and_no_fake_reference_count(case):
    config, boxes, current, target, modes, paths = case
    metrics, _ = scene_metrics(paths, np.ones(paths.shape[:2]), current, boxes, target, [modes[0],modes[0],None], config)
    assert metrics['known_reference_types'] == 1 and metrics['KnownReferenceTypeCoverageAtK'] == 1
    assert metrics['UniqueClassifiedTipValidAtK'] == 2  # New valid relation need not have a reference.


def test_over_is_its_actual_type_and_missing_events_are_failures(case):
    config, boxes, current, target, _, _ = case
    path = np.array([config['entry_xyz'], [.07,0.,1.0], [.4,0.,1.0], config['goal_xyz'][1]])[None]
    metrics, rows = scene_metrics(path, np.ones(path.shape[:2]), current, boxes, target, [], config)
    assert metrics['TipValidAtK'] == 1 and tuple(rows[0]['declared_passage_type']) == ('over','over')
    missing, _ = scene_metrics(path, None, current, boxes, target, [], config)
    assert missing['TipValidAtK'] == 0 and missing['format_failure_count'] == 1


@pytest.mark.parametrize('change', ['clearance','goal','physical_box','one_box','type'])
def test_schema_or_threshold_changes_do_not_silently_reuse_two_row_evaluation(case, change):
    config, boxes, current, target, modes, paths = copy.deepcopy(case)
    if change == 'clearance':config['tip_clearance_m'] = .01
    elif change == 'goal':target['centers'][1][0] += .01
    elif change == 'physical_box':boxes['obstacle_centers'][0, 0] += .01
    elif change == 'one_box':boxes['obstacle_centers'] = boxes['obstacle_centers'][:1]
    else:modes = [('guide0', 'guide1')]
    with pytest.raises(ValueError):scene_metrics(paths, np.ones(paths.shape[:2]), current, boxes, target, modes, config)
