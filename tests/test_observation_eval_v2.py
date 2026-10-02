"""Collector failures must remain in semantic evaluation denominators."""
import json
import numpy as np
from test_observed_routes import make_observation_fixture
from routeset.observed_route_head import load_observed_dataset
from scripts.train_observed_routes import observation_metrics


def test_missing_references_keep_observation_and_semantics(tmp_path):
    observations, labels, cache = make_observation_fixture(tmp_path)
    rows = [json.loads(line) for line in labels.read_text().splitlines()]
    rows[0]['routes'] = []  # TRAIN no route: exclude only from supervised loss.
    rows[-1]['routes'] = []  # DEV no route: keep in semantic metric denominator.
    labels.write_text('\n'.join(json.dumps(row) for row in rows)+'\n')
    data = load_observed_dataset(observations, labels, cache)
    assert len(data['features']) == 4 and len(data['unreferenced']) == 2
    train_ids = np.flatnonzero((data['splits'] == 'TRAIN') & data['path_mask'].any(1))
    dev_ids = np.flatnonzero(data['splits'] == 'DEV_MODEL')
    assert train_ids.tolist() == [1] and dev_ids.tolist() == [2, 3]
    paths = np.zeros((2, 1, 24, 3), np.float32)
    paths[0] = data['paths'][2, :1]
    spec = data['semantic_targets'][3]
    paths[1, 0] = spec['centers'][spec['target_index']]
    result, per_scene = observation_metrics(paths, np.ones((2, 1, 24)), data, dev_ids)
    assert result['evaluation_protocol'] == 'observation_eval_v2'
    assert result['examples'] == 2 and result['reference_evaluation_examples'] == 1
    assert result['semantic_evaluation_examples'] == 2
    assert result['candidate_matched_ADE_m'] == 0 and result['semantic_goal_accuracy'] == 1
    assert per_scene[1]['candidate_matched_ADE_m'] is None
    assert per_scene[1]['event_sequence_accuracy'] is None
    assert per_scene[1]['semantic_goal_accuracy'] == 1
    assert per_scene[1]['ValidAtK'] is None
    # A failed no-reference semantic prediction must lower the full denominator.
    paths[1, 0, -1] += .1
    result, _ = observation_metrics(paths, np.ones((2, 1, 24)), data, dev_ids)
    assert result['semantic_goal_accuracy'] == .5
