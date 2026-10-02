import ast
import itertools
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_observed_reference_coverage import (assignment_audit, audit_scene,
    constant_event_reference, coverage_audit, validate_predictions)


def test_saturation_optimum_and_runner_up_match_exhaustive():
    rng=np.random.default_rng(7)
    for m in range(1,7):
        cost=rng.uniform(size=(4,m))
        possible=[float(cost[np.arange(4),ids].mean()) for ids in itertools.product(range(m),repeat=4)
                  if len(set(ids))==min(4,m)]
        expected=sorted(possible);actual=assignment_audit(cost)
        assert actual['best_cost']==pytest.approx(expected[0])
        assert actual['second_assignment_cost']==pytest.approx(expected[1]) if len(expected)>1 else actual['second_assignment_cost'] is None
    assert assignment_audit(np.empty((4,0))) is None
    assert assignment_audit(np.zeros((4,2)))['second_minus_best']==0


def test_constant_reference_matches_actual_training_source():
    path=Path(__file__).resolve().parents[1]/'routeset/observed_route_head.py'
    tree=ast.parse(path.read_text(encoding='utf-8-sig'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='resample_event_segments')
    namespace={'np':np};exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),namespace)
    rng=np.random.default_rng(9)
    for poses in (rng.normal(size=(37,7)),np.ones((3,7)),np.zeros((1,7))):
        expected=namespace['resample_event_segments'](poses,np.ones(len(poses)),24)
        actual=constant_event_reference(poses,np.ones(len(poses)),24)
        assert all(np.array_equal(a,b) for a,b in zip(expected,actual))
    with pytest.raises(ValueError,match='constant-event'):
        constant_event_reference(np.zeros((2,7)),[0,1],24)


def route(kind,valid=True):
    return dict(declared_passage_type=None if kind is None else [kind],TipValid=valid)


def test_unknowns_do_not_become_duplicate_or_novel_classes():
    result=coverage_audit([route('a'),route('b'),route(None),route(None)],
                          [route('a'),route('a'),route(None),route('b',False)])
    assert result['unknown_valid_reference_count']==2
    assert result['valid_classified_duplicate_candidates']==1
    assert result['missing_supported_known_types']==[['b']]
    assert result['duplicate_replacement_reference_oracle_gain']==1
    assert coverage_audit([], [route(None)]*4)['duplicate_replacement_reference_oracle_gain']==0


def test_valid_opposite_routes_can_have_colliding_interpolation():
    refs=np.array([[[-1,0,0],[-1,1,0],[1,1,0],[1,0,0]],
                   [[-1,0,0],[-1,-1,0],[1,-1,0],[1,0,0]]],dtype=np.float32)
    current=dict(gripper_pose=np.array([-1,0,0,0,0,0,1]),gripper_open=np.array(1.))
    geometry=dict(obstacle_centers=np.zeros((1,3)),obstacle_halfsizes=np.ones((1,3))*.1)
    label=dict(semantic_targets=dict(centers=[[1,0,0]],target_index=0,tolerance=.03),route_types=[])
    out=audit_scene(np.repeat(refs[:1],4,axis=0),np.ones((4,4)),refs,np.ones((2,4)),current,geometry,label,.2)
    assert all(r['TipValid'] for r in out['references'])
    midpoint=next(r for r in out['interpolation_probes'] if r['alpha']==.5)
    assert not midpoint['tip_check']['tip_segments_clear']
    empty=audit_scene(np.repeat(refs[:1],4,axis=0),np.ones((4,4)),np.asarray([]),np.asarray([]),current,geometry,label,.2)
    assert empty['assignment'] is None and len(empty['candidates'])==4


def test_all_input_shapes_and_parent_identity_are_checked():
    rows=[dict(id='a',parent_id='p'),dict(id='b',parent_id='q')]
    paths=np.zeros((2,4,24,3));events=np.ones((2,4,24))
    assert validate_predictions(['a','b'],['p','q'],paths,events,rows,24)=={'a':0,'b':1}
    with pytest.raises(ValueError):validate_predictions(['a','b'],['p','q'],paths[:1],events,rows,24)
    with pytest.raises(ValueError):validate_predictions(['a','b'],['p','p'],paths,events,rows,24)
    with pytest.raises(ValueError):validate_predictions(['a'],['p'],paths[:1],events[:1],rows,24)
