from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import observation_spatial_penalty_astar as spatial
from scripts import evaluate_two_row_astar_v2 as control


def config():
    return json.loads((Path(__file__).resolve().parents[1]/'configs/observed_two_row_spatial_penalty_v1.json').read_text())


def update_counter(counter, cells):
    if cells is None:
        return
    counter[('start', cells[0])] += 1
    counter[('goal', cells[-1])] += 1
    for a, b in zip(cells, cells[1:]):
        counter[('grid',)+tuple(sorted((a, b)))] += 1


def test_continuous_segment_distance_includes_interiors_and_degenerate_segments():
    raw = np.array([[0.,0.,0.],[0.,0.,0.],[1.,1.,0.]])
    points = np.array([[.5,.5,0.],[.5,.5,.05],[-1.,0.,0.],[2.,2.,0.]])
    np.testing.assert_allclose(spatial.squared_polyline_distance(points, raw), [0.,.0025,1.,2.], atol=1e-15)
    with pytest.raises(ValueError):
        spatial.squared_polyline_distance(points, [[0,0,np.nan],[1,1,1]])


def test_field_exact_gaussian_sum_not_edge_frequency_or_vertex_distance():
    free = np.ones((5,5,3), bool)
    raw = np.array([[0.,0.,0.],[.1,.1,0.]])
    field = spatial.spatial_field(free, np.zeros(3), [raw])
    assert field[1,1,0] == pytest.approx(1.)  # On continuous interior, no raw vertex here.
    assert field[1,1,2] == pytest.approx(np.exp(-.5))
    np.testing.assert_allclose(spatial.spatial_field(free, np.zeros(3), [raw,raw]), 2*field)
    assert not spatial.spatial_field(free, np.zeros(3), []).any()
    with pytest.raises(ValueError):
        spatial.spatial_field(free, np.zeros(3), [raw], sigma=.1)


def test_virtual_costs_unchanged_grid_uses_endpoint_mean():
    a, b = (0,0,0), (1,0,0)
    counter = Counter({('start',a):2, ('goal',b):3, ('grid',a,b):100})
    field = np.array([[[.2]],[[.6]]])
    cost = spatial.SpatialCosts(counter, field)
    assert cost[('start',a)] == 2 and cost[('goal',b)] == 3
    assert cost[('grid',a,b)] == pytest.approx(.4)
    assert .025*(1+4*cost[('grid',a,b)]) == pytest.approx(.065)
    assert cost[('start',b)] == 0


def test_zero_history_calls_actual_original_search_without_cost_proxy():
    planner = control.dependencies()
    free = np.ones((8,8,2), bool)
    lower, current = np.zeros(3), np.array([0.,0.,0.,0,0,0,1,1])
    goal = np.array([.175,.175,.025])
    starts, goals = {(0,0,0):{'length_m':0.}}, {(7,7,1):{'length_m':0.}}
    permission, counter = np.zeros_like(free), Counter()
    expected, record = planner.astar_virtual(free,lower,goal,starts,goals,counter,permission,permission)
    calls = []
    original = planner.astar_virtual
    def observed(*args):
        calls.append(args[5] is counter)
        return original(*args)
    wrapped = spatial.SpatialSearch(observed,current,planner.CONFIG)
    actual, result = wrapped(free,lower,goal,starts,goals,counter,permission,permission)
    assert actual == expected and calls == [True]
    assert result['expanded_nodes'] == record['expanded_nodes']
    assert result['spatial_penalty']['previous_returned_raw_paths'] == 0


def test_all_returned_paths_included_failures_not_added_and_counter_verified():
    planner = control.dependencies()
    a, b = (0,0,0), (1,1,0)
    histories, paths = [], [[a,b],None,[a,b],None]
    def original(free,lower,goal,starts,goals,cost,*unused):
        histories.append(cost[('grid',a,b)])
        path = paths[len(histories)-1]
        # Deliberately failed proxy flags do not suppress a complete path.
        return path, dict(search_seconds=.001, raw_original_grid_proxy_clear=False,
                         complete_raw_paths_emitted=int(path is not None))
    wrapped = spatial.SpatialSearch(original,np.zeros(8),planner.CONFIG)
    free = np.ones((2,2,1),bool)
    used = Counter()
    for i in range(4):
        cells, result = wrapped(free,np.zeros(3),np.array([.025,.025,0]),{}, {},used,free,free)
        assert result['spatial_penalty']['previous_returned_raw_paths'] == [0,1,1,2][i]
        update_counter(used,cells)
    np.testing.assert_allclose(histories,[0,1,1,2])
    with pytest.raises(ValueError,match='Four searches'):
        wrapped(free,np.zeros(3),np.zeros(3),{}, {},used,free,free)
    another = spatial.SpatialSearch(original,np.zeros(8),planner.CONFIG)
    with pytest.raises(ValueError,match='Counter differs'):
        another(free,np.zeros(3),np.zeros(3),{}, {},used,free,free)


def test_scoped_adapter_restores_on_error_and_rejects_nesting():
    planner = control.dependencies()
    original = planner.astar_virtual
    with pytest.raises(RuntimeError,match='deliberate'):
        with spatial.spatial_adapter(planner,np.zeros(8)):
            assert planner.astar_virtual is not original
            with pytest.raises(RuntimeError,match='Nested'):
                with spatial.spatial_adapter(planner,np.zeros(8)):
                    pass
            raise RuntimeError('deliberate')
    assert planner.astar_virtual is original and spatial._ACTIVE is False


@pytest.mark.parametrize('role', ['TRAIN','DEV_MODEL'])
def test_request_preserves_real_role_seals_before_labels_and_keeps_all_failures(tmp_path, monkeypatch, role):
    from test_two_row_astar_control import make_fixture
    data = tmp_path/'data'
    row,label,manifest,planner,calls = make_fixture(data)
    row['split'] = label['split'] = role
    (data/'supervision.jsonl').write_text(json.dumps(label)+'\n')
    output = tmp_path/'request'
    read_label = spatial.label_for
    def checked_label(data,row):
        seal = json.loads((output/'generation_seal.json').read_text())
        assert seal['split'] == role and not seal['evaluation_labels_opened']
        assert seal['prediction_sha256'] == control.digest(output/'predictions.npz')
        assert calls == ['known']
        return read_label(data,row)
    monkeypatch.setattr(spatial,'label_for',checked_label)
    result,paths,_ = spatial.one_request(data,row,manifest,{'prototypes':{}},{'lower':[],'upper':[]},planner,output,'edge')
    assert result['split'] == role and result['generation']['failed_slots'] == 2
    assert np.isnan(paths[2:]).all() and np.array_equal(paths[0],paths[1])
    assert result['reference_count'] == 0
    assert spatial.aggregate([result])['KnownReferenceTypeCoverageAtK'] is None


def test_changed_labels_do_not_change_generation_input_or_output(tmp_path):
    from test_two_row_astar_control import make_fixture
    data = tmp_path/'data'
    row,label,manifest,planner,calls = make_fixture(data)
    row['split'] = label['split'] = 'TRAIN'
    (data/'supervision.jsonl').write_text(json.dumps(label)+'\n')
    args = (data,row,manifest,{'prototypes':{}},{'lower':[],'upper':[]},planner)
    first,paths,_ = spatial.one_request(*args,tmp_path/'one','edge')
    label['semantic_targets']['target_index'] = 0
    (data/'supervision.jsonl').write_text(json.dumps(label)+'\n')
    second,again,_ = spatial.one_request(*args,tmp_path/'two','edge')
    np.testing.assert_array_equal(paths,again)
    assert calls == ['known','known']
    assert first['metrics']['semantic_goal_accuracy'] != second['metrics']['semantic_goal_accuracy']
    with pytest.raises(ValueError,match='Strict registered'):
        spatial.load_current(dict(row,target_xyz=[0,0,0]),manifest,{},planner)


def test_id_first_label_reader_does_not_decode_other_payload(tmp_path):
    row = dict(id='train0',parent_id='p',split='TRAIN')
    (tmp_path/'supervision.jsonl').write_text('{"id":"dev0","unrelated_payload":malformed}\n'+json.dumps(row)+'\n')
    assert spatial.label_for(tmp_path,row) == row


def registered_rows():
    return [dict(id='two_row_reach_%d_target%d'%(283200+i,t), parent_id='two_row_reach_%d'%(283200+i),
                 split='TRAIN' if i<64 else 'DEV_MODEL',image='observed.png',instruction='input')
            for i in list(range(32))+list(range(64,76)) for t in range(3)]


def test_exact_train_and_dev_registration_rejects_relabeling_or_tuning():
    selection = json.loads((Path(__file__).resolve().parents[1]/'configs/observed_two_row_prefix44_selection_v1.json').read_text())
    rows, manifest, cfg = registered_rows(), {'selection':selection}, config()
    train = spatial.validate_registration(cfg,manifest,rows,'train_preflight')
    dev = spatial.validate_registration(cfg,manifest,rows,'dev')
    assert len(train) == 12 and all(r['split']=='TRAIN' for r in train)
    assert len(dev) == 36 and all(r['split']=='DEV_MODEL' for r in dev)
    with pytest.raises(ValueError,match='Fixed spatial'):
        spatial.validate_registration(dict(cfg,sigma_m=.1),manifest,rows,'train_preflight')
    with pytest.raises(ValueError,match='identity/role'):
        spatial.validate_registration(cfg,manifest,[dict(rows[0],split='DEV_MODEL')]+rows[1:],'train_preflight')
    with pytest.raises(ValueError,match='first4'):
        spatial.validate_registration(cfg,manifest,rows[1:],'train_preflight')


def test_original_three_sources_unchanged_and_unknown_instruction_has_four_failures():
    planner = control.dependencies()
    before = {name:control.digest(Path(control.__file__).with_name(name)) for name in control.FROZEN_SOURCES}
    inputs = (None,None,None,None,None,None,np.array([0,0,0,0,0,0,1,1]))
    paths, raw, record = spatial.generate(planner,inputs,'unknown language',{'prototypes':{}},None,'spatial')
    control.validate_pool(paths,raw,record)
    assert record['spatial_adapter']['actual_astar_calls'] == 0 and np.isnan(paths).all()
    assert record['submitted_candidate_budget'] == 4
    assert before == {name:control.digest(Path(control.__file__).with_name(name)) for name in control.FROZEN_SOURCES}


def test_fit_identity_excludes_only_measured_runtime_not_model_or_training_provenance():
    model = dict(prototypes={'known':{'center':[.1,.2,.3]}}, training_seconds=1.,
        config={'rule':'unchanged'},training_rows=[{'id':'p0'}],training_source_sha256={'positive':'frozen'})
    prior = dict(lower=[0,0,0],upper=[1,1,1],source_sha256={'positive':'frozen'})
    original = spatial.fit_identity(model,prior)
    assert original == spatial.fit_identity(dict(model,training_seconds=2.),prior)
    assert model['training_seconds'] == 1.  # Saved full metadata stays intact.
    assert original != spatial.fit_identity(dict(model,prototypes={'different':{}}),prior)
    assert original != spatial.fit_identity(dict(model,training_source_sha256={'positive':'changed'}),prior)
    assert original != spatial.fit_identity(model,dict(prior,lower=[1,0,0]))
    with pytest.raises(ValueError,match='runtime'):
        spatial.fit_identity(dict(model,training_seconds=float('nan')),prior)
