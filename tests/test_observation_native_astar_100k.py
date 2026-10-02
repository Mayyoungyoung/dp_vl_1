from collections import Counter
from contextlib import contextmanager
from types import SimpleNamespace
import shutil
import numpy as np
import pytest
from scripts import evaluate_two_row_astar_v2 as control
from scripts import observation_native_astar as base
from scripts import observation_native_astar_100k as native
from scripts import observation_spatial_penalty_astar as spatial


@pytest.fixture(scope='session')
def library100k(tmp_path_factory):
    if shutil.which('g++') is None:
        pytest.fail('Actual compiler required, no skipped differential evidence')
    folder=tmp_path_factory.mktemp('native_astar100k')/'build'
    base.build_library_for_testing(folder)
    return native.NativeLibrary(folder/'build_receipt.json',for_testing=True)


@contextmanager
def reference_budget(planner):
    before=dict(planner.CONFIG)
    planner.CONFIG.update(maximum_expanded_nodes_per_candidate=100000,search_deadline_seconds=float('inf'))
    try:yield
    finally:
        planner.CONFIG.clear();planner.CONFIG.update(before)


@pytest.mark.parametrize('seed',range(8))
def test100k_exact_python_path_and_all_deterministic_counters(library100k,seed):
    planner=control.dependencies();rng=np.random.default_rng(seed+100)
    free=rng.random((9,8,4))>.11;a,b=(0,0,1),(8,7,2);free[a]=free[b]=True
    lower=rng.uniform(-.8,.8,3);goal=lower+np.array(b)*.025+rng.uniform(-.008,.008,3)
    starts={a:{'length_m':.006},(0,0,2):{'length_m':.007}}
    goals={b:{'length_m':.008},(8,7,1):{'length_m':.009}}
    counter=Counter({('start',a):seed%3,('goal',b):seed%2})
    cost=spatial.SpatialCosts(counter,spatial.spatial_field(free,lower,[np.stack([lower,goal])])) if seed%2 else counter
    args=(free,lower,goal,starts,goals,cost,rng.random(free.shape)>.8,rng.random(free.shape)>.8)
    search=native.TestSearch100K(planner,library100k)
    with reference_budget(planner):expected,left=planner.astar_virtual(*args)
    actual,right=search(*args)
    assert expected==actual
    ignored={'search_seconds','search_deadline_seconds','native'}
    assert {k:v for k,v in left.items() if k not in ignored}=={k:v for k,v in right.items() if k not in ignored}
    assert right['maximum_expanded_nodes']==100000 and right['search_deadline_seconds']==2.
    assert right['native']['effective_search_config']==native.EFFECTIVE_SEARCH_CONFIG
    assert right['native']['abi_protocol']==base.PROTOCOL
    assert planner.CONFIG['maximum_expanded_nodes_per_candidate']==20000


def test_actual100000_node_cap_with_production2second_budget(library100k):
    planner=control.dependencies();free=np.ones((150,150,6),bool);free[145,:,:]=False
    search=native.TestSearch100K(planner,library100k)
    a,b=(0,0,2),(149,149,2)
    cells,record=search(free,np.zeros(3),np.array(b)*.025,{a:{'length_m':0.}},
        {b:{'length_m':0.}},Counter(),np.zeros_like(free),np.zeros_like(free))
    assert cells is None and record['status']=='expanded_node_budget_exhausted'
    assert record['expanded_nodes']==100000 and record['maximum_expanded_nodes']==100000
    assert record['search_deadline_seconds']==2.
    assert record['native']['clock_check_interval_expanded_nodes']==64


def test_explicit_production_never_accepts_test_library_or_budget_mutation(library100k):
    planner=control.dependencies()
    with pytest.raises(ValueError,match='production/test'):
        native.ProductionSearch100K(planner,library100k)
    fake=SimpleNamespace(receipt={'test_only_build':False})
    instance=native.ProductionSearch100K(planner,fake)
    assert type(instance) is native.ProductionSearch100K and instance.testing is False
    assert instance.maximum_nodes==100000 and instance.deadline_seconds==2.
    for kwargs in ({'maximum_nodes':20000},{'deadline_seconds':3.},{'testing':True}):
        with pytest.raises(TypeError):native.ProductionSearch100K(planner,fake,**kwargs)
    instance.maximum_nodes=20000
    with pytest.raises(ValueError,match='mutated'):instance()
    planner.CONFIG['maximum_expanded_nodes_per_candidate']=100000
    try:
        with pytest.raises(ValueError,match='Frozen base'):native.ProductionSearch100K(planner,fake)
    finally:planner.CONFIG['maximum_expanded_nodes_per_candidate']=20000


def test_production_context_restores_graph_and_refuses_nesting():
    planner=control.dependencies();original=planner.astar_virtual
    config=dict(planner.CONFIG);fake=SimpleNamespace(receipt={'test_only_build':False})
    with pytest.raises(RuntimeError,match='deliberate'):
        with native.native_planner(planner,fake):
            assert type(planner.astar_virtual) is native.ProductionSearch100K
            with pytest.raises(RuntimeError,match='Nested'):
                with native.native_planner(planner,fake):pass
            raise RuntimeError('deliberate')
    assert planner.astar_virtual is original and planner.CONFIG==config and native._ACTIVE is False
    with base.native_planner(planner,fake):
        with pytest.raises(RuntimeError,match='Nested'):
            with native.native_planner(planner,fake):pass


def test_actual_build_hash_and_old20k_production_class_remain_separate(library100k):
    r=library100k.receipt
    assert r['test_only_build'] is True and base.digest(r['library_path'])==r['library_sha256']
    assert r['protocol']==base.PROTOCOL and r['cpp_sha256']==base.digest(base.SOURCE)
    assert base.build_library_for_testing(library100k.receipt_path.parent)==r
    fake=SimpleNamespace(receipt={'test_only_build':False})
    assert base.NativeSearch(control.dependencies(),fake).maximum_nodes==20000
    with pytest.raises(ValueError,match='Production budget'):
        base.NativeSearch(control.dependencies(),fake,maximum_nodes=100000)
