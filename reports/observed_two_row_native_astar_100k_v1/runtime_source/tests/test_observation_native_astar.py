from collections import Counter
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import evaluate_two_row_astar_v2 as control
from scripts import observation_native_astar as native
from scripts import observation_spatial_penalty_astar as spatial


@pytest.fixture(scope='session')
def library(tmp_path_factory):
    if shutil.which('g++') is None:
        pytest.fail('Existing g++ required for actual differential tests; no skip-as-success')
    folder=tmp_path_factory.mktemp('native_astar')/'build'
    native.build_library_for_testing(folder)
    return native.NativeLibrary(folder/'build_receipt.json',for_testing=True)


@contextmanager
def python_fixture_budget(planner,maximum_nodes):
    previous=dict(planner.CONFIG)
    planner.CONFIG.update(maximum_expanded_nodes_per_candidate=maximum_nodes,search_deadline_seconds=float('inf'))
    try:yield
    finally:
        planner.CONFIG.clear();planner.CONFIG.update(previous)


def compare(library,free,lower,goal,starts,goals,cost,start_permission=None,target_permission=None,maximum_nodes=20000):
    planner=control.dependencies()
    if start_permission is None:start_permission=np.zeros_like(free)
    if target_permission is None:target_permission=np.zeros_like(free)
    kernel=native.native_search_for_testing(planner,library,maximum_nodes=maximum_nodes)
    args=(free,np.asarray(lower),np.asarray(goal),starts,goals,cost,start_permission,target_permission)
    with python_fixture_budget(planner,maximum_nodes):
        expected,a=planner.astar_virtual(*args)
    actual,b=kernel(*args)
    assert actual==expected
    ignored={'search_seconds','search_deadline_seconds','native'}
    assert {k:v for k,v in a.items() if k not in ignored}=={k:v for k,v in b.items() if k not in ignored}
    return actual,b


def test_actual_cpp_build_receipt_hash_flags_and_reuse(library):
    r=library.receipt
    assert r['exit_code']==0 and r['elapsed_seconds']>0
    assert '-fno-fast-math' in r['flags'] and '-ffp-contract=off' in r['flags']
    assert native.digest(r['library_path'])==r['library_sha256']
    assert native.digest(r['cpp_source'])==r['cpp_sha256']
    assert native.build_library_for_testing(library.receipt_path.parent)==r
    with pytest.raises(ValueError,match='receipt mismatch'):
        native.NativeLibrary(library.receipt_path,for_testing=False)


def test_open_grid_ties_and_multiple_exact_virtual_connectors(library):
    free=np.ones((9,9,3),bool);lower=np.array([-.103,.027,.831]);goal=lower+np.array([8,8,1])*.025
    starts={(0,0,1):{'length_m':.005},(0,1,1):{'length_m':.005},(1,0,1):{'length_m':.005}}
    goals={(8,8,1):{'length_m':.003},(8,7,1):{'length_m':.003},(7,8,1):{'length_m':.003}}
    permissions=np.zeros_like(free);permissions[:2]=True
    target=np.zeros_like(free);target[-2:]=True
    cost=Counter({('start',(0,0,1)):2,('goal',(8,7,1)):3,('grid',(0,1,1),(1,2,1)):2})
    compare(library,free,lower,goal,starts,goals,cost,permissions,target)


@pytest.mark.parametrize('case',['no_start','no_goal','blocked','node_cap','same_cell'])
def test_failed_virtual_attachments_supercover_limits_and_zero_grid_route(library,case):
    free=np.ones((5,5,3),bool);a,b=(0,0,1),(4,4,1)
    starts={a:{'length_m':0.}};goals={b:{'length_m':.01}};maximum=20000
    if case=='no_start':starts={}
    if case=='no_goal':goals={}
    if case=='blocked':free[2,:,:]=False
    if case=='node_cap':maximum=2
    if case=='same_cell':goals={a:{'length_m':.001}}
    _,result=compare(library,free,np.zeros(3),np.array(b)*.025,starts,goals,Counter(),maximum_nodes=maximum)
    assert result['status']=={'no_start':'no_admissible_virtual_start_attachment','no_goal':'no_admissible_virtual_goal_attachment',
        'blocked':'open_set_exhausted','node_cap':'expanded_node_budget_exhausted','same_cell':'path_found'}[case]


@pytest.mark.parametrize('seed',range(12))
def test_fixed_random_graphs_exact_full_path_and_all_deterministic_counters(library,seed):
    rng=np.random.default_rng(seed)
    free=rng.random((9,8,4))>.11;a,b=(0,0,1),(8,7,2);free[a]=free[b]=True
    lower=rng.uniform(-.8,.8,3);goal=lower+np.array(b)*.025+rng.uniform(-.008,.008,3)
    starts={a:{'length_m':.006}};goals={b:{'length_m':.008}}
    counter=Counter({('start',a):seed%3,('goal',b):seed%2})
    cost=spatial.SpatialCosts(counter,spatial.spatial_field(free,lower,[np.stack([lower,goal])])) if seed%2 else counter
    compare(library,free,lower,goal,starts,goals,cost,rng.random(free.shape)>.8,rng.random(free.shape)>.8)


def test_symmetric_ties_counter_order_and_spatial_repeat_same_first_path(library):
    free=np.ones((11,11,3),bool);free[4:7,4:7,:]=False
    lower=np.zeros(3);a,b=(1,5,1),(9,5,1);goal=np.array(b)*.025
    starts={a:{'length_m':.002}};goals={b:{'length_m':.002}}
    first,_=compare(library,free,lower,goal,starts,goals,Counter())
    planner=control.dependencies();permission=np.zeros_like(free)
    native_kernel=native.native_search_for_testing(planner,library)
    wrapped=spatial.SpatialSearch(native_kernel,np.array(a)*.025,planner.CONFIG)
    used=Counter()
    for slot in range(4):
        cells,record=wrapped(free,lower,goal,starts,goals,used,permission,permission)
        if slot==0:assert cells==first
        # Compare subsequent fields with the Python reference at the same history.
        costs=used if slot==0 else spatial.SpatialCosts(used,spatial.spatial_field(free,lower,wrapped.raw_paths[:-1]))
        expected,_=compare(library,free,lower,goal,starts,goals,costs)
        assert cells==expected
        used[('start',cells[0])]+=1;used[('goal',cells[-1])]+=1
        for u,v in zip(cells,cells[1:]):used[('grid',)+tuple(sorted((u,v)))]+=1


def test_deadline_checked_at64_and_production_limits_not_configurable(library):
    planner=control.dependencies();free=np.ones((15,15,3),bool);free[7,:,:]=False
    search=native.native_search_for_testing(planner,library,deadline_seconds=0.)
    _,record=search(free,np.zeros(3),np.array([.3,.3,.025]),{(0,0,1):{'length_m':0.}},
        {(14,14,1):{'length_m':0.}},Counter(),np.zeros_like(free),np.zeros_like(free))
    assert record['status']=='search_time_budget_exhausted' and record['expanded_nodes']==64
    assert record['native']['ctypes_preparation_counts_toward_search_deadline'] is True
    fake=SimpleNamespace(receipt={'test_only_build':False})
    production=native.NativeSearch(planner,fake)
    assert production.maximum_nodes==20000 and production.deadline_seconds==2.
    with pytest.raises(ValueError,match='Production budget'):
        native.NativeSearch(planner,fake,maximum_nodes=30000)
    with pytest.raises(ValueError,match='Production budget'):
        native.NativeSearch(planner,fake,deadline_seconds=-1.)


def test_production_context_restores_original_even_after_exception():
    planner=control.dependencies();original=planner.astar_virtual;fake=SimpleNamespace(receipt={'test_only_build':False})
    with pytest.raises(RuntimeError,match='deliberate'):
        with native.native_planner(planner,fake):
            assert isinstance(planner.astar_virtual,native.NativeSearch)
            with pytest.raises(RuntimeError,match='Nested'):
                with native.native_planner(planner,fake):pass
            raise RuntimeError('deliberate')
    assert planner.astar_virtual is original and native._ACTIVE is False


def test_actual_kernel_uses_original20000_node_cap_with2second_deadline(library):
    planner=control.dependencies();free=np.ones((90,90,5),bool);free[85,:,:]=False
    # Same exact budget tuple as the production constructor; the artifact is a
    # locally compiled test binary, never a claimed frozen server experiment.
    search=native.native_search_for_testing(planner,library,maximum_nodes=20000,deadline_seconds=2.)
    cells,record=search(free,np.zeros(3),np.array([89,89,2])*.025,{(0,0,2):{'length_m':0.}},
        {(89,89,2):{'length_m':0.}},Counter(),np.zeros_like(free),np.zeros_like(free))
    assert cells is None and record['status']=='expanded_node_budget_exhausted'
    assert record['expanded_nodes']==20000 and record['maximum_expanded_nodes']==20000
    assert record['search_deadline_seconds']==2.


def test_build_cannot_write_immutable_source_or_unregistered_production_location(tmp_path):
    with pytest.raises(ValueError,match='source/release'):
        native.build_library_for_testing(Path(native.__file__).resolve().parent/'forbidden_build')
    with pytest.raises(ValueError,match='Immutable source'):
        native.build_library(tmp_path/'not_registered','a'*40)
