"""One explicit production 100k-node budget check; original ABI/graph stay frozen.

This is five times the old node allowance for BOTH arms, not a new planner.
There is no runtime cap/deadline override and no production test-only shortcut.
"""
from contextlib import contextmanager
import hashlib
import numpy as np
from scripts import observation_native_astar as base

PROTOCOL = 'observed_astar_native_exact_graph_100k_v1'
ABI_PROTOCOL = base.PROTOCOL
MAXIMUM_NODES = 100000
DEADLINE_SECONDS = 2.
EFFECTIVE_SEARCH_CONFIG = dict(protocol=PROTOCOL, maximum_expanded_nodes_per_candidate=MAXIMUM_NODES,
    search_deadline_seconds=DEADLINE_SECONDS, clock_check_interval_expanded_nodes=64,
    historical_node_cap=20000, node_budget_multiplier=5, both_arms=True)
# These aliases reuse the unchanged ABI compilation and receipt verification.
digest, FLAGS = base.digest, base.FLAGS
NativeLibrary, build_library = base.NativeLibrary, base.build_library
_ACTIVE = False


def _initialize(search, planner, library, test_only):
    if library.receipt.get('test_only_build') is not test_only:
        raise ValueError('Explicit100k production/test build mismatch')
    if (planner.CONFIG['voxel_m'], planner.CONFIG['used_edge_penalty_multiplier'], planner.CONFIG['candidates'],
        planner.CONFIG['horizon'], planner.CONFIG['maximum_expanded_nodes_per_candidate'],
        planner.CONFIG['search_deadline_seconds']) != (.025,4.,4,24,20000,2.):
        raise ValueError('Frozen base planner geometry/configuration changed')
    search.planner, search.library, search.testing = planner, library, test_only
    # Independent explicit construction. The old20k constructor is never called.
    search.maximum_nodes, search.deadline_seconds = MAXIMUM_NODES, DEADLINE_SECONDS
    neighbors = planner.v1.NEIGHBORS
    if len(neighbors) != 26:
        raise ValueError('Exactly original26 neighbors required')
    search.deltas = base._array([r[0] for r in neighbors],np.int32)
    search.lengths = base._array([r[2] for r in neighbors],np.float64)
    search.prefix = base._array(np.r_[0,np.cumsum([len(r[1]) for r in neighbors])],np.int32)
    search.touches = base._array([cell for r in neighbors for cell in r[1]],np.int32)
    search.neighbors_sha256 = hashlib.sha256(b''.join(x.tobytes() for x in (
        search.deltas,search.lengths,search.prefix,search.touches))).hexdigest()


class ProductionSearch100K(base.NativeSearch):
    def __init__(self, planner, library):
        _initialize(self,planner,library,False)

    def __call__(self, *args):
        if self.maximum_nodes != MAXIMUM_NODES or self.deadline_seconds != DEADLINE_SECONDS:
            raise ValueError('Fixed100k production budget mutated')
        cells, result = super().__call__(*args)
        result['native'].update(protocol=PROTOCOL, abi_protocol=ABI_PROTOCOL,
            effective_search_config=dict(EFFECTIVE_SEARCH_CONFIG), budget_adapter_sha256=digest(__file__))
        return cells,result


class TestSearch100K(ProductionSearch100K):
    """Only synthetic fixtures. Production context cannot instantiate this class."""
    def __init__(self, planner, library):
        _initialize(self,planner,library,True)


@contextmanager
def native_planner(planner, library):
    global _ACTIVE
    if _ACTIVE or base._ACTIVE:
        raise RuntimeError('Nested/concurrent native planner context prohibited')
    original = planner.astar_virtual
    adapted = ProductionSearch100K(planner,library)
    _ACTIVE = True
    planner.astar_virtual = adapted
    try:
        yield adapted
    finally:
        planner.astar_virtual = original
        _ACTIVE = False
