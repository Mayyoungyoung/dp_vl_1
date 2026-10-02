"""Private-build C++17 search adapter; frozen observation geometry stays Python.

Both edge and spatial baselines use the same library. Production has no
configurable node/deadline budget. Test-only differential entry is explicit.
"""
from collections import Counter
from contextlib import contextmanager
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

import numpy as np

from scripts import evaluate_two_row_astar_v2 as control
from scripts import observation_spatial_penalty_astar as spatial

PROTOCOL = 'observed_astar_native_exact_graph_v1'
SOURCE = Path(__file__).resolve().parents[1]/'routeset/native/observed_astar.cpp'
FLAGS = ['-std=c++17','-O3','-shared','-fno-fast-math','-ffp-contract=off']
COUNTERS = ('expanded_nodes','neighbor_edges_examined','free_edges_examined',
    'diagonal_supercover_voxels_examined','virtual_start_edges','virtual_goal_edges',
    'virtual_goal_edges_examined','permission_start_edge_checks','permission_target_edge_checks')
SUCCESS_COUNTERS = ('successful_route_grid_edges','successful_route_virtual_edges',
    'successful_route_edges_in_start_permission','successful_route_edges_in_target_permission')
STATUSES = ('path_found','no_admissible_virtual_start_attachment','no_admissible_virtual_goal_attachment',
    'expanded_node_budget_exhausted','search_time_budget_exhausted','open_set_exhausted')
_ACTIVE = False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _build(build_dir, source_commit, compiler, testing):
    source_root = Path(__file__).resolve().parents[1]
    build_dir = Path(build_dir).resolve()
    if source_root == build_dir or source_root in build_dir.parents:
        # Local tests also keep compiler output outside the source tree.
        raise ValueError('Never build inside the source/release tree')
    if not testing and (not re.fullmatch('[0-9a-f]{40}',source_commit) or source_root.name != source_commit
            or list(build_dir.parts[-3:]) != ['research_v2','native_builds',source_commit]):
        raise ValueError('Immutable source commit and project-private native_builds/<commit> required')
    compiler = Path(shutil.which(compiler) or compiler).resolve()
    if not compiler.is_file():
        raise FileNotFoundError('Existing C++ compiler required; never install a toolchain')
    flags = FLAGS + (['-static-libgcc','-static-libstdc++'] if os.name == 'nt' else ['-fPIC'])
    library = build_dir/('observed_astar.dll' if os.name == 'nt' else 'observed_astar.so')
    receipt_path = build_dir/'build_receipt.json'
    identity = dict(protocol=PROTOCOL, source_commit=source_commit, test_only_build=testing,
        cpp_source=str(SOURCE), cpp_sha256=digest(SOURCE), adapter_source=str(Path(__file__).resolve()),
        adapter_sha256=digest(__file__), compiler_path=str(compiler), compiler_sha256=digest(compiler),
        flags=flags, flags_sha256=hashlib.sha256(json.dumps(flags,separators=(',',':')).encode()).hexdigest(),
        library_path=str(library), system=platform.system(), machine=platform.machine())
    if build_dir.exists():
        if not receipt_path.is_file():
            raise ValueError('Unsealed prior build: preserve artifacts, no automatic overwrite/recompile')
        saved = json.loads(receipt_path.read_text())
        if saved.get('exit_code') != 0 or any(saved.get(k) != v for k,v in identity.items()) or digest(library) != saved['library_sha256']:
            raise ValueError('Existing private build source/compiler/flags/binary mismatch')
        return saved
    build_dir.mkdir(parents=True)
    started = time.perf_counter()
    version = subprocess.run([str(compiler),'--version'],capture_output=True,text=True,check=True)
    command = [str(compiler),*flags,str(SOURCE),'-o',str(library)]
    result = subprocess.run(command,capture_output=True,text=True)
    receipt = dict(identity, command=command, compiler_version=version.stdout,
        compiler_version_sha256=hashlib.sha256(version.stdout.encode()).hexdigest(),
        exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr,
        elapsed_seconds=time.perf_counter()-started, library_sha256=digest(library) if library.is_file() else None)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode:
        raise RuntimeError('Private compile failed; receipt preserved, no automatic retry')
    return receipt


def build_library(build_dir, source_commit, compiler='g++'):
    return _build(build_dir,source_commit,compiler,False)


def build_library_for_testing(build_dir, compiler='g++'):
    return _build(build_dir,'test-only-uncommitted-source',compiler,True)


class NativeLibrary:
    def __init__(self, receipt_path, for_testing=False):
        self.receipt_path = Path(receipt_path).resolve()
        receipt = json.loads(self.receipt_path.read_text())
        expected_flags = FLAGS + (['-static-libgcc','-static-libstdc++'] if os.name == 'nt' else ['-fPIC'])
        if (receipt['protocol'] != PROTOCOL or receipt['exit_code'] != 0
                or receipt['test_only_build'] != bool(for_testing)
                or receipt['cpp_sha256'] != digest(SOURCE) or receipt['adapter_sha256'] != digest(__file__)
                or receipt['library_sha256'] != digest(receipt['library_path'])
                or receipt['compiler_sha256'] != digest(receipt['compiler_path'])
                or receipt['flags'] != expected_flags
                or receipt['flags_sha256'] != hashlib.sha256(json.dumps(expected_flags,separators=(',',':')).encode()).hexdigest()):
            raise ValueError('Native binary/source/compiler/flags receipt mismatch')
        self.receipt = receipt
        self.receipt_sha256 = digest(self.receipt_path)
        self.dll_directory = os.add_dll_directory(str(Path(receipt['compiler_path']).parent)) if os.name == 'nt' else None
        self.library = ctypes.CDLL(receipt['library_path'])
        self.library.observed_astar_abi_version.restype = ctypes.c_int
        if self.library.observed_astar_abi_version() != 1:
            raise ValueError('Native ABI mismatch')
        i, d, p = ctypes.c_int32, ctypes.c_double, ctypes.c_void_p
        self.function = self.library.observed_astar_v1
        self.function.argtypes = [i,i,i,p,p,p,p,p,i,p,p,p,i,p,p,p,p,p,p,p,i,p,p,p,i,d,d,p,i,p,p]
        self.function.restype = ctypes.c_int


def _array(value, dtype):
    return np.ascontiguousarray(value,dtype=dtype)


def _pointer(array):
    return None if array is None else ctypes.c_void_p(array.ctypes.data)


class NativeSearch:
    def __init__(self, planner, library, testing=False, maximum_nodes=20000, deadline_seconds=2.):
        if not testing and (maximum_nodes != 20000 or deadline_seconds != 2.):
            raise ValueError('Production budget is fixed at 20000 nodes /2 seconds')
        if testing != bool(library.receipt['test_only_build']):
            raise ValueError('Test/production native build mismatch')
        if (planner.CONFIG['voxel_m'],planner.CONFIG['used_edge_penalty_multiplier'],planner.CONFIG['candidates'],
                planner.CONFIG['horizon'],planner.CONFIG['maximum_expanded_nodes_per_candidate'],
                planner.CONFIG['search_deadline_seconds']) != (.025,4.,4,24,20000,2.):
            raise ValueError('Frozen planner parameters changed')
        self.planner, self.library, self.testing = planner, library, testing
        self.maximum_nodes, self.deadline_seconds = int(maximum_nodes), float(deadline_seconds)
        if self.maximum_nodes < 1 or self.maximum_nodes > 20000 or self.deadline_seconds < -1:
            raise ValueError('Invalid bounded test fixture')
        neighbors = planner.v1.NEIGHBORS
        if len(neighbors) != 26:
            raise ValueError('Exactly original26 neighbors required')
        self.deltas = _array([row[0] for row in neighbors],np.int32)
        self.lengths = _array([row[2] for row in neighbors],np.float64)
        self.prefix = _array(np.r_[0,np.cumsum([len(row[1]) for row in neighbors])],np.int32)
        self.touches = _array([cell for row in neighbors for cell in row[1]],np.int32)
        self.neighbors_sha256 = hashlib.sha256(b''.join(x.tobytes() for x in (
            self.deltas,self.lengths,self.prefix,self.touches))).hexdigest()

    def __call__(self, free, lower, exact_goal, starts, goals, used_edges, start_permission, target_permission):
        began = time.perf_counter()
        free = _array(free,np.uint8)
        if free.ndim != 3 or not 0 < free.size <= 2_000_000 or not np.isin(free,[0,1]).all():
            raise ValueError('Original bounded binary3D grid required')
        start_permission, target_permission = _array(start_permission,np.uint8), _array(target_permission,np.uint8)
        if start_permission.shape != free.shape or target_permission.shape != free.shape:
            raise ValueError('Permission arrays must match grid')
        lower, exact_goal = _array(lower,np.float64), _array(exact_goal,np.float64)
        if lower.shape != (3,) or exact_goal.shape != (3,) or not np.isfinite([lower,exact_goal]).all():
            raise ValueError('Finite exact endpoints required')
        counter = used_edges.used_edges if isinstance(used_edges,spatial.SpatialCosts) else used_edges
        if not isinstance(counter,Counter):
            raise ValueError('Original Counter or spatial cost proxy required')
        def index(cell):
            if len(cell) != 3 or any(type(x) not in (int,np.int32,np.int64) for x in cell):
                raise ValueError('Integer grid cell required')
            return int(np.ravel_multi_index(cell,free.shape))
        def attachments(items,kind):
            ids, lengths, penalties = [],[],[]
            for cell, attachment in items.items():  # Preserve exact Python dict insertion order.
                ids.append(index(cell)); lengths.append(attachment['length_m']); penalties.append(counter[(kind,cell)])
            return _array(ids,np.int32),_array(lengths,np.float64),_array(penalties,np.float64)
        start_ids, start_lengths, start_penalties = attachments(starts,'start')
        goal_ids, goal_lengths, goal_penalties = attachments(goals,'goal')
        edge_ids, edge_penalties = [],[]
        for key, value in counter.items():
            if not np.isfinite(value) or value < 0:
                raise ValueError('Nonnegative finite edge counts required')
            if key[0] == 'grid':
                edge_ids.append([index(key[1]),index(key[2])]);edge_penalties.append(value)
            elif key[0] not in ('start','goal'):
                raise ValueError('Unknown Counter edge')
        edge_ids, edge_penalties = _array(edge_ids,np.int32),_array(edge_penalties,np.float64)
        field = _array(used_edges.field,np.float64) if isinstance(used_edges,spatial.SpatialCosts) else None
        if field is not None and (field.shape != free.shape or not np.isfinite(field).all() or np.any(field < 0)):
            raise ValueError('Finite nonnegative original spatial field required')
        for values in (start_lengths,goal_lengths,start_penalties,goal_penalties):
            if not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError('Finite nonnegative virtual costs required')
        output = np.empty(self.maximum_nodes+1,dtype=np.int32)
        stats, kernel = np.zeros(14,dtype=np.int64),np.zeros(1,dtype=np.float64)
        prep = time.perf_counter()-began
        status = self.library.function(*free.shape,_pointer(free),_pointer(start_permission),_pointer(target_permission),
            _pointer(lower),_pointer(exact_goal),len(start_ids),_pointer(start_ids),_pointer(start_lengths),_pointer(start_penalties),
            len(goal_ids),_pointer(goal_ids),_pointer(goal_lengths),_pointer(goal_penalties),
            _pointer(self.deltas),_pointer(self.prefix),_pointer(self.touches),_pointer(self.lengths),
            len(edge_penalties),_pointer(edge_ids),_pointer(edge_penalties),_pointer(field),
            self.maximum_nodes,self.deadline_seconds,prep,_pointer(output),len(output),_pointer(stats),_pointer(kernel))
        if not 0 <= status < len(STATUSES):
            raise RuntimeError('Native kernel failed with status%d; no fallback or replacement'%status)
        cells = [tuple(int(x) for x in np.unravel_index(int(n),free.shape)) for n in output[:int(stats[13])]] if status == 0 else None
        result = {name:int(stats[i]) for i,name in enumerate(COUNTERS)}
        if status == 0:
            result.update({name:int(stats[9+i]) for i,name in enumerate(SUCCESS_COUNTERS)})
            result.update(selected_start_attachment=list(cells[0]),selected_goal_attachment=list(cells[-1]))
        result.update(status=STATUSES[status],search_seconds=time.perf_counter()-began,complete_raw_paths_emitted=int(status==0),
            maximum_expanded_nodes=self.maximum_nodes,search_deadline_seconds=self.deadline_seconds,
            native=dict(protocol=PROTOCOL,test_only=self.testing,abi_version=1,
                build_receipt_sha256=self.library.receipt_sha256,library_sha256=self.library.receipt['library_sha256'],
                original_neighbor_table_sha256=self.neighbors_sha256,ctypes_preparation_seconds=prep,
                kernel_seconds=float(kernel[0]),ctypes_preparation_counts_toward_search_deadline=True,
                clock_check_interval_expanded_nodes=64))
        return cells,result


@contextmanager
def native_planner(planner, library):
    """Production context, no CLI/argument for relaxing either search limit."""
    global _ACTIVE
    if _ACTIVE:
        raise RuntimeError('Nested/concurrent native planner context prohibited')
    original = planner.astar_virtual
    adapted = NativeSearch(planner,library)
    _ACTIVE = True
    planner.astar_virtual = adapted
    try:
        yield adapted
    finally:
        planner.astar_virtual = original
        _ACTIVE = False


def native_search_for_testing(planner,library,maximum_nodes=20000,deadline_seconds=-1.):
    return NativeSearch(planner,library,testing=True,maximum_nodes=maximum_nodes,deadline_seconds=deadline_seconds)
