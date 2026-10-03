"""Independent ordinary budget-conditioned controlled baseline.

No historical source is edited. Production identity and budget are fixed by
the registered config. Planned checkpoint resumes are exact; an issued but
unsealed step/evaluation is fail-closed, never silently replayed.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import uuid

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "budget_conditioned_regression_v1"
BUDGETS = (1, 2, 4, 8)
SOURCES = (
    "routeset/budget_conditioned_regression.py", "scripts/train_budget_conditioned_regression.py",
    "routeset/models.py", "routeset/train_v2.py", "routeset/common.py", "routeset/multigate.py", "routeset/geometry.py",
)
DATA_SHA = "f93c8c4b54e44c365d323e4efe0bdf9f6c895fd611a830338971a242d233e057"


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


class _CUuuid(ctypes.Structure):
    _fields_ = [('bytes', ctypes.c_ubyte * 16)]


def cuda_driver_identity(config, library=None):
    """Public CUDA 12.1 metadata; no explicit context/memory/kernel API call.

    CUDA_VISIBLE_DEVICES must be the full authorized UUID before cuInit.
    cuDeviceGetUuid fills the official 16-byte CUuuid structure. A supplied
    library is only an injection point for pure ABI/call-order mock tests.
    """
    if os.environ.get('CUDA_VISIBLE_DEVICES') != config['gpu_visible_device'] or os.environ.get('RESEARCH_GPU_UUID') != config['gpu_uuid']:
        raise ValueError('full authorized GPU UUID environment required')
    if config['gpu_visible_device'] != config['gpu_uuid']:
        raise ValueError('visible device must be the full fixed UUID, not an ordinal')
    if library is None:
        if not sys.platform.startswith('linux'):
            raise ValueError('actual Driver metadata is restricted to authorized Linux deployment')
        library = ctypes.CDLL('libcuda.so.1')
    signatures = {
        'cuInit': [ctypes.c_uint],
        'cuDeviceGetCount': [ctypes.POINTER(ctypes.c_int)],
        'cuDeviceGet': [ctypes.POINTER(ctypes.c_int), ctypes.c_int],
        'cuDeviceGetUuid': [ctypes.POINTER(_CUuuid), ctypes.c_int],
    }
    functions = {}
    for name, arguments in signatures.items():
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, ctypes.c_int
        functions[name] = function
    calls = []
    def checked(name, *args):
        code = int(functions[name](*args))
        calls.append(dict(api=name, result=code))
        if code != 0:
            raise RuntimeError('CUDA Driver %s returned CUresult %d' % (name, code))
    checked('cuInit', 0)
    count = ctypes.c_int()
    checked('cuDeviceGetCount', ctypes.byref(count))
    if count.value != 1:
        raise ValueError('CUDA Driver must expose exactly one authorized device')
    device, value = ctypes.c_int(), _CUuuid()
    checked('cuDeviceGet', ctypes.byref(device), 0)
    checked('cuDeviceGetUuid', ctypes.byref(value), device.value)
    actual = 'GPU-'+str(uuid.UUID(bytes=bytes(value.bytes)))
    if actual != config['gpu_uuid']:
        raise ValueError('actual CUDA Driver UUID mismatch')
    return dict(visible_index=0, visible_count=count.value, actual_uuid=actual,
                authority='libcuda.so.1 public cuInit/cuDeviceGetCount/cuDeviceGet/cuDeviceGetUuid',
                driver_calls=calls, explicit_context_create_calls=0, model_forwards=0, kernel_launch_calls=0)


def verify_cuda_device(torch_api, config):
    """Driver proves UUID even on Torch 2.4.1, which exposes no uuid field."""
    driver = cuda_driver_identity(config)
    if torch_api.cuda.device_count() != 1:
        raise ValueError('exactly one authorized visible CUDA device required')
    props = torch_api.cuda.get_device_properties(0)
    torch_api.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'], device=0)
    torch_api.cuda.reset_peak_memory_stats(0)
    return dict(visible_index=0, actual_uuid=driver['actual_uuid'], driver_metadata=driver, name=props.name, total_memory_bytes=int(props.total_memory),
                memory_fraction_set=config['gpu_memory_fraction'], configured_allocator_cap_bytes=int(props.total_memory*config['gpu_memory_fraction']))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    def numpy_value(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError('unsupported JSON type: '+type(obj).__name__)
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=numpy_value)+'\n', encoding='utf-8')
    os.replace(str(temp), str(path))


def k_for_step(step):
    if type(step) is not int or step < 1:
        raise ValueError('one-based integer step required')
    return BUDGETS[(step-1) % len(BUDGETS)]


def budget_counts(steps, batch):
    if steps < 0 or batch < 1:
        raise ValueError('invalid budget')
    by_k = {str(k): sum(k_for_step(i) == k for i in range(1, steps+1))*batch for k in BUDGETS}
    return dict(requests=steps*batch, slots=sum(int(k)*count for k, count in by_k.items()), requests_by_k=by_k)


def selection_score(metrics):
    if set(metrics) != set(map(str, BUDGETS)):
        raise ValueError('all four independently generated K results required')
    return float(np.mean([metrics[str(k)]['unique_valid']/k + .05*metrics[str(k)]['valid_rate'] for k in BUDGETS]))


def validate_config(config):
    exact = dict(protocol=PROTOCOL, data='data/multigate_v1_partitions/development.npz', dataset_sha256=DATA_SHA,
                 train_parents=768, dev_parents=128, geometry_dim=34, horizon=24, max_references=16,
                 budgets=list(BUDGETS), steps=3000, batch_size=64, width=192, depth=3, heads=4,
                 lr=3e-4, weight_decay=1e-4, gradient_clip_norm=1., seed=0, sample_seed=100000,
                 objective='saturation', schedule='constant', eval_every=500, checkpoint_every=250,
                 selection='mean_over_K(unique_valid/K + 0.05*valid_rate)', train_requests=192000,
                 train_output_slots=720000, selection_requests=3072, selection_output_slots=11520,
                 selection_opportunities=6, latency_warmups=2, latency_timed=10,
                 latency_parent_id='multigate_v1_TRAIN_00000', maximum_timing_output_slots=360,
                 maximum_total_output_slots=731880, cpu_threads=1, gpu_visible_device='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab',
                 gpu_uuid='GPU-7506746b-d0ba-f6fe-44ce-8a1f97dde2ab', gpu_memory_fraction=.35,
                 final_dev_policy='Reuse sealed original selection pools by checkpoint, no new DEV forward')
    for key, value in exact.items():
        if key not in config or config[key] != value:
            raise ValueError('registered config mismatch: '+key)
    counts = budget_counts(config['steps'], config['batch_size'])
    if counts['slots'] != config['train_output_slots'] or counts['requests'] != config['train_requests']:
        raise ValueError('budget arithmetic mismatch')
    return config


def load_development(path, expected_sha=DATA_SHA):
    if digest(path) != expected_sha:
        raise ValueError('development bytes mismatch')
    with np.load(path, allow_pickle=False) as archive:
        # No geometry/path labels are decoded until the role boundary is checked.
        metadata = {key: archive[key] for key in ('splits', 'parent_ids', 'scene_ids')}
        roles = metadata['splits']
        if roles.ndim != 1 or set(roles.tolist()) != {'TRAIN', 'DEV_MODEL'}:
            raise ValueError('physically separated TRAIN/DEV_MODEL archive required')
        if any(array.shape != (896,) for array in metadata.values()):
            raise ValueError('registered metadata shapes required')
        expected = {'TRAIN': 768, 'DEV_MODEL': 128}
        for role, count in expected.items():
            ids = np.flatnonzero(roles == role)
            parents = ['multigate_v1_%s_%05d' % (role, i) for i in range(count)]
            if metadata['parent_ids'][ids].tolist() != parents or metadata['scene_ids'][ids].tolist() != [p+'_c00' for p in parents]:
                raise ValueError('registered complete role identity required')
        data = dict(metadata)
        for key in ('scenes', 'paths', 'path_mask', 'modes'):
            data[key] = archive[key]
    if (data['scenes'].shape != (896, 34) or data['paths'].shape != (896, 16, 24, 3)
            or data['path_mask'].shape != (896, 16) or data['path_mask'].dtype != np.bool_
            or data['modes'].shape != (896, 16) or not data['path_mask'].any(1).all()):
        raise ValueError('registered positive pool shape required')
    return data


def probe_cuda_metadata(config_path, output):
    """Standalone zero-model metadata gate, deliberately never importing Torch."""
    config_path, output = Path(config_path), Path(output)
    if output.exists():
        raise ValueError('fresh metadata probe output required')
    started = time.perf_counter()
    result = dict(protocol=PROTOCOL+'_cuda_metadata', pid=os.getpid(), model_forwards=0,
                  optimizer_steps=0, dataset_payload_reads=0, configured_memory_fraction_applied=False,
                  source_sha256=digest(Path(__file__)), config_sha256=digest(config_path))
    try:
        config = validate_config(json.loads(config_path.read_text(encoding='utf-8')))
        result['actual_device'] = cuda_driver_identity(config)
        result.update(status='completed', exit_code=0)
    except BaseException as exc:
        result.update(status='failed', exit_code=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        result['elapsed_wall_s'] = time.perf_counter()-started
        atomic_json(output, result)
    print(json.dumps(result, allow_nan=False))
    return result


def new_stream():
    return dict(chain=hashlib.sha256(b'budget-conditioned-parent-k-chain-v1').hexdigest(), steps=0, requests=0, slots=0)


def advance_stream(stream, step, k, indices):
    if step != stream['steps']+1 or k != k_for_step(step):
        raise ValueError('stream step/K mismatch')
    raw = np.asarray([step, k]+list(map(int, indices)), dtype='<i8').tobytes()
    return dict(chain=hashlib.sha256(bytes.fromhex(stream['chain'])+raw).hexdigest(), steps=step,
                requests=stream['requests']+len(indices), slots=stream['slots']+len(indices)*k)


def state_hash(state_dict):
    h = hashlib.sha256()
    for name, value in sorted(state_dict.items()):
        array = value.detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(array.dtype).encode()); h.update(str(array.shape).encode()); h.update(array.tobytes())
    return h.hexdigest()


class Ledger:
    def __init__(self, path):
        self.path = Path(path)
        self.events = []
        if self.path.exists():
            for line in self.path.read_text(encoding='utf-8').splitlines():
                self.events.append(json.loads(line))

    def add(self, kind, **fields):
        event = dict(event=len(self.events), kind=kind, **fields)
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(canonical(event)+'\n'); handle.flush(); os.fsync(handle.fileno())
        self.events.append(event)
        return event['event']

    def sha(self):
        return digest(self.path) if self.path.exists() else hashlib.sha256(b'').hexdigest()

    def verify_checkpoint(self, checkpoint):
        if len(self.events) != checkpoint['ledger_events'] or self.sha() != checkpoint['ledger_sha256']:
            raise ValueError('ledger extends or differs from last checkpoint; preserve partial work, no automatic replay')


def sealed(path):
    """Completed prediction pools can be reused; incomplete pools cannot replay."""
    path = Path(path)
    receipt = path / 'receipt.json'
    if not receipt.exists():
        raise ValueError('issued output is unsealed; refuse extra forward')
    result = json.loads(receipt.read_text(encoding='utf-8'))
    for name, expected in result['files_sha256'].items():
        if Path(name).name != name or digest(path/name) != expected:
            raise ValueError('sealed prediction pool hash mismatch')
    return result


def model_step(model, optimizer, schedule, data, train_ids, config, state, rng, sample_rng, device, before=None, after=None):
    """One ordinary saturation optimization step, also used by tiny tests."""
    import torch
    from routeset.common import encode_paths
    from routeset.train_v2 import positive_assignment_loss
    step = state['step']+1
    k = k_for_step(step)
    indices = sample_rng.choice(train_ids, config['batch_size'], replace=True)
    if before:
        before(step, k, indices)
    model.train()
    scenes = torch.as_tensor(data['scenes'][indices], device=device)
    target = encode_paths(torch.as_tensor(data['paths'][indices], device=device), scenes)
    optimizer.zero_grad(set_to_none=True)
    loss = positive_assignment_loss(model(scenes, k), target, data['path_mask'][indices], 'saturation', rng)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
    optimizer.step(); schedule.step()
    state['step'] = step
    state['stream'] = advance_stream(state['stream'], step, k, indices)
    reference_counts = data['path_mask'][indices].sum(1)
    state['reference_pool_access'] += int(reference_counts.sum())
    previous = state.get('target_count_chain', hashlib.sha256(b'budget-conditioned-target-counts-v1').hexdigest())
    target_bytes = np.asarray([step, k]+reference_counts.tolist(), dtype='<i8').tobytes()
    state['target_count_chain'] = hashlib.sha256(bytes.fromhex(previous)+target_bytes).hexdigest()
    state['loss_window'] = (state['loss_window']+[float(loss.detach())])[-200:]
    if after:
        after(step, k, indices, float(loss.detach()), float(norm))
    return indices


def save_pool(model, data, ids, k, device, output, ledger, step):
    import torch
    from routeset.common import decode_paths
    from routeset.multigate import route_metrics
    output = Path(output)
    if output.exists():
        receipt = sealed(output)
        if receipt['step'] != step or receipt['k'] != k or receipt['model_sha256'] != state_hash(model.state_dict()):
            raise ValueError('sealed pool belongs to another checkpoint')
        return json.loads((output/'metrics.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True)
    ledger.add('selection_issued', step=step, k=k, requests=len(ids), slots=len(ids)*k)
    started = time.perf_counter()
    model.eval()
    predictions = []
    with torch.no_grad():
        for first in range(0, len(ids), 32):
            cond = torch.as_tensor(data['scenes'][ids[first:first+32]], device=device)
            predictions.append(decode_paths(model(cond, k), cond).cpu().numpy())
    predictions = np.concatenate(predictions)
    # Save actual pool first; geometry/reference labels never enter model forward.
    np.savez_compressed(output/'predictions.npz', paths=predictions, parent_ids=data['parent_ids'][ids], scene_ids=data['scene_ids'][ids])
    pool_sha = digest(output/'predictions.npz')
    metrics = route_metrics(predictions, data['scenes'][ids], reference_modes=data['modes'][ids], reference_mask=data['path_mask'][ids])
    per_scene = metrics.pop('per_scene')
    known_valid = per_scene['valid'] & (per_scene['modes'] >= 0)
    metrics['duplicate_valid'] = float(np.mean(known_valid.sum(1)-per_scene['unique_count']))
    metrics['selected_valid'] = None  # no learned selector in this ordinary generator
    atomic_json(output/'per_scene.json', dict(parent_ids=data['parent_ids'][ids], scene_ids=data['scene_ids'][ids], **per_scene))
    atomic_json(output/'metrics.json', metrics)
    receipt = dict(step=step, k=k, requests=len(ids), slots=len(ids)*k, model_sha256=state_hash(model.state_dict()),
                   prediction_sealed_before_label_evaluation_sha256=pool_sha,
                   elapsed_wall_s=time.perf_counter()-started,
                   files_sha256={n:digest(output/n) for n in ('predictions.npz', 'metrics.json', 'per_scene.json')})
    atomic_json(output/'receipt.json', receipt)
    ledger.add('selection_completed', step=step, k=k, requests=len(ids), slots=len(ids)*k, receipt_sha256=digest(output/'receipt.json'))
    return metrics


def measure_latency(model, scene, config, device, output, ledger, step):
    import torch
    from routeset.common import decode_paths
    from routeset.multigate import path_validity
    output = Path(output)
    if output.exists():
        receipt = sealed(output)
        if receipt['step'] != step or receipt['model_sha256'] != state_hash(model.state_dict()):
            raise ValueError('timing pool belongs to another checkpoint')
        return receipt
    output.mkdir(parents=True)
    model.eval()
    results, times = {}, {}
    with torch.no_grad():
        for k in BUDGETS:
            arrays, elapsed = [], []
            for iteration in range(config['latency_warmups']+config['latency_timed']):
                ledger.add('latency_issued', step=step, k=k, iteration=iteration, requests=1, slots=k)
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                condition = torch.as_tensor(scene[None], device=device)
                paths = decode_paths(model(condition, k), condition).cpu().numpy()[0]
                path_validity(paths, scene)  # controlled exact checker, same input geometry
                if device.startswith('cuda'):
                    torch.cuda.synchronize()
                duration = time.perf_counter()-start
                arrays.append(paths); elapsed.append(duration)
                ledger.add('latency_completed', step=step, k=k, iteration=iteration, requests=1, slots=k, elapsed_s=duration)
            results['k%d' % k] = np.stack(arrays)
            timed = elapsed[config['latency_warmups']:]
            times[str(k)] = dict(all_call_s=elapsed, warmup_count=config['latency_warmups'], median_s=float(np.median(timed)),
                                 mean_s=float(np.mean(timed)), timed_count=len(timed))
    np.savez_compressed(output/'predictions.npz', **results)
    atomic_json(output/'times.json', times)
    result = dict(step=step, model_sha256=state_hash(model.state_dict()),
                  requests=(config['latency_warmups']+config['latency_timed'])*4,
                  slots=(config['latency_warmups']+config['latency_timed'])*sum(BUDGETS),
                  scope='single CPU input transfer + head + decode + CPU transfer + controlled deterministic validity; excludes offline reference metrics; no VLM exists in tier A',
                  files_sha256={n:digest(output/n) for n in ('predictions.npz', 'times.json')})
    atomic_json(output/'receipt.json', result)
    return result


def run(config_path, output, device='cuda', resume=False, stop_after=None):
    overall_start = time.perf_counter()
    import torch
    from routeset.budget_conditioned_regression import BudgetConditionedRegressor
    from routeset.common import seed_all
    from routeset.train_v2 import atomic_checkpoint, rng_state, restore_rng
    config_path, output = Path(config_path), Path(output)
    config = validate_config(json.loads(config_path.read_text(encoding='utf-8')))
    if stop_after is not None and (not 1 <= stop_after <= config['steps']):
        raise ValueError('stop-after outside registered steps')
    if device not in ('cpu', 'cuda'):
        raise ValueError('only explicit cpu or authorized visible cuda accepted')
    torch.set_num_threads(1)
    device_info = dict(device='cpu')
    if device == 'cuda':
        device_info = verify_cuda_device(torch, config)
    if not re.fullmatch(r'[0-9a-f]{40}', os.environ.get('CODE_COMMIT', '')):
        raise ValueError('immutable CODE_COMMIT required')
    source = {name:digest(ROOT/name) for name in SOURCES}
    identity = dict(config_sha256=digest(config_path), source_sha256=source, device=device,
                    dataset_sha256=config['dataset_sha256'], source_commit=os.environ['CODE_COMMIT'])
    data = load_development(ROOT/config['data'])
    train_ids, dev_ids = [np.flatnonzero(data['splits'] == role) for role in ('TRAIN','DEV_MODEL')]
    if output.exists() != resume:
        raise ValueError('fresh output required, or explicitly resume an existing run')
    output.mkdir(parents=True, exist_ok=resume)
    lock = output/'active.lock'
    fd = os.open(str(lock), os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    status, started = dict(status='running', pid=os.getpid(), exit_code=None, actual_device=device_info), overall_start
    session_id = '%s_%d_%d' % (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), os.getpid(), time.time_ns())
    try:
        if (output/'summary.json').exists():
            if not resume:
                raise ValueError('completed run exists')
            complete = json.loads((output/'summary.json').read_text(encoding='utf-8'))
            if complete['identity'] != identity:
                raise ValueError('completed source/config identity mismatch')
            for path, expected in complete['artifact_sha256'].items():
                if digest(output/path) != expected:
                    raise ValueError('completed artifact changed')
            status.update(status='completed_reused_no_calls', exit_code=0)
            return complete
        seed_all(config['seed'])
        model = BudgetConditionedRegressor(width=config['width'], depth=config['depth'], heads=config['heads']).to(device)
        initial_sha = state_hash(model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        rng, sample_rng = np.random.default_rng(config['seed']), np.random.default_rng(config['sample_seed'])
        state = dict(step=0, stream=new_stream(), loss_window=[], reference_pool_access=0, history=[], best_score=-1., best_step=None, elapsed_s=0.)
        state['target_count_chain'] = hashlib.sha256(b'budget-conditioned-target-counts-v1').hexdigest()
        ledger = Ledger(output/'calls.jsonl')
        if resume:
            checkpoint = torch.load(output/'last.pt', map_location=device, weights_only=False)
            if 'scaler' not in checkpoint or checkpoint['scaler'] is not None or 'target_count_chain' not in checkpoint['state']:
                raise ValueError('required checkpoint state is missing')
            if checkpoint['identity'] != identity or checkpoint['initial_model_sha256'] != initial_sha:
                raise ValueError('resume identity/source/initial model mismatch')
            if checkpoint['state']['step'] < config['steps']:
                ledger.verify_checkpoint(checkpoint)
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer']); schedule.load_state_dict(checkpoint['scheduler'])
            restore_rng(checkpoint['rng'], rng); sample_rng.bit_generator.state = checkpoint['sampler_state']
            state = checkpoint['state']
        else:
            atomic_json(output/'config.json', dict(config=config, identity=identity, initial_model_sha256=initial_sha))

        def checkpoint_now():
            record = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=schedule.state_dict(), scaler=None,
                          rng=rng_state(rng), sampler_state=sample_rng.bit_generator.state, state=state,
                          identity=identity, config=config, initial_model_sha256=initial_sha,
                          ledger_events=len(ledger.events), ledger_sha256=ledger.sha())
            atomic_checkpoint(output/'last.pt', record)
            if state['best_step'] == state['step']:
                atomic_checkpoint(output/'best.pt', record)
            atomic_json(output/'history.json', state['history'])

        train_start, elapsed_before = time.perf_counter(), state['elapsed_s']
        for _ in range(state['step'], config['steps']):
            model_step(model, optimizer, schedule, data, train_ids, config, state, rng, sample_rng, device,
                before=lambda step,k,idx:ledger.add('train_issued', step=step, k=k, requests=len(idx), slots=len(idx)*k, parent_indices=idx.tolist(),
                    positive_reference_counts=data['path_mask'][idx].sum(1).tolist(), padded_cost_entries=len(idx)*k*config['max_references']),
                after=lambda step,k,idx,loss,norm:ledger.add('train_completed', step=step, k=k, requests=len(idx), slots=len(idx)*k, loss=loss, gradient_norm=norm))
            step = state['step']
            if step % config['eval_every'] == 0:
                metrics = {str(k):save_pool(model, data, dev_ids, k, device, output/'selection'/('step%04d'%step)/('k%d'%k), ledger, step) for k in BUDGETS}
                score = selection_score(metrics)
                if score > state['best_score']:
                    state['best_score'], state['best_step'] = score, step
                state['history'].append(dict(step=step, loss=float(np.mean(state['loss_window'])), score=score, metrics=metrics))
            if step % config['checkpoint_every'] == 0 or step == stop_after or step == config['steps']:
                state['elapsed_s'] = elapsed_before+time.perf_counter()-train_start
                checkpoint_now()
                print(json.dumps(dict(step=step, train_slots=state['stream']['slots'], loss=float(np.mean(state['loss_window'])), best_step=state['best_step'])), flush=True)
            if step == stop_after and step < config['steps']:
                status.update(status='paused_at_checkpoint', exit_code=0, step=step)
                return None
        # Reuse the sealed original selection pools. No duplicate final DEV calls.
        final = {}
        for label, step in (('best', state['best_step']), ('last', config['steps'])):
            pools = {}
            for k in BUDGETS:
                path = output/'selection'/('step%04d'%step)/('k%d'%k)
                receipt = sealed(path)
                pools[str(k)] = dict(path=str(path.relative_to(output)), receipt_sha256=digest(path/'receipt.json'), prediction_sha256=receipt['files_sha256']['predictions.npz'])
            ck = torch.load(output/('%s.pt'%label), map_location=device, weights_only=False)
            if ck['state']['step'] != step:
                raise ValueError('final checkpoint selection step mismatch')
            model.load_state_dict(ck['model'])
            for k in BUDGETS:
                if sealed(output/pools[str(k)]['path'])['model_sha256'] != state_hash(model.state_dict()):
                    raise ValueError('selected prediction pool/weights differ')
            timing_path = output/'timing'/('step%04d'%step)
            timing = measure_latency(model, data['scenes'][train_ids[0]], config, device, timing_path, ledger, step)
            final[label] = dict(step=step, checkpoint_sha256=digest(output/('%s.pt'%label)), pools=pools,
                                timing_path=str(timing_path.relative_to(output)), timing_slots=timing['slots'])
        actual = {kind:dict(events=sum(e['kind']==kind for e in ledger.events),
                    requests=sum(e.get('requests',0) for e in ledger.events if e['kind']==kind),
                    slots=sum(e.get('slots',0) for e in ledger.events if e['kind']==kind))
                  for kind in ('train_issued','train_completed','selection_issued','selection_completed','latency_issued','latency_completed')}
        for prefix in ('train','selection','latency'):
            if actual[prefix+'_issued'] != actual[prefix+'_completed']:
                raise ValueError('unclosed issued budget remains')
        if (actual['train_issued']['slots'] != config['train_output_slots']
                or actual['selection_issued']['slots'] != config['selection_output_slots']
                or actual['latency_issued']['slots'] > config['maximum_timing_output_slots']):
            raise ValueError('actual budget differs from registered schedule')
        artifacts = {str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name not in ('active.lock','status.json','summary.json')}
        summary = dict(protocol=PROTOCOL, identity=identity, config=config, final=final, actual_budget=actual,
                       stream=state['stream'], initial_model_sha256=initial_sha, reference_pool_access=state['reference_pool_access'],
                       target_count_chain=state['target_count_chain'],
                       padded_pairwise_costs_computed=state['stream']['slots']*16,
                       training_and_selection_elapsed_s=state['elapsed_s'],
                       training_and_selection_gpu_reserved_hours=state['elapsed_s']/3600 if device=='cuda' else 0,
                       finalization_this_process_s=time.perf_counter()-train_start-(state['elapsed_s']-elapsed_before),
                       best_step=state['best_step'], peak_memory_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                       peak_memory_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0,
                       artifact_sha256=artifacts, training_claim='ordinary baseline, no innovation or K4-prefix proxy')
        atomic_json(output/'summary.json', summary)
        status.update(status='completed', exit_code=0, step=config['steps'])
        return summary
    except BaseException as exc:
        status.update(status='failed', exit_code=1, error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        status.update(session_elapsed_s=time.perf_counter()-started)
        status.update(device=device, session_id=session_id, session_gpu_reserved_hours=status['session_elapsed_s']/3600 if device=='cuda' else 0,
                      cost_scope='Sum inner sessions and sum external record_job wrappers separately. Outer contains inner; choose one level for total cost, never add both.')
        status.update(peak_memory_allocated_bytes=torch.cuda.max_memory_allocated() if device=='cuda' else 0,
                      peak_memory_reserved_bytes=torch.cuda.max_memory_reserved() if device=='cuda' else 0)
        atomic_json(output/'sessions'/(session_id+'.json'), status)
        atomic_json(output/'status.json', status)
        lock.unlink()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--probe-cuda-metadata', action='store_true', help='Only public Driver UUID metadata, no Torch/model/data; output is a fresh JSON file')
    args = parser.parse_args()
    if args.probe_cuda_metadata:
        if args.resume or args.stop_after is not None:
            parser.error('metadata probe cannot resume or specify training steps')
        probe_cuda_metadata(args.config, args.output)
    else:
        run(args.config, args.output, args.device, args.resume, args.stop_after)
