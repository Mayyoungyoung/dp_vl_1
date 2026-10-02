"""Controlled multi-gate development experiments; locked splits are never used.

Historical SetRegressor is unchanged. The all-positive assignment is a strong
loss control, not a claimed novel method. It selects K positive references via
rectangular matching, avoiding random-subset regression across incompatible
modes. Reference counts are not treated as a general solution-count label.
"""
import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from .common import decode_paths, encode_paths, seed_all, sha256, write_json
from .models import SetRegressor


def positive_assignment_loss(pred, targets, mask, objective, rng):
    """Match only known positives; never supervise unmatched output as absent."""
    b, k = pred.shape[:2]
    costs = (pred[:, :, None] - targets[:, None]).square().mean((-1, -2))
    detached = costs.detach().cpu().numpy()
    selected = []
    for row in range(b):
        ids = np.flatnonzero(mask[row])
        if not len(ids):
            raise ValueError('Every training item needs at least one known positive')
        if objective == 'subset':
            ids = rng.permutation(ids)[:k]
        if objective == 'saturation' and len(ids) < k:
            # Exact minimum-cost assignment that covers every known positive
            # at least once while letting excess candidates use any positive.
            # Randomly duplicating targets would force stochastic multiplicity
            # into deterministic queries and can average incompatible routes.
            small = detached[row][:, ids]
            nearest = small.argmin(1)
            base = small.min(1)
            target_rows, candidate_cols = linear_sum_assignment((small-base[:, None]).T)
            nearest[candidate_cols] = target_rows
            selected.append(costs[row, np.arange(k), ids[nearest]].mean())
            continue
        # If there are fewer types than K, duplicates are legitimate candidates.
        # Cover every known type before adding repeated positive targets.
        if len(ids) < k:
            ids = np.concatenate([ids, rng.choice(ids, k - len(ids), replace=True)])
        rows, cols = linear_sum_assignment(detached[row][:, ids])
        selected.append(costs[row, rows, ids[cols]].mean())
    return torch.stack(selected).mean()


def rng_state(rng):
    return {'numpy_generator': rng.bit_generator.state, 'numpy': np.random.get_state(),
            'python': random.getstate(), 'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state, rng):
    rng.bit_generator.state = state['numpy_generator']
    np.random.set_state(state['numpy'])
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def atomic_checkpoint(path, state):
    temp = path.with_suffix('.tmp')
    torch.save(state, temp)
    os.replace(str(temp), str(path))


def synchronized_time(device):
    if str(device).startswith('cuda'):
        torch.cuda.synchronize()
    return time.perf_counter()


@torch.no_grad()
def evaluate(model, data, ids, k, device, output=None):
    from .multigate import route_metrics
    model.eval()
    predictions = []
    for group in np.array_split(ids, max(1, (len(ids) + 31) // 32)):
        c = torch.as_tensor(data['scenes'][group], device=device)
        predictions.append(decode_paths(model(c, k), c).cpu().numpy())
    predictions = np.concatenate(predictions)
    result = route_metrics(predictions, data['scenes'][ids],
                           reference_modes=data['modes'][ids], reference_mask=data['path_mask'][ids])
    # This is single-request controlled geometry head latency, not VLM latency.
    c = torch.as_tensor(data['scenes'][ids[:1]], device=device)
    for _ in range(5):
        decode_paths(model(c, k), c)
    latencies = []
    for _ in range(30):
        start = synchronized_time(device)
        decode_paths(model(c, k), c)
        latencies.append((synchronized_time(device)-start)*1000)
    scalar = {key: float(value) for key, value in result.items() if np.isscalar(value)}
    scalar.update({'head_batch1_ms_median': float(np.median(latencies)),
                   'head_batch1_ms_p95': float(np.percentile(latencies, 95)),
                   'candidates': k, 'forward_passes': 1})
    if output:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/'predictions.npz', paths=predictions,
                            scene_ids=data['scene_ids'][ids], parent_ids=data['parent_ids'][ids])
        per_scene = []
        for row, idx in enumerate(ids):
            r = route_metrics(predictions[row:row+1], data['scenes'][idx:idx+1],
                              reference_modes=data['modes'][idx:idx+1], reference_mask=data['path_mask'][idx:idx+1])
            r = {key: float(value) for key, value in r.items() if np.isscalar(value)}
            r.update({'scene_id': str(data['scene_ids'][idx]), 'parent_id': str(data['parent_ids'][idx]),
                      'reference_types': int(data['path_mask'][idx].sum())})
            per_scene.append(r)
        write_json(output/'per_scene.json', per_scene)
        write_json(output/'metrics.json', scalar)
    return scalar


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--objective', choices=['subset', 'positive', 'saturation'], default='subset')
    p.add_argument('--steps', type=int, default=3000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--candidates', type=int, default=4)
    p.add_argument('--width', type=int, default=192)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=500)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--stop-after', type=int, help='Checkpointed interruption for resume verification')
    args = p.parse_args()
    from .multigate import load_dataset
    torch.set_num_threads(4)
    if args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(.35)
    seed_all(args.seed)
    data = load_dataset(args.data)
    train_ids = np.flatnonzero(data['splits'] == 'TRAIN')
    dev_ids = np.flatnonzero(data['splits'] == 'DEV_MODEL')
    assert len(train_ids) and len(dev_ids)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:
        raise RuntimeError('Existing checkpoint: use --resume or a new run_id')
    lock = out/'active.lock'
    try:
        fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError('Run lock exists; verify PID before recovering stale lock')
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        config = vars(args).copy()
        config.update({'dataset_sha256': sha256(args.data), 'horizon': int(data['paths'].shape[-2]),
                       'cond_dim': int(data['scenes'].shape[-1]),
                       'code_commit': os.environ.get('CODE_COMMIT', 'unrecorded'),
                       'gpu_uuid': os.environ.get('RESEARCH_GPU_UUID'),
                       'selection_split': 'DEV_MODEL', 'information': 'controlled true geometry and endpoints',
                       'supervision': 'same full positive reference pool; matching objective differs'})
        model = SetRegressor(config['cond_dim'], config['horizon'], args.candidates, args.width, args.depth).to(args.device)
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        # Fixed schedule makes continuation transparent; no gradient scaler in FP32.
        schedule = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.)
        rng = np.random.default_rng(args.seed)
        # Separate sampler prevents different assignment RNG usage changing scenes.
        sample_rng = np.random.default_rng(args.seed + 100000)
        start_step, best, elapsed_before, exposures = 0, -float('inf'), 0., 0
        reference_pool_access = 0
        history = []
        if args.resume:
            ck = torch.load(out/'last.pt', map_location=args.device, weights_only=False)
            for key in ['dataset_sha256', 'objective', 'candidates', 'width', 'depth', 'seed', 'lr', 'batch_size', 'steps']:
                if config[key] != ck['config'][key]:
                    raise ValueError('Resume config mismatch: '+key)
            model.load_state_dict(ck['model']); optim.load_state_dict(ck['optimizer']); schedule.load_state_dict(ck['scheduler'])
            restore_rng(ck['rng'], rng); sample_rng.bit_generator.state = ck['sampler_state']
            start_step, best, elapsed_before, exposures = ck['step'], ck['best'], ck['elapsed_s'], ck['trajectory_exposures']
            reference_pool_access = ck.get('reference_pool_access', 0)
            history = ck['history']
        # Preserve the original config if an incompatible resume is rejected.
        write_json(out/'config.json', config)
        start = synchronized_time(args.device)
        losses = []
        for step in range(start_step+1, args.steps+1):
            model.train()
            idx = sample_rng.choice(train_ids, args.batch_size, replace=True)
            scenes = torch.as_tensor(data['scenes'][idx], device=args.device)
            paths = torch.as_tensor(data['paths'][idx], device=args.device)
            reference_pool_access += int(data['path_mask'][idx].sum())
            target = encode_paths(paths, scenes)
            optim.zero_grad(set_to_none=True)
            loss = positive_assignment_loss(model(scenes), target, data['path_mask'][idx], args.objective, rng)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optim.step(); schedule.step()
            losses.append(float(loss.detach())); exposures += args.batch_size * args.candidates
            if step % 200 == 0:
                print(json.dumps({'step':step,'loss':float(np.mean(losses[-200:])),
                                  'elapsed_s':elapsed_before+synchronized_time(args.device)-start}), flush=True)
            if step % args.eval_every == 0 or step == args.steps or step == args.stop_after:
                metrics = evaluate(model, data, dev_ids, args.candidates, args.device)
                score = metrics.get('unique_valid', metrics.get('UniqueValid@K', 0)) + .05*metrics.get('valid_rate', metrics.get('Valid@K', 0))
                improved = score > best
                best = max(best, score)
                history.append({'step':step,'loss':float(np.mean(losses[-200:])), 'dev_model':metrics})
                elapsed = elapsed_before+synchronized_time(args.device)-start
                checkpoint = {'model':model.state_dict(),'optimizer':optim.state_dict(),'scheduler':schedule.state_dict(),
                              'scaler':None,'step':step,'config':config,'rng':rng_state(rng),'sampler_state':sample_rng.bit_generator.state,
                              'best':best,'history':history,'elapsed_s':elapsed,'trajectory_exposures':exposures,
                              'reference_pool_access':reference_pool_access}
                atomic_checkpoint(out/'last.pt', checkpoint)
                if improved:
                    atomic_checkpoint(out/'best.pt', checkpoint)
                write_json(out/'history.json', history)
                print(json.dumps(history[-1]), flush=True)
                if step == args.stop_after and step < args.steps:
                    write_json(out/'status.json', {'status':'interrupted_for_resume_check','step':step,'exit_code':0})
                    return
        ck = torch.load(out/'best.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(ck['model'])
        metrics = evaluate(model, data, dev_ids, args.candidates, args.device, out/'dev_model')
        elapsed = elapsed_before+synchronized_time(args.device)-start
        summary = {'metrics':metrics,'elapsed_s':elapsed,'gpu_hours_reserved':elapsed/3600 if args.device.startswith('cuda') else 0,
                   'trajectory_exposures':exposures,'positive_pool_references_per_scene':'variable, all available to both methods',
                   'gradient_target_slots':exposures,'reference_pool_access':reference_pool_access,
                   'padded_pairwise_costs_computed':args.steps*args.batch_size*args.candidates*data['paths'].shape[1],
                   'assignment_selectable_references':'random K subset with repeats' if args.objective=='subset' else 'all known positives',
                   'parameters':model.active_parameter_count(),'peak_cuda_memory_mb':torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else 0,
                   'best_step':ck['step'],'best_checkpoint_sha256':sha256(out/'best.pt'),
                   'prediction_sha256':sha256(out/'dev_model'/'predictions.npz')}
        write_json(out/'summary.json', summary)
        write_json(out/'status.json', {'status':'completed','step':args.steps,'exit_code':0})
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', {'status':'failed','exit_code':1,'exception':repr(exc)})
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
