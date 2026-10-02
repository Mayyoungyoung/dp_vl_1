"""Train equally exposed baselines; validation, never test, selects checkpoints."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import balanced_targets, decode_paths, encode_paths, load_data, matching_loss, seed_all, sha256, write_json
from .diffusion import DiffusionSchedule
from .geometry import route_metrics
from .models import RouteDenoiser, SetRegressor


def build_model(config, cond_dim):
    common = dict(cond_dim=cond_dim, horizon=config['horizon'], width=config['width'], depth=config['depth'], heads=4)
    if config['model'] == 'regressor':
        return SetRegressor(max_candidates=config['candidates'], **common)
    return RouteDenoiser(set_attention=config['model'] == 'set_diffusion', **common)


@torch.no_grad()
def predict(model, schedule, condition, scenes, kind, k=3, sample_steps=40, generator=None):
    if kind == 'regressor':
        residual = model(condition, k=k)
    else:
        residual = schedule.sample(model, condition, k=k, steps=sample_steps, generator=generator)
    return decode_paths(residual, scenes)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data/routes.npz')
    p.add_argument('--features')
    p.add_argument('--model', choices=['independent', 'set_diffusion', 'regressor'], required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--steps', type=int, default=4000)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--candidates', type=int, default=3)
    p.add_argument('--width', type=int, default=192)
    p.add_argument('--depth', type=int, default=3)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--eval-every', type=int, default=1000)
    p.add_argument('--sample-steps', type=int, default=40)
    p.add_argument('--cover-weight', type=float, default=0.0)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    torch.set_num_threads(4)
    seed_all(args.seed)
    if args.device.startswith('cuda'):
        assert torch.cuda.is_available(), 'CUDA requested but unavailable'
        torch.cuda.set_per_process_memory_fraction(0.35)
    data = load_data(args.data, args.features)
    args.horizon = int(data['paths'].shape[-2])
    config = vars(args).copy()
    config['cond_dim'] = int(data['condition'].shape[-1])
    config['dataset_sha256'] = sha256(args.data)
    config['features_sha256'] = sha256(args.features) if args.features else None
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'config.json', config)
    model = build_model(config, config['cond_dim']).to(args.device)
    schedule = DiffusionSchedule(steps=100, device=args.device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    train_ids = np.flatnonzero(data['splits'] == 'train')
    val_ids = np.flatnonzero(data['splits'] == 'val')
    assert len(train_ids) and len(val_ids)
    rng = np.random.default_rng(args.seed)
    best = -float('inf')
    start_time = time.perf_counter()
    history = []
    losses = []
    for step in range(1, args.steps+1):
        model.train()
        idx = rng.choice(train_ids, size=args.batch_size, replace=True)
        scenes = torch.as_tensor(data['scenes'][idx], device=args.device)
        condition = torch.as_tensor(data['condition'][idx], device=args.device)
        paths = torch.as_tensor(balanced_targets(data, idx, args.candidates, rng), device=args.device)
        targets = encode_paths(paths, scenes)
        optim.zero_grad(set_to_none=True)
        if args.model == 'regressor':
            pred = model(condition, k=args.candidates)
            loss = matching_loss(pred, targets)
        else:
            t = torch.randint(0, 100, (len(idx),), device=args.device)
            noisy, eps = schedule.q_sample(targets, t)
            pred = model(noisy, t.float()/99, condition)
            loss = (pred-eps).square().mean()
            if args.cover_weight:
                low = t < 25
                if low.any():
                    x0 = schedule.x0_from_eps(noisy, t, pred)
                    # Only low noise. Assignment acts on actual full-route targets.
                    loss = loss + args.cover_weight * matching_loss(x0[low], targets[low])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        losses.append(float(loss.detach()))
        if step % 200 == 0:
            print(json.dumps({'step':step,'loss':float(np.mean(losses[-200:])), 'elapsed_s':round(time.perf_counter()-start_time,2)}), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            predictions = []
            gen = torch.Generator(device=args.device).manual_seed(args.seed+10000)
            for ids in np.array_split(val_ids, max(1, (len(val_ids)+31)//32)):
                c = torch.as_tensor(data['condition'][ids], device=args.device)
                s = torch.as_tensor(data['scenes'][ids], device=args.device)
                predictions.append(predict(model,schedule,c,s,args.model,args.candidates,args.sample_steps,gen).cpu().numpy())
            metrics = route_metrics(np.concatenate(predictions), data['scenes'][val_ids])
            metrics = {key:float(value) for key,value in metrics.items() if np.isscalar(value)}
            score = metrics['unique_valid'] + 0.05 * metrics['valid_rate']
            record = {'step':step,'train_loss':float(np.mean(losses[-200:])), 'validation':metrics}
            history.append(record)
            print(json.dumps(record), flush=True)
            checkpoint = {'model':model.state_dict(), 'config':config, 'step':step, 'validation':metrics}
            torch.save(checkpoint, out/'last.pt')
            if score > best:
                best = score
                torch.save(checkpoint, out/'best.pt')
            write_json(out/'history.json', history)
    if args.device.startswith('cuda'):
        torch.cuda.synchronize()
    summary = {'elapsed_s':time.perf_counter()-start_time, 'trajectory_exposures':args.steps*args.batch_size*args.candidates,
               'optimizer_steps':args.steps,'parameters':sum(x.numel() for x in model.parameters()),
               'active_parameters':model.active_parameter_count(),
               'peak_cuda_memory_mb':torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else 0,
               'best_validation_score':best, 'device':torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}
    write_json(out/'training_summary.json', summary)
    print(json.dumps(summary),flush=True)


if __name__ == '__main__':
    main()
