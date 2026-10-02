"""Learn and calibrate toy task-checker confidence, with strict scene splits.

This estimates the probability that a route satisfies routeset.geometry's toy
task-level checker. It is not a probability of robot execution success. Training
uses only TRAIN routes, calibration uses the first half of VAL scenes, and the
remaining VAL scenes assess calibration before final TEST/OOD reporting.

Example:
  python scripts/train_confidence.py --data data/routes.npz \
    --prediction-dirs runs/independent/eval runs/set_diffusion/eval \
    --output runs/confidence --device cuda
"""

import argparse
import hashlib
import inspect
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from routeset.common import load_data, seed_all, sha256, write_json
from routeset.geometry import path_validity
from routeset.models import Critic


SCOPE = 'Probability of satisfying the synthetic task-layer geometric checker; not robot execution success.'


def residuals(paths, scene_indices, data):
    scenes = data['scenes'][scene_indices]
    time = np.linspace(0, 1, paths.shape[-2], dtype=np.float32)[None, :, None]
    line = scenes[:, None, :3] * (1 - time) + scenes[:, None, 3:6] * time
    return (paths[:, 1:-1] - line[:, 1:-1]).astype(np.float32)


def label_paths(paths, scene_indices, data):
    """Group paths by scene, re-evaluating labels instead of trusting saved valid."""
    result = np.zeros(len(paths), dtype=np.float32)
    for scene_index in np.unique(scene_indices):
        mask = scene_indices == scene_index
        result[mask] = np.asarray(path_validity(paths[mask], data['scenes'][scene_index])['valid'], dtype=np.float32)
    return result


def synthetic_examples(data, scene_indices, rng, count=32):
    """Positive references and several realistic interior-only perturbations."""
    all_paths, all_indices = [], []
    horizon = data['paths'].shape[-2]
    time = np.linspace(0, 1, horizon, dtype=np.float32)[:, None]
    envelope = np.sin(np.pi * time)
    for scene_index in scene_indices:
        scene = data['scenes'][scene_index]
        refs = data['paths'][scene_index].astype(np.float32)
        line = scene[:3][None] * (1 - time) + scene[3:6][None] * time
        paths = [route.copy() for route in refs]
        paths.append(line.copy())
        modes = data['modes'][scene_index]
        for number in range(count):
            first = int(rng.integers(len(refs)))
            variant = number % 5
            route = refs[first].copy()
            if variant == 0:
                # Smooth low-frequency changes resemble imperfect model routes.
                scale = float(rng.uniform(0.015, 0.65))
                coefficients = rng.normal(size=(3, 3)).astype(np.float32) * scale
                route += sum(np.sin(math.pi * frequency * time) * coefficients[frequency - 1]
                             for frequency in range(1, 4))
            elif variant == 1:
                others = np.flatnonzero(modes != modes[first])
                second = int(rng.choice(others)) if len(others) else int(rng.integers(len(refs)))
                blend = float(rng.uniform(0.1, 0.9))
                route = blend * route + (1 - blend) * refs[second]
            elif variant == 2:
                # Shortcut toward the line, often crossing an obstacle.
                blend = float(rng.uniform(0.0, 1.0))
                route = blend * route + (1 - blend) * line
            elif variant == 3:
                route += envelope * rng.normal(0, float(rng.uniform(0.02, 0.3)), size=route.shape).astype(np.float32)
            else:
                # Whole-route offset with a smooth endpoint-preserving envelope.
                route += envelope * rng.normal(0, float(rng.uniform(0.01, 0.8)), size=(1, 3)).astype(np.float32)
            route[0] = scene[:3]
            route[-1] = scene[3:6]
            paths.append(route)
        all_paths.extend(paths)
        all_indices.extend([scene_index] * len(paths))
    paths = np.asarray(all_paths, dtype=np.float32)
    indices = np.asarray(all_indices, dtype=np.int64)
    return paths, indices, label_paths(paths, indices, data)


def load_prediction_sources(directories, data):
    sources = []
    names = set()
    for directory in directories:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError('prediction directory does not exist: ' + str(directory))
        display_name = directory.parent.name if directory.name in ('evaluation', 'eval', 'predictions') else directory.name
        name = re.sub(r'[^A-Za-z0-9_.-]+', '_', display_name)
        if name in names:
            name += '_' + hashlib.sha256(str(directory.resolve()).encode()).hexdigest()[:8]
        names.add(name)
        found = sorted(directory.glob('*_predictions.npz'))
        if not found:
            raise ValueError('no *_predictions.npz found in ' + str(directory))
        for path in found:
            declared_split = path.name.split('_predictions.npz')[0]
            if declared_split not in ('train', 'val', 'test', 'ood'):
                continue
            with np.load(path, allow_pickle=False) as archive:
                paths = archive['paths'].astype(np.float32)
                indices = archive['indices'].astype(np.int64)
                saved_scene_ids = archive['scene_ids']
            if paths.ndim != 5 or paths.shape[0] != len(indices) or paths.shape[-2:] != data['paths'].shape[-2:]:
                raise ValueError('invalid prediction shape in ' + str(path))
            if len(indices) == 0 or np.any(indices < 0) or np.any(indices >= len(data['scenes'])):
                raise ValueError('invalid scene indices in ' + str(path))
            if not np.array_equal(saved_scene_ids, data['scene_ids'][indices]):
                raise ValueError('prediction scene IDs do not match dataset: ' + str(path))
            if not np.all(data['splits'][indices] == declared_split):
                raise ValueError('prediction filename and actual scene splits differ: ' + str(path))
            flat_indices = np.repeat(indices, paths.shape[1] * paths.shape[2])
            flat_paths = paths.reshape(-1, paths.shape[-2], 3)
            labels = label_paths(flat_paths, flat_indices, data)
            sources.append(dict(method=name, source=str(path.resolve()), split=declared_split,
                                paths=flat_paths, indices=flat_indices, labels=labels,
                                scene_indices=indices, original_shape=paths.shape[:3]))
    return sources


def examples_for_sources(sources, accepted_scenes):
    paths, indices, labels = [], [], []
    for source in sources:
        selected = np.isin(source['indices'], accepted_scenes)
        if selected.any():
            paths.append(source['paths'][selected])
            indices.append(source['indices'][selected])
            labels.append(source['labels'][selected])
    if not paths:
        return None
    return np.concatenate(paths), np.concatenate(indices), np.concatenate(labels)


@torch.no_grad()
def predict_logits(critic, encoded, indices, data, device, batch_size=512):
    critic.eval()
    output = []
    for begin in range(0, len(encoded), batch_size):
        end = begin + batch_size
        x = torch.as_tensor(encoded[begin:end, None], device=device)
        condition = torch.as_tensor(data['condition'][indices[begin:end]], device=device)
        output.append(critic(x, condition).squeeze(1).cpu().numpy())
    return np.concatenate(output)


def fit_platt(logits, labels):
    """Fit one affine logistic calibrator using calibration scenes only."""
    if np.unique(labels).size == 1:
        # Finite Laplace-smoothed estimate when the checker labels one class.
        positive = float(labels.sum())
        return 0.0, math.log((positive + 1) / (len(labels) - positive + 1))
    x = torch.as_tensor(logits, dtype=torch.float64)
    y = torch.as_tensor(labels, dtype=torch.float64)
    parameters = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float64))
    optimizer = torch.optim.LBFGS([parameters], lr=0.5, max_iter=100, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        affine = parameters[0] * x + parameters[1]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(affine, y)
        loss = loss + 1e-4 * ((parameters[0] - 1).square() + parameters[1].square())
        loss.backward()
        return loss

    optimizer.step(closure)
    if not torch.isfinite(parameters).all():
        raise RuntimeError('nonfinite Platt calibration parameters')
    return float(parameters[0].detach()), float(parameters[1].detach())


def probabilities(logits, slope=1.0, bias=0.0):
    affine = np.clip(slope * logits.astype(np.float64) + bias, -50, 50)
    return (1 / (1 + np.exp(-affine))).astype(np.float32)


def reliability(probability, labels, bins=10):
    if len(labels) == 0:
        return {'n': 0}
    assignments = np.minimum((probability * bins).astype(np.int64), bins - 1)
    details = []
    ece = 0.0
    for bin_index in range(bins):
        selected = assignments == bin_index
        if not selected.any():
            continue
        mean_probability = float(probability[selected].mean())
        valid_fraction = float(labels[selected].mean())
        count = int(selected.sum())
        ece += count / len(labels) * abs(mean_probability - valid_fraction)
        details.append({'bin': bin_index, 'n': count, 'mean_confidence': mean_probability,
                        'valid_fraction': valid_fraction})
    clipped = np.clip(probability, 1e-6, 1 - 1e-6)
    return {'n': len(labels), 'valid_rate': float(labels.mean()),
            'mean_confidence': float(probability.mean()),
            'brier': float(np.mean((probability - labels) ** 2)), 'ece_10_uniform': float(ece),
            'binary_nll': float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))),
            'reliability_bins': details}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--features')
    parser.add_argument('--prediction-dirs', nargs='+', default=[])
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--synthetic-per-scene', type=int, default=32)
    parser.add_argument('--width', type=int, default=192)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=61)
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 2:
        raise ValueError('positive steps and batch_size >= 2 required')
    torch.set_num_threads(4)
    seed_all(args.seed)
    data = load_data(args.data, args.features)
    train_ids = np.flatnonzero(data['splits'] == 'train')
    validation_ids = np.flatnonzero(data['splits'] == 'val')
    if len(train_ids) == 0 or len(validation_ids) < 2:
        raise ValueError('TRAIN and at least two VAL scenes are required')
    midpoint = len(validation_ids) // 2
    calibration_ids = validation_ids[:midpoint]
    holdout_val_ids = validation_ids[midpoint:]
    sources = load_prediction_sources(args.prediction_dirs, data)
    rng = np.random.default_rng(args.seed)
    train_paths, train_indices, train_labels = synthetic_examples(data, train_ids, rng, args.synthetic_per_scene)
    generated_train = examples_for_sources(sources, train_ids)
    if generated_train is not None:
        train_paths = np.concatenate((train_paths, generated_train[0]))
        train_indices = np.concatenate((train_indices, generated_train[1]))
        train_labels = np.concatenate((train_labels, generated_train[2]))
    encoded_train = residuals(train_paths, train_indices, data)
    del train_paths
    classes = [np.flatnonzero(train_labels == label) for label in (0, 1)]
    if not all(len(indices) for indices in classes):
        raise ValueError('training checker labels must include valid and invalid routes')
    critic = Critic(cond_dim=data['condition'].shape[-1], horizon=data['paths'].shape[-2], width=args.width).to(args.device)
    optimizer = torch.optim.AdamW(critic.parameters(), lr=args.lr, weight_decay=1e-4)
    losses = []
    for step in range(1, args.steps + 1):
        critic.train()
        # Balanced sampling prevents collapse; held-out Platt fitting restores
        # the prevalence of the supplied generated validation-route population.
        batch = np.concatenate([rng.choice(group, args.batch_size // 2, replace=True) for group in classes])
        rng.shuffle(batch)
        x = torch.as_tensor(encoded_train[batch, None], device=args.device)
        condition = torch.as_tensor(data['condition'][train_indices[batch]], device=args.device)
        labels = torch.as_tensor(train_labels[batch, None], device=args.device)
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(critic(x, condition), labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 200 == 0 or step == args.steps:
            print(json.dumps({'confidence_step': step, 'train_bce': float(np.mean(losses[-200:]))}), flush=True)

    calibration = examples_for_sources(sources, calibration_ids)
    calibration_source = 'generated validation predictions'
    if calibration is None:
        calibration = synthetic_examples(data, calibration_ids, np.random.default_rng(args.seed + 1), args.synthetic_per_scene)
        calibration_source = 'synthetic validation corruption population; generated-route calibration unverified'
    calibration_paths, calibration_indices, calibration_labels = calibration
    calibration_logits = predict_logits(critic, residuals(calibration_paths, calibration_indices, data), calibration_indices, data, args.device)
    slope, bias = fit_platt(calibration_logits, calibration_labels)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config.update(cond_dim=int(data['condition'].shape[-1]), horizon=int(data['paths'].shape[-2]),
                  dataset_sha256=sha256(args.data), features_sha256=sha256(args.features) if args.features else None,
                  geometry_checker_sha256=sha256(inspect.getfile(path_validity)), confidence_scope=SCOPE,
                  train_examples=len(train_labels), train_valid_fraction=float(train_labels.mean()),
                  calibration_scene_indices=calibration_ids.tolist(), holdout_val_scene_indices=holdout_val_ids.tolist(),
                  calibration_source=calibration_source)
    checkpoint = {'model': critic.cpu().state_dict(), 'config': config,
                  'platt_slope': slope, 'platt_bias': bias, 'step': args.steps}
    torch.save(checkpoint, out / 'calibrated_critic.pt')
    critic.to(args.device).eval()
    summary = {'confidence_scope': SCOPE, 'calibration_source': calibration_source,
               'platt': {'slope': slope, 'bias': bias}, 'training': {'n': len(train_labels),
               'valid_fraction': float(train_labels.mean()), 'final_bce': float(np.mean(losses[-200:]))},
               'calibration_fit': reliability(probabilities(calibration_logits, slope, bias), calibration_labels),
               'methods': {}}
    for source in sources:
        # TRAIN predictions were optional critic training inputs, so reporting
        # held-out confidence quality on those would be misleading.
        if source['split'] == 'train':
            continue
        logits = predict_logits(critic, residuals(source['paths'], source['indices'], data), source['indices'], data, args.device)
        confidence = probabilities(logits, slope, bias)
        report_indices = np.isin(source['indices'], holdout_val_ids) if source['split'] == 'val' else np.ones(len(logits), dtype=bool)
        metrics = {'calibrated': reliability(confidence[report_indices], source['labels'][report_indices]),
                   'uncalibrated': reliability(probabilities(logits)[report_indices], source['labels'][report_indices])}
        candidate_confidence=confidence.reshape(source['original_shape'])
        candidate_labels=source['labels'].reshape(source['original_shape'])
        selected=np.argmax(candidate_confidence,axis=-1)
        selected_labels=np.take_along_axis(candidate_labels,selected[...,None],axis=-1)[...,0]
        selected_scenes=np.isin(source['scene_indices'],holdout_val_ids) if source['split']=='val' else np.ones(len(source['scene_indices']),dtype=bool)
        metrics['top1_checker_valid_rate']=float(selected_labels[selected_scenes].mean())
        key = 'val_calibration_holdout' if source['split'] == 'val' else source['split']
        summary['methods'].setdefault(source['method'], {})[key] = metrics
        method_out = out / source['method']
        method_out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(method_out / (source['split'] + '_confidence.npz'),
                            confidence=confidence.reshape(source['original_shape']),
                            raw_logits=logits.reshape(source['original_shape']),
                            checker_valid=source['labels'].reshape(source['original_shape']).astype(bool),
                            indices=source['scene_indices'], scene_ids=data['scene_ids'][source['scene_indices']],
                            confidence_scope=np.asarray(SCOPE), source_prediction_file=np.asarray(source['source']),
                            calibration_fit_scene=np.isin(source['scene_indices'], calibration_ids))
        write_json(method_out / 'metrics.json', summary['methods'][source['method']])
        print(json.dumps({'method': source['method'], 'split': key, 'calibrated': metrics['calibrated']}), flush=True)
    if not any(source['split'] == 'val' for source in sources):
        holdout = synthetic_examples(data, holdout_val_ids, np.random.default_rng(args.seed + 2), args.synthetic_per_scene)
        logits = predict_logits(critic, residuals(holdout[0], holdout[1], data), holdout[1], data, args.device)
        summary['synthetic_val_calibration_holdout'] = reliability(probabilities(logits, slope, bias), holdout[2])
    write_json(out / 'config.json', config)
    write_json(out / 'metrics.json', summary)
    print(json.dumps({'confidence_checkpoint': str((out / 'calibrated_critic.pt').resolve()),
                      'confidence_scope': SCOPE, 'calibration_source': calibration_source}), flush=True)


if __name__ == '__main__':
    main()
