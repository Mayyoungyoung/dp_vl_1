"""Frozen-Qwen + observed RGB-D spatial grounding baseline; no privileged input."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from routeset.common import seed_all, sha256, write_json
from routeset.observed_geometry import (backproject_rgbd, ObservedGeometryRouteHead,
                                       positive_endpoint_attention_loss)
from routeset.observed_route_head import load_observed_dataset
from routeset.train_v2 import atomic_checkpoint, positive_assignment_loss, restore_rng, rng_state, synchronized_time
from scripts.train_observed_routes import observation_metrics, paired_language_indices


POINT_FIELDS = ('world_xyz', 'rgb', 'uv', 'depth', 'valid_mask')


def validate_anchor_resume(current_config, saved_config):
    if current_config.get('anchor_mode', 'soft') != saved_config.get('anchor_mode', 'soft'):
        raise ValueError('resume config mismatch: anchor_mode')


def read_geometry(image, observation, pixel_stride):
    """Whitelist current RGB/depth/camera; never flatten arbitrary NPZ fields."""
    with Image.open(image) as source:
        rgb = np.asarray(source.convert('RGB'))
    with np.load(observation, allow_pickle=False) as archive:
        points = backproject_rgbd(rgb, archive['depth'], archive['camera_intrinsics'],
                                  archive['camera_extrinsics'], pixel_stride=pixel_stride)
    return {name: points[name] for name in POINT_FIELDS}


def load_geometry(data, observations, supervision, pixel_stride=2):
    started = time.perf_counter()
    observations, supervision = Path(observations), Path(supervision)
    rows = {r['id']: r for r in [json.loads(x) for x in observations.read_text(encoding='utf-8-sig').splitlines() if x.strip()]}
    labels = {r['id']: r for r in [json.loads(x) for x in supervision.read_text(encoding='utf-8-sig').splitlines() if x.strip()]}
    items, parent_index, indices, sources, timings = [], {}, [], [], []
    hashes = {}
    for sample_id, parent_id in zip(data['scene_ids'], data['parent_ids']):
        row, label = rows[str(sample_id)], labels[str(sample_id)]
        image, observation = observations.parent / row['image'], supervision.parent / label['observation']
        key = (str(image.resolve()), str(observation.resolve()))
        if str(parent_id) in parent_index:
            index = parent_index[str(parent_id)]
            if sources[index] != key:
                raise ValueError('parent current geometry must be identical across target instructions')
        else:
            tic = time.perf_counter()
            items.append(read_geometry(*key, pixel_stride))
            timings.append(time.perf_counter()-tic)
            index = len(items)-1
            sources.append(key)
            parent_index[str(parent_id)] = index
            hashes[str(image)], hashes[str(observation)] = sha256(image), sha256(observation)
        indices.append(index)
    packed = {name: np.stack([item[name] for item in items]) for name in POINT_FIELDS}
    metadata = dict(unique_current_observations=len(items), sampled_points=int(packed['depth'].shape[1]),
                    pixel_stride=pixel_stride, load_preprocess_total_s=time.perf_counter()-started,
                    image_npz_load_backproject_ms_median=float(np.median(timings)*1000),
                    image_npz_load_backproject_ms_p95=float(np.percentile(timings, 95)*1000),
                    cached_content='raw observed RGB/XYZ/UV/depth/mask only; trainable point features recomputed every forward',
                    source_hashes=hashes)
    fingerprint = hashlib.sha256(json.dumps(dict(dataset=data['fingerprint'], sources=hashes,
                                               stride=pixel_stride), sort_keys=True).encode()).hexdigest()
    return dict(points=packed, index=np.asarray(indices), sources=sources, metadata=metadata, fingerprint=fingerprint)


def batch_inputs(data, geometry, ids, device, language_ids=None):
    language_ids = ids if language_ids is None else language_ids
    batch = dict(features=torch.as_tensor(data['features'][language_ids], device=device),
                 current=torch.as_tensor(data['current'][ids], device=device))
    for name in POINT_FIELDS:
        batch[name] = torch.as_tensor(geometry['points'][name][geometry['index'][ids]], device=device)
    return batch


@torch.no_grad()
def evaluate(model, data, geometry, ids, device, output=None, batch_size=8):
    model.eval()
    predictions, events, anchors = [], [], []
    for start in range(0, len(ids), batch_size):
        xyz, opened, details = model(**batch_inputs(data, geometry, ids[start:start+batch_size], device))
        predictions.append(xyz.cpu().numpy())
        events.append(opened.cpu().numpy())
        anchors.append(details['anchor_xyz'].cpu().numpy())
    predictions, events, anchors = map(np.concatenate, (predictions, events, anchors))
    result, rows = observation_metrics(predictions, events, data, ids)
    # Strict semantic metrics retain the original identity and 3 cm criteria.
    # Anchor error uses recorded route endpoints only, never target coordinates.
    anchor_error = [float(np.linalg.norm(anchors[row]-data['paths'][idx, data['path_mask'][idx], -1], axis=-1).min())
                    if data['path_mask'][idx].any() else None for row, idx in enumerate(ids)]
    finite_anchor_error = [error for error in anchor_error if error is not None]
    result['learned_surface_anchor_reference_endpoint_error_m'] = float(np.mean(finite_anchor_error)) if finite_anchor_error else None
    for row, error in zip(rows, anchor_error):
        row['learned_surface_anchor_reference_endpoint_error_m'] = error
    switched = paired_language_indices(data, ids)
    eligible = np.flatnonzero(switched >= 0)
    swapped_xyz, swapped_events = [], []
    if len(eligible):
        for start in range(0, len(eligible), batch_size):
            subset = eligible[start:start+batch_size]
            xyz, opened, _ = model(**batch_inputs(data, geometry, ids[subset], device, switched[subset]))
            swapped_xyz.append(xyz.cpu().numpy())
            swapped_events.append(opened.cpu().numpy())
        swapped_xyz, swapped_events = np.concatenate(swapped_xyz), np.concatenate(swapped_events)
        wrong, _ = observation_metrics(swapped_xyz, swapped_events, data, ids[eligible])
        own, _ = observation_metrics(swapped_xyz, swapped_events, data, switched[eligible])
        result['paired_language_control'] = dict(examples=len(eligible),
            same_image_other_target_language_original_goal_accuracy=wrong['semantic_goal_accuracy'],
            same_image_other_target_language_new_goal_accuracy=own['semantic_goal_accuracy'],
            mean_endpoint_response_m=float(np.linalg.norm(swapped_xyz[:, :, -1]-predictions[eligible, :, -1], axis=-1).mean()),
            explanation='Only Qwen image-language feature changes; RGB-D/current remain identical')
    else:
        result['paired_language_control'] = None
    result['selection_score'] = -result['candidate_matched_ADE_m']
    if output is not None:
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/'predictions.npz', paths=predictions, gripper_open=events,
                            learned_surface_anchor=anchors, scene_ids=data['scene_ids'][ids], parent_ids=data['parent_ids'][ids])
        if len(eligible):
            np.savez_compressed(output/'paired_language_predictions.npz', paths=swapped_xyz, gripper_open=swapped_events,
                                original_scene_ids=data['scene_ids'][ids[eligible]], language_source_ids=data['scene_ids'][switched[eligible]])
        write_json(output/'metrics.json', result)
        write_json(output/'per_scene.json', rows)
    return result


@torch.no_grad()
def measure_latency(model, data, geometry, idx, device, pixel_stride):
    """Re-read a genuine single RGB-D request; Qwen time remains separate."""
    model.eval()
    cached_times, request_times, preprocess_times = [], [], []
    source = geometry['sources'][geometry['index'][idx]]
    for repeat in range(12):
        tic = synchronized_time(device)
        xyz, opened, _ = model(**batch_inputs(data, geometry, np.array([idx]), device))
        xyz.cpu().numpy(); opened.cpu().numpy()
        elapsed = synchronized_time(device)-tic
        request_start = synchronized_time(device)
        raw = read_geometry(*source, pixel_stride)
        preprocessing = synchronized_time(device)-request_start
        inputs = {name: torch.as_tensor(value[None], device=device) for name, value in raw.items()}
        inputs.update(features=torch.as_tensor(data['features'][idx:idx+1], device=device),
                      current=torch.as_tensor(data['current'][idx:idx+1], device=device))
        xyz, opened, _ = model(**inputs)
        xyz.cpu().numpy(); opened.cpu().numpy()
        request = synchronized_time(device)-request_start
        if repeat >= 2:
            cached_times.append(elapsed*1000)
            request_times.append(request*1000)
            preprocess_times.append(preprocessing*1000)
    return dict(raw_geometry_cached_head_batch1_ms_median=float(np.median(cached_times)),
                raw_geometry_cached_head_batch1_ms_p95=float(np.percentile(cached_times, 95)),
                rgbd_read_preprocess_head_batch1_ms_median=float(np.median(request_times)),
                rgbd_read_preprocess_head_batch1_ms_p95=float(np.percentile(request_times, 95)),
                rgbd_read_preprocess_batch1_ms_median=float(np.median(preprocess_times)),
                latency_scope='single-request RGB-D disk read/backprojection/transfers/trainable point encoder/set head; excludes Qwen, checking and scoring; OS file pages may be warm')


def train(args):
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda'):
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
            raise RuntimeError('physical GPU1 must be explicitly selected via CUDA_VISIBLE_DEVICES=1')
        torch.cuda.set_per_process_memory_fraction(.35)
        torch.cuda.reset_peak_memory_stats()
    seed_all(args.seed)
    load_start = time.perf_counter()
    data = load_observed_dataset(args.observations, args.supervision, args.cache_dir, args.horizon, args.pooling)
    geometry = load_geometry(data, args.observations, args.supervision, args.pixel_stride)
    data_load_seconds = time.perf_counter()-load_start
    train_ids = np.flatnonzero((data['splits'] == 'TRAIN') & data['path_mask'].any(axis=1))
    dev_ids = np.flatnonzero(data['splits'] == 'DEV_MODEL')
    if not len(train_ids) or not len(dev_ids) or set(data['parent_ids'][train_ids]) & set(data['parent_ids'][dev_ids]):
        raise ValueError('nonempty parent-disjoint TRAIN and DEV_MODEL required')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'last.pt').exists() and not args.resume:
        raise RuntimeError('existing checkpoint; use --resume or new output')
    lock = out/'active.lock'
    fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    try:
        config = vars(args).copy()
        config.update(feature_dim=int(data['features'].shape[1]), dataset_fingerprint=geometry['fingerprint'],
            code_commit=os.environ.get('CODE_COMMIT', 'unrecorded'), source_script_sha256=sha256(Path(__file__)),
            objective='saturation', selection_split='DEV_MODEL', selection_metric='negative candidate_matched_ADE_m',
            condition_fields=['frozen real-Qwen RGB+instruction hidden states', 'current gripper pose7/open1',
                              'current RGB pixels + metric depth + camera calibration'],
            endpoints='learned observed surface anchor + coordinatewise bounded residual; see endpoint_residual_bound; no given target',
            target_coordinate_policy='semantic coordinates used only in evaluation; training supervision is recorded route and event',
            geometry_role='conventional spatial grounding baseline, not claimed core mechanism',
            train_examples=len(train_ids), dev_examples=len(dev_ids),
            train_parents=len(set(data['parent_ids'][train_ids])), dev_parents=len(set(data['parent_ids'][dev_ids])),
            skipped_supervision=data['skipped'], unreferenced_observations=data['unreferenced'],
            evaluation_protocol=data['evaluation_protocol'], cache_config=data['cache_config'], geometry_preprocessing=geometry['metadata'])
        model = ObservedGeometryRouteHead(config['feature_dim'], args.horizon, args.candidates, args.width,
                                          args.depth, args.point_width, args.endpoint_residual_bound,
                                          anchor_mode=args.anchor_mode).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        rng, sampler = np.random.default_rng(args.seed), np.random.default_rng(args.seed+100000)
        first_step, best, elapsed_before, exposures, history = 0, -float('inf'), 0., 0, []
        if args.resume:
            checkpoint = torch.load(out/'last.pt', map_location=args.device, weights_only=False)
            validate_anchor_resume(config, checkpoint['config'])
            for key in ('dataset_fingerprint', 'feature_dim', 'horizon', 'candidates', 'width', 'depth', 'steps', 'seed',
                        'lr', 'batch_size', 'event_scale', 'pooling', 'pixel_stride', 'point_width', 'endpoint_residual_bound'):
                if config[key] != checkpoint['config'][key]:
                    raise ValueError('resume config mismatch: '+key)
            for key, default in [('grounding_weight',0.),('grounding_sigma',.025)]:
                if config[key] != checkpoint['config'].get(key,default):
                    raise ValueError('resume config mismatch: '+key)
            model.load_state_dict(checkpoint['model']); optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler']); restore_rng(checkpoint['rng'], rng)
            sampler.bit_generator.state = checkpoint['sampler_state']
            first_step, best, elapsed_before = checkpoint['step'], checkpoint['best'], checkpoint['elapsed_s']
            exposures, history = checkpoint['trajectory_exposures'], checkpoint['history']
        write_json(out/'config.json', config)
        write_json(out/'source_hashes.json', dict(data['source_hashes'], **geometry['metadata']['source_hashes']))
        write_json(out/'status.json', dict(status='running', pid=os.getpid(), step=first_step))
        started, losses, path_losses, grounding_losses = synchronized_time(args.device), [], [], []
        for step in range(first_step+1, args.steps+1):
            model.train()
            ids = sampler.choice(train_ids, args.batch_size, replace=True)
            inputs = batch_inputs(data, geometry, ids, args.device)
            xyz, opened, details = model(**inputs)
            target_xyz = torch.as_tensor(data['paths'][ids], device=args.device)
            target_events = torch.as_tensor(data['events'][ids], device=args.device)
            prediction = torch.cat([xyz[:, :, 1:], opened[:, :, 1:, None]*args.event_scale], dim=-1)
            target = torch.cat([target_xyz[:, :, 1:], target_events[:, :, 1:, None]*args.event_scale], dim=-1)
            path_loss = positive_assignment_loss(prediction, target, data['path_mask'][ids], 'saturation', rng)
            grounding_loss = xyz.new_zeros(())
            if args.grounding_weight:
                grounding_loss = positive_endpoint_attention_loss(details['attention'], inputs['world_xyz'],
                    inputs['valid_mask'], target_xyz[:, :, -1],
                    torch.as_tensor(data['path_mask'][ids], device=args.device), args.grounding_sigma)
            loss = path_loss + args.grounding_weight*grounding_loss
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step()
            exposures += args.batch_size*args.candidates
            losses.append(float(loss.detach()))
            path_losses.append(float(path_loss.detach()))
            grounding_losses.append(float(grounding_loss.detach()))
            if step % 100 == 0:
                print(json.dumps(dict(step=step, loss=float(np.mean(losses[-100:])),
                                     path_loss=float(np.mean(path_losses[-100:])),
                                     grounding_loss=float(np.mean(grounding_losses[-100:])),
                                     elapsed_s=elapsed_before+synchronized_time(args.device)-started)), flush=True)
            if step % args.eval_every == 0 or step == args.steps or step == args.stop_after:
                metrics = evaluate(model, data, geometry, dev_ids, args.device)
                score = metrics['selection_score']; improved = score > best; best = max(best, score)
                elapsed = elapsed_before+synchronized_time(args.device)-started
                history.append(dict(step=step, loss=float(np.mean(losses[-100:])), dev_model=metrics))
                checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    scaler=None, step=step, config=config, rng=rng_state(rng), sampler_state=sampler.bit_generator.state,
                    best=best, history=history, elapsed_s=elapsed, trajectory_exposures=exposures)
                atomic_checkpoint(out/'last.pt', checkpoint)
                if improved:
                    atomic_checkpoint(out/'best.pt', checkpoint)
                write_json(out/'history.json', history)
                print(json.dumps(history[-1]), flush=True)
                if step == args.stop_after and step < args.steps:
                    write_json(out/'status.json', dict(status='interrupted_for_resume_check', step=step, exit_code=0))
                    return
        checkpoint = torch.load(out/'best.pt', map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint['model'])
        metrics = evaluate(model, data, geometry, dev_ids, args.device, out/'dev_model')
        train_metrics = evaluate(model, data, geometry, np.flatnonzero(data['splits'] == 'TRAIN'), args.device, out/'train')
        latency = measure_latency(model, data, geometry, int(dev_ids[0]), args.device, args.pixel_stride)
        elapsed = elapsed_before+synchronized_time(args.device)-started
        summary = dict(metrics=metrics, train_metrics=train_metrics, latency=latency, elapsed_s=elapsed,
            data_load_preprocess_s=data_load_seconds, geometry_preprocessing=geometry['metadata'],
            gpu_hours_reserved=elapsed/3600 if args.device.startswith('cuda') else 0.,
            parameters=model.active_parameter_count(), geometry_parameters=model.geometry.active_parameter_count(),
            trajectory_exposures=exposures,
            peak_cuda_memory_mb=torch.cuda.max_memory_allocated()/2**20 if args.device.startswith('cuda') else 0.,
            best_step=checkpoint['step'], best_checkpoint_sha256=sha256(out/'best.pt'),
            prediction_sha256=sha256(out/'dev_model'/'predictions.npz'),
            evidence_scope='frozen real-Qwen plus RGB-D ordinary set regression pilot; not mechanism or robot execution evidence')
        write_json(out/'summary.json', summary)
        write_json(out/'status.json', dict(status='completed', step=args.steps, exit_code=0))
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        write_json(out/'status.json', dict(status='failed', exit_code=1, exception=repr(exc)))
        raise
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('observations', 'supervision', 'cache-dir', 'output'):
        parser.add_argument('--'+key, required=True)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--candidates', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--pooling', choices=('mean', 'last', 'both'), default='both')
    parser.add_argument('--geometry-pooling', choices=('spatial',), default='spatial')
    parser.add_argument('--anchor-mode', choices=('soft', 'straight_through_peak'), default='soft')
    parser.add_argument('--point-width', type=int, default=64)
    parser.add_argument('--pixel-stride', type=int, default=2)
    parser.add_argument('--endpoint-residual-bound', type=float, default=.05)
    parser.add_argument('--grounding-weight', type=float, default=0.)
    parser.add_argument('--grounding-sigma', type=float, default=.025)
    parser.add_argument('--event-scale', type=float, default=.2)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if min(args.steps, args.eval_every, args.batch_size, args.pixel_stride, args.point_width) < 1 or not 1 <= args.threads <= 4:
        parser.error('positive steps/batch/eval/point sizes and 1--4 threads required')
    if args.grounding_weight < 0 or args.grounding_sigma <= 0:
        parser.error('nonnegative grounding weight and positive sigma required')
    train(args)


if __name__ == '__main__':
    main()
