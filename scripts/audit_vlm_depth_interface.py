"""Eight predeclared TRAIN observations; actual CPU processor, no model/answers.

Invert patch permutation and normalization, not interpolation or a ViT embedding.
The audit measures representation distortion; it cannot attribute model failure.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import time

import numpy as np

from routeset.common import sha256, write_json
from routeset.vlm_route_serialization import depth_image
from routeset.vlm_sft_data import prepare_prefix, read_observation, resolve

PROTOCOL = 'vlm_depth_interface_train8_v1'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
PARENTS = ['obstacle_reach_%d' % n for n in range(272000, 272008)]


def select_samples(rows, plan):
    """Metadata-only exact selection. Nonselected raw records are never opened."""
    if plan['protocol'] != PROTOCOL or [r['parent_id'] for r in plan['samples']] != PARENTS:
        raise ValueError('Exactly the predeclared eight TRAIN parents required')
    result = []
    for expected in plan['samples']:
        matching = sorted([r for r in rows if r['parent_id'] == expected['parent_id']], key=lambda r: r['id'])
        if not matching or matching[0]['id'] != expected['id']:
            raise ValueError('Predeclared first observation is missing')
        if len({r['id'] for r in matching}) != len(matching) or any(r['split'] != 'TRAIN' for r in matching):
            raise ValueError('Selected parent identity/role mismatch')
        if set(matching[0]) != {'id', 'parent_id', 'split', 'instruction', 'image'}:
            raise ValueError('Observation-only manifest required')
        result.append(matching[0])
    return result


def unpack_pixels(flat, grid, patch_size, merge_size, temporal_patch_size):
    """Inverse of pinned Qwen2VL image processor patch permutation, NumPy only."""
    gt, gh, gw = map(int, grid)
    p, m, tp = patch_size, merge_size, temporal_patch_size
    flat = np.asarray(flat)
    if gt != 1 or gh % m or gw % m or flat.shape != (gt*gh*gw, 3*tp*p*p):
        raise ValueError('Unexpected still-image patch shape/grid')
    grouped = flat.reshape(gt, gh//m, gw//m, m, m, 3, tp, p, p)
    frames = grouped.transpose(0, 6, 5, 1, 3, 7, 2, 4, 8).reshape(gt*tp, 3, gh*p, gw*p)
    if not np.array_equal(frames, np.repeat(frames[:1], gt*tp, axis=0)):
        raise ValueError('Repeated still-image temporal patches differ')
    return frames[0].transpose(1, 2, 0)


def unnormalize_pixels(normalized, mean, std, rescale_factor):
    if not np.isfinite(rescale_factor) or rescale_factor <= 0:
        raise ValueError('Positive rescale factor required')
    return (np.asarray(normalized, dtype=np.float64)*np.asarray(std)+np.asarray(mean))/rescale_factor


def stats(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    values = values[np.isfinite(values)]
    if not len(values):
        return dict(count=0, mean=None, median=None, p95=None, max=None)
    return dict(count=len(values), mean=float(values.mean()), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), max=float(values.max()))


def compare_arrays(depth, raw_bytes, after_float, metric_resized, nearest_depth, nearest_valid, support):
    """Neither reference resize is truth: separate coding distortion from resampling."""
    valid = np.isfinite(depth) & (depth > 0)
    after_bytes = np.clip(np.rint(after_float), 0, 255).astype(np.uint8)
    raw_decoded = (raw_bytes[..., 0].astype(float)*256+raw_bytes[..., 1])/1000
    decoded = (after_bytes[..., 0].astype(float)*256+after_bytes[..., 1])/1000
    after_valid = after_bytes[..., 2] == 255
    after_unknown = after_bytes[..., 2] == 0
    # Also retain the algebraic decoding before rounding recovered channels.
    decoded_float = (after_float[..., 0]*256+after_float[..., 1])/1000
    supported = after_valid & (support >= 1-1e-6) & nearest_valid
    result = dict(
        raw_shape=list(depth.shape), resized_shape=list(decoded.shape),
        raw_valid_pixels=int(valid.sum()), raw_invalid_pixels=int((~valid).sum()), all_raw_pixels_valid=bool(valid.all()),
        raw_quantization_abs_m=stats(np.abs(raw_decoded[valid]-depth[valid])),
        recovered_channel_distance_to_integer=stats(np.abs(after_float-np.rint(after_float))),
        resized_pixels=decoded.size, resized_B255_pixels=int(after_valid.sum()),
        resized_B0_pixels=int(after_unknown.sum()),
        resized_B_intermediate_pixels=int((~after_valid & ~after_unknown).sum()),
        validity_disagreement_with_nearest=int((after_valid != nearest_valid).sum()),
        B255_with_partial_bilinear_valid_support=int((after_valid & (support < 1-1e-6)).sum()),
        common_bilinear_supported_pixels=int(supported.sum()),
        byte_decode_vs_float_metric_bicubic_abs_m=stats(np.abs(decoded[supported]-metric_resized[supported])),
        byte_decode_vs_raw_nearest_abs_m=stats(np.abs(decoded[supported]-nearest_depth[supported])),
        float_metric_bicubic_vs_raw_nearest_abs_m=stats(np.abs(metric_resized[supported]-nearest_depth[supported])),
        float_channel_decode_vs_float_metric_bicubic_abs_m=stats(np.abs(decoded_float[supported]-metric_resized[supported])),
        byte_decode_error_gt_1mm=int((np.abs(decoded[supported]-metric_resized[supported]) > .001).sum()),
        byte_decode_error_gt_1cm=int((np.abs(decoded[supported]-metric_resized[supported]) > .01).sum()),
        byte_decode_error_gt_10cm=int((np.abs(decoded[supported]-metric_resized[supported]) > .1).sum()))
    return result, decoded, after_bytes, supported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('configs/vlm_depth_interface_train8_v1.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh diagnostic output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'):
        raise ValueError('Hide all GPUs for this CPU-only diagnostic')
    started = time.perf_counter()
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    config_path = args.training/'config.json'
    config = json.loads(config_path.read_text())
    index_path = args.training/'data_source_hashes.json'
    index = json.loads(index_path.read_text())
    fingerprint = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
    if fingerprint != plan['training_data_fingerprint'] or fingerprint != config['data_fingerprint']:
        raise ValueError('Original SFT source index changed')
    if config['model_revision'] != REVISION or config['processor_revision'] != REVISION:
        raise ValueError('Pinned actual model/processor revision required')
    if config['min_pixels'] != 65536 or config['max_pixels'] != 65536:
        raise ValueError('Actual SFT pixel settings required')
    root = Path(__file__).resolve().parents[1]
    reservation = root/'configs/observation_partition_reservation_v1.json'
    if sha256(reservation) != plan['reservation_sha256']:
        raise ValueError('Registered parent reservation changed')
    manifest = Path(config['observations'])
    if sha256(manifest) != plan['observations_sha256'] or index.get(str(manifest)) != sha256(manifest):
        raise ValueError('Original observation manifest changed')
    for name in ('routeset/vlm_route_serialization.py', 'routeset/vlm_sft_data.py'):
        if sha256(root/name) != config['source_sha256'][name]:
            raise ValueError('Actual trained input helper changed: '+name)
    # No supervision/route/verification files are read. This JSONL is metadata.
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    selected = select_samples(rows, plan)
    samples, input_hashes = [], {}
    for row, declared in zip(selected, plan['samples']):
        image_path = resolve(manifest.parent, row['image']).resolve()
        observation_path = image_path.with_name('observation.npz')
        if image_path.parent.name != row['parent_id'] or image_path.name != 'front.png':
            raise ValueError('Selected current observation parent/path changed')
        for key, path in [('image_sha256', image_path), ('observation_sha256', observation_path)]:
            digest = sha256(path)
            if digest != declared[key] or digest != index.get(str(path)):
                raise ValueError('Selected TRAIN current input changed')
            input_hashes[str(path)] = digest
        samples.append(dict(id=row['id'], parent_id=row['parent_id'], split='TRAIN',
                            instruction=row['instruction'], image_path=str(image_path), observation_path=str(observation_path)))
    import torch
    import transformers
    import torchvision
    from transformers import AutoProcessor
    from transformers.image_utils import SizeDict
    from torchvision.transforms import InterpolationMode
    if transformers.__version__ != '4.57.1' or not torch.__version__.startswith('2.4.1'):
        raise ValueError('Pinned actual CPU processor runtime required')
    torch.set_num_threads(1)
    model_path = Path(config['model'])
    if sha256(model_path/'provenance.json') != config['model_provenance_sha256']:
        raise ValueError('Pinned model provenance changed')
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, min_pixels=65536, max_pixels=65536)
    ip = processor.image_processor
    if type(ip).__name__ != 'Qwen2VLImageProcessorFast' or not ip.do_resize or not ip.do_rescale or not ip.do_normalize:
        raise ValueError('Unexpected actual processor implementation; do not substitute a mock')
    if int(ip.resample) != 3:
        raise ValueError('Expected actual bicubic resize')
    args.output.mkdir(parents=True)
    records = []
    for sample in samples:
        observed = read_observation(sample)  # whitelist current RGB/depth/calibration/state only
        batch = prepare_prefix(processor, observed, 1, 24)  # exact SFT image grouping/processing
        grids = batch['image_grid_thw'].cpu().numpy()
        if grids.shape != (2, 3):
            raise ValueError('Exactly current RGB and depth images required')
        rgb_patches = int(np.prod(grids[0])); depth_patches = int(np.prod(grids[1]))
        flat = batch['pixel_values'].cpu().numpy()
        if len(flat) != rgb_patches+depth_patches:
            raise ValueError('Unexpected extra image patches')
        normalized = unpack_pixels(flat[rgb_patches:], grids[1], ip.patch_size, ip.merge_size, ip.temporal_patch_size)
        after_float = unnormalize_pixels(normalized, ip.image_mean, ip.image_std, ip.rescale_factor)
        raw = depth_image(observed['depth'])
        height, width = normalized.shape[:2]
        size = SizeDict(height=height, width=width)
        def resize_float(array, interpolation):
            tensor = torch.from_numpy(np.asarray(array, dtype=np.float32).copy())[None, None]
            return ip.resize(tensor, size, interpolation=interpolation).numpy()[0, 0]
        direct = ip.resize(torch.from_numpy(raw.copy()).permute(2, 0, 1)[None], size,
                           interpolation=InterpolationMode.BICUBIC).numpy()[0].transpose(1, 2, 0)
        direct_max_error = float(np.abs(after_float-direct).max())
        if direct_max_error > .001:
            raise ValueError('Patch/normalization inverse does not match actual resize')
        valid = np.isfinite(observed['depth']) & (observed['depth'] > 0)
        finite_depth = np.where(valid, observed['depth'], 0)
        metric = resize_float(finite_depth, InterpolationMode.BICUBIC)
        nearest = resize_float(finite_depth, InterpolationMode.NEAREST_EXACT)
        nearest_valid = resize_float(valid, InterpolationMode.NEAREST_EXACT) > .5
        support = resize_float(valid, InterpolationMode.BILINEAR)
        record, decoded, after_bytes, supported = compare_arrays(
            observed['depth'], raw, after_float, metric, nearest, nearest_valid, support)
        record.update(id=sample['id'], parent_id=sample['parent_id'], split='TRAIN',
                      image_grid_thw=grids.tolist(), actual_inverse_vs_direct_resize_max_channel_error=direct_max_error)
        artifact = args.output/(sample['parent_id']+'.npz')
        np.savez_compressed(artifact, raw_depth=observed['depth'], raw_bytes=raw, actual_depth_pixel_values=flat[rgb_patches:],
                            recovered_resized_channels=after_float.astype(np.float32), resized_bytes=after_bytes,
                            decoded_m=decoded, float_metric_bicubic_m=metric, raw_nearest_m=nearest,
                            nearest_valid=nearest_valid, bilinear_valid_support=support, common_supported=supported)
        record['artifact_sha256'] = sha256(artifact)
        records.append(record)
        write_json(args.output/'progress.json', dict(completed=len(records), expected=8, latest=record))
        print(json.dumps(record), flush=True)
    source_paths = [Path(__file__), root/'routeset/vlm_route_serialization.py', root/'routeset/vlm_sft_data.py', args.plan,
                    Path(inspect.getfile(type(ip))), Path(inspect.getfile(type(processor))),
                    Path(inspect.getfile(ip.resize.__func__))]
    summary = dict(protocol=PROTOCOL, status='completed', code_commit=os.environ.get('CODE_COMMIT'),
        train_parents=8, raw_observations_opened=8, dev_raw_opened=0, reference_paths_opened=0, model_forwards=0,
        processor_revision=REVISION, processor_class=type(processor).__name__, image_processor_class=type(ip).__name__,
        image_processor_config=ip.to_dict(), training_config_sha256=sha256(config_path), data_fingerprint=fingerprint,
        dependencies=dict(torch=torch.__version__, transformers=transformers.__version__, torchvision=torchvision.__version__, numpy=np.__version__),
        plan_sha256=sha256(args.plan), input_sha256=input_hashes,
        source_sha256={str(path.resolve()): sha256(path) for path in source_paths},
        processor_artifact_sha256={name: sha256(model_path/name) for name in
            ('preprocessor_config.json', 'processor_config.json', 'tokenizer_config.json', 'chat_template.json', 'chat_template.jinja')
            if (model_path/name).exists()},
        elapsed_seconds=time.perf_counter()-started, reserved_gpu_hours=0, cpu_threads=1, records=records,
        interpretation='Inverse recovers processed pixels, not original samples or ViT features. Resizes are diagnostic comparators, not ground truth. Bilinear valid support does not certify the wider bicubic stencil; invalid depth is zero-filled only for comparison. No causal claim about generation.')
    write_json(args.output/'summary.json', summary)
    write_json(args.output/'artifacts_sha256.json', {path.name: sha256(path) for path in sorted(args.output.iterdir()) if path.is_file()})


if __name__ == '__main__':
    main()
