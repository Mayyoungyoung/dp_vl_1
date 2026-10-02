"""True Qwen + current RGB-D + learned route head, timed one request at a time.

Measures observation-to-path inference, including file reads and preprocessing.
Model loading is separate. Collision checking, scoring and robot execution are
not implemented here and must not be described as included in this latency.
"""
import argparse
import json
import os
from pathlib import Path
import time
import hashlib

REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
INPUT_KEYS = {'id','parent_id','split','image','instruction'}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def resolve(base, value):
    path = Path(value)
    return path if path.is_absolute() else base/path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--requests', type=int, default=24)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    if args.requests < 3 or not 1 <= args.threads <= 4:
        parser.error('at least three requests and 1--4 CPU threads required')
    config = json.loads((args.run/'config.json').read_text())
    if config.get('evaluation_protocol') != 'observation_eval_v2':
        raise ValueError('v2 evaluation protocol required')
    provenance = json.loads((args.model/'provenance.json').read_text())
    if provenance['revision'] != REVISION or not provenance['all_hashes_verified']:
        raise ValueError('pinned officially verified real Qwen weights required')
    if config['cache_config']['revision'] != REVISION or config['cache_config']['processor'] != REVISION:
        raise ValueError('training model/processor revisions must match')
    observations = Path(config['observations']); supervision = Path(config['supervision'])
    rows = [json.loads(line) for line in observations.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if any(set(row) != INPUT_KEYS for row in rows):
        raise ValueError('only RGB+instruction manifest fields accepted')
    rows = sorted([row for row in rows if row['split'] == 'DEV_MODEL'], key=lambda row:row['id'])
    labels = {row['id']:row for row in [json.loads(line) for line in supervision.read_text().splitlines() if line.strip()]}
    selected = rows[:args.requests]
    if len(selected) != args.requests:
        raise ValueError('requested more distinct DEV observations than available')
    # Only the current-observation file pointer is read from supervision;
    # semantic target centers, future routes and acceptance never enter inference.
    current_files = {row['id']:resolve(supervision.parent, labels[row['id']]['observation']) for row in selected}
    image_files = {row['id']:resolve(observations.parent, row['image']) for row in selected}
    required = [args.run/'best.pt']+list(current_files.values())+list(image_files.values())
    if any(not path.is_file() for path in required):
        raise ValueError('missing actual checkpoint or current observation files')
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = dict(run=str(args.run.resolve()), checkpoint_sha256=file_hash(args.run/'best.pt'),
        model_revision=REVISION, processor_revision=REVISION, requests=args.requests,
        input_manifest_sha256=file_hash(observations), source_script_sha256=file_hash(__file__),
        input_fields=['RGB','instruction','current depth','camera calibration','current gripper pose/open'],
        excluded=['future path tokens','true target coordinates','complete obstacle geometry','verification labels'],
        scope='single-request observation-to-path inference; excludes collision validation, scoring, robot execution',
        cache_use='no cached Qwen or learned point features; actual processor/model/geometry/head every request')
    if args.validate_only:
        metadata.update(status='input_and_source_preflight_only', gpu_executed=False)
        (args.output/'preflight.json').write_text(json.dumps(metadata, indent=2))
        print(json.dumps(metadata), flush=True)
        return
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise RuntimeError('use only authorized physical GPU1 via CUDA_VISIBLE_DEVICES=1')
    import numpy as np
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from routeset.observed_geometry import backproject_rgbd, ObservedGeometryRouteHead
    torch.set_num_threads(args.threads)
    torch.cuda.set_per_process_memory_fraction(.35)
    torch.cuda.reset_peak_memory_stats()
    device = 'cuda'
    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()
    load_started = clock()
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.bfloat16,
            local_files_only=True, attn_implementation='sdpa', device_map={'':device})
    model.requires_grad_(False).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
        min_pixels=64*32*32, max_pixels=config['cache_config']['max_pixels'])
    head = ObservedGeometryRouteHead(config['feature_dim'], config['horizon'], config['candidates'], config['width'],
        config['depth'], config['point_width'], config['endpoint_residual_bound']).to(device)
    checkpoint = torch.load(args.run/'best.pt', map_location=device, weights_only=False)
    head.load_state_dict(checkpoint['model']); head.requires_grad_(False).eval()
    loading_seconds = clock()-load_started
    predictions, opened_predictions, records = [], [], []
    with torch.inference_mode():
        for row in selected:
            start = clock()
            with Image.open(image_files[row['id']]) as source:
                image = source.convert('RGB')
            with np.load(current_files[row['id']], allow_pickle=False) as archive:
                observed = {key:archive[key] for key in ('depth','camera_intrinsics','camera_extrinsics','gripper_pose','gripper_open')}
            current = np.r_[observed['gripper_pose'].reshape(7), observed['gripper_open'].reshape(1)].astype(np.float32)
            read_end = clock()
            points = backproject_rgbd(np.asarray(image), observed['depth'], observed['camera_intrinsics'],
                observed['camera_extrinsics'], pixel_stride=config['pixel_stride'])
            point_inputs = {key:torch.as_tensor(points[key][None], device=device) for key in ('world_xyz','rgb','uv','depth','valid_mask')}
            point_inputs['current'] = torch.as_tensor(current[None], device=device)
            geometry_end = clock()
            messages = [{'role':'user','content':[{'type':'image'}, {'type':'text','text':row['instruction']}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[prompt], images=[image], return_tensors='pt').to(device)
            preprocessing_end = clock()
            output = model.model(**inputs, use_cache=False, return_dict=True)
            hidden = output.last_hidden_state[0][inputs.attention_mask[0].bool()].float()
            mean, last = hidden.mean(0), hidden[-1]
            pooled = torch.cat([mean,last]) if config['pooling'] == 'both' else (mean if config['pooling'] == 'mean' else last)
            point_inputs['features'] = pooled[None]
            qwen_end = clock()
            xyz, opened, _ = head(**point_inputs)
            predictions.append(xyz.cpu().numpy()[0]); opened_predictions.append(opened.cpu().numpy()[0])
            finish = clock()
            record = dict(id=row['id'], parent_id=row['parent_id'], candidates=config['candidates'],
                read_current_inputs_ms=(read_end-start)*1000,
                rgbd_backprojection_transfer_ms=(geometry_end-read_end)*1000,
                qwen_processor_transfer_ms=(preprocessing_end-geometry_end)*1000,
                qwen_real_forward_pool_ms=(qwen_end-preprocessing_end)*1000,
                learned_geometry_route_head_output_transfer_ms=(finish-qwen_end)*1000,
                observation_to_paths_ms=(finish-start)*1000,
                input_tokens=int(inputs.attention_mask.sum()), sampled_points=len(points['depth']))
            records.append(record); print(json.dumps(record), flush=True)
    np.savez_compressed(args.output/'predictions.npz', paths=np.asarray(predictions), gripper_open=np.asarray(opened_predictions),
                        scene_ids=np.asarray([row['id'] for row in selected]))
    elapsed = np.asarray([row['observation_to_paths_ms'] for row in records])
    summary = dict(metadata, status='measured_true_qwen_rgbd_route_inference', gpu_executed=True,
        model_loading_seconds=loading_seconds, first_request_ms=float(elapsed[0]),
        all_requests_ms_median=float(np.median(elapsed)), all_requests_ms_p95=float(np.percentile(elapsed,95)),
        after_first_two_requests_ms_median=float(np.median(elapsed[2:])), after_first_two_requests_ms_p95=float(np.percentile(elapsed[2:],95)),
        loaded_checkpoint_step=checkpoint['step'], cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        torch=torch.__version__, transformers=transformers.__version__,
        per_request=records, prediction_sha256=file_hash(args.output/'predictions.npz'),
        timing_notes='Actual distinct DEV inputs evaluated serially with synchronized stages. File-system pages may already be warm. Model loading is separately reported; this is not cached-feature throughput.')
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key:value for key,value in summary.items() if key != 'per_request'}), flush=True)


if __name__ == '__main__':
    main()
