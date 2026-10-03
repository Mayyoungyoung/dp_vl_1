"""Cache frozen, genuinely loaded Qwen3-VL RGB + instruction hidden states.

Manifest is JSONL with exactly id,parent_id,split,image,instruction. Supervision,
true target coordinates, object geometry, and future routes cannot enter this
loader. Camera/depth/current-state inputs belong in a separate observation
encoder. These caches become invalid after ANY backbone/LoRA parameter change.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
ALLOWED = {'id', 'parent_id', 'split', 'image', 'instruction'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path):
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not rows:
        raise ValueError('Empty observation manifest')
    seen, parent_splits = set(), {}
    for row in rows:
        if set(row) != ALLOWED:
            raise ValueError('Observation-only manifest must contain exactly ' + str(sorted(ALLOWED)))
        if row['id'] in seen:
            raise ValueError('Duplicate sample id')
        seen.add(row['id'])
        if row['parent_id'] in parent_splits and parent_splits[row['parent_id']] != row['split']:
            raise ValueError('Parent scene leaks across splits')
        parent_splits[row['parent_id']] = row['split']
        if not isinstance(row['instruction'], str) or not row['instruction'].strip():
            raise ValueError('Missing language instruction')
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--gpu-memory-fraction', type=float, default=0.35)
    parser.add_argument('--max-pixels', type=int, default=256 * 32 * 32)
    args = parser.parse_args()
    rows = read_manifest(args.manifest)
    provenance = json.loads((args.model / 'provenance.json').read_text())
    if provenance['revision'] != REVISION or not provenance['all_hashes_verified']:
        raise ValueError('Model provenance does not match the pinned official snapshot')
    import numpy as np
    from PIL import Image
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    torch.set_num_threads(args.threads)
    if args.device == 'cuda':
        # The runner must expose only authorized physical GPU1.
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
            raise ValueError('This project currently requires CUDA_VISIBLE_DEVICES=1')
        if args.gpu_memory_fraction > 0.35:
            raise ValueError('GPU memory fraction exceeds current authorization')
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
    started = time.perf_counter()
    dtype = torch.bfloat16 if args.device == 'cuda' else torch.float32
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, local_files_only=True,
        attn_implementation='sdpa', device_map={'': args.device})
    model.requires_grad_(False).eval()
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True,
                                              min_pixels=64 * 32 * 32, max_pixels=args.max_pixels)
    config = {'model': provenance['model_id'], 'revision': REVISION,
              'processor': REVISION, 'manifest_sha256': sha256(args.manifest),
              'torch': torch.__version__, 'transformers': transformers.__version__,
              'model_parameter_count': sum(p.numel() for p in model.parameters()),
              'model_trainable_parameter_count': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'input_contract': sorted(ALLOWED), 'max_pixels': args.max_pixels,
              'dtype': str(dtype), 'cache_scope': 'frozen backbone only; invalid after LoRA/weight updates'}
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / 'cache_config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Cache configuration changed; use a new output directory')
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    load_seconds = time.perf_counter() - started
    for row in rows:
        image_path = Path(row['image'])
        if not image_path.is_absolute():
            image_path = args.manifest.parent / image_path
        row_key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
        destination = args.output / (row_key + '.npz')
        image_hash = sha256(image_path)
        if destination.exists():
            with np.load(destination) as previous:
                if str(previous['image_sha256']) == image_hash:
                    continue
            raise ValueError('Image changed underneath existing cache: ' + row['id'])
        tic = time.perf_counter()
        rgb = Image.open(image_path).convert('RGB')
        messages = [{'role': 'user', 'content': [{'type': 'image'},
                     {'type': 'text', 'text': row['instruction']}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[rgb], return_tensors='pt').to(args.device)
        with torch.inference_mode():
            outputs = model.model(**inputs, use_cache=False, return_dict=True)
            hidden = outputs.last_hidden_state[0]
            mask = inputs.attention_mask[0].bool()
            selected = hidden[mask].float()
            mean = selected.mean(0).cpu().numpy()
            last = selected[-1].cpu().numpy()
        if not np.isfinite(mean).all() or not np.isfinite(last).all():
            raise RuntimeError('Non-finite Qwen hidden state for ' + row['id'])
        if args.device == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - tic
        temp = destination.with_suffix('.tmp')
        with temp.open('wb') as f:
            np.savez_compressed(f, mean_hidden=mean, last_hidden=last, id=row['id'],
                                parent_id=row['parent_id'], split=row['split'],
                                image_sha256=image_hash, input_tokens=int(mask.sum()))
        temp.replace(destination)
        record = {'id': row['id'], 'file': destination.name, 'sha256': sha256(destination),
                  'single_request_seconds': elapsed, 'input_tokens': int(mask.sum()),
                  'image_sha256': image_hash}
        print(json.dumps(record), flush=True)
        with (args.output / 'samples.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\n')
    report = {'load_seconds': load_seconds, 'total_seconds': time.perf_counter()-started,
              'samples': len(rows), 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
              'status': 'frozen_rgb_language_feature_extraction_complete',
              'not_evidence_of': ['route training', 'robot performance', 'semantic goal accuracy', 'LoRA training']}
    (args.output / 'status.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
