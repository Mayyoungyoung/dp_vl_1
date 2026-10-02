"""Autoregressive independent4 versus whole4 from one completed SFT checkpoint.

Only observed RGB/depth/language/camera/current-state records enter generation.
Reference paths/targets/verification geometry are not opened here. A separate
CPU analysis may inspect them after the complete prediction pool is frozen.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from routeset.common import seed_all,sha256,write_json
from routeset.observed_route_head import QWEN_REVISION
from routeset.vlm_sft_data import (TRAINING_PROTOCOL, prepare_prefix, read_observation,
                                  resolve, validate_reserved_observation_manifest)
from routeset.vlm_sft_evaluation import EVALUATION_PROTOCOL,generate_comparison
from scripts.train_observed_lora import install_lora,load_adapters


def generation_inputs(config, source_hashes):
    observations,supervision=Path(config['observations']),Path(config['supervision'])
    rows=validate_reserved_observation_manifest(observations)
    for path in (observations,supervision):
        if source_hashes.get(str(path.resolve()))!=sha256(path):
            raise ValueError('Original SFT manifest changed')
    # Supervision manifest supplies only the current observation file identity.
    # Future routes, semantic targets, route types and verification files are
    # neither joined to these generation records nor opened by this entrypoint.
    labels={row['id']:row for row in map(json.loads,supervision.read_text(encoding='utf-8-sig').splitlines()) if row['split']=='DEV_MODEL'}
    selected=sorted([row for row in rows if row['split']=='DEV_MODEL'],key=lambda row:row['id'])
    if set(labels)!={row['id'] for row in selected} or len(selected)!=24:
        raise ValueError('All 24 reserved DEV instructions, including no-reference ones, required')
    samples,hashes=[],{}
    for row in selected:
        label=labels[row['id']]
        if label['parent_id']!=row['parent_id'] or label['split']!=row['split']:
            raise ValueError('DEV observation identity mismatch')
        image=resolve(observations.parent,row['image']).resolve()
        observed=resolve(supervision.parent,label['observation']).resolve()
        if image.parent.name!=row['parent_id'] or observed.parent.name!=row['parent_id']:
            raise ValueError('DEV current input resolves outside selected parent')
        for path in (image,observed):
            hashes[str(path)]=sha256(path)
            if source_hashes.get(str(path))!=hashes[str(path)]:
                raise ValueError('Observed input changed since SFT training')
        samples.append(dict(id=row['id'],parent_id=row['parent_id'],split=row['split'],instruction=row['instruction'],
                            image_path=str(image),observation_path=str(observed)))
    return samples,hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--repeats',type=int,default=1)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Fresh autoregressive output required; do not overwrite or silently retry')
    if args.repeats<1:raise ValueError('Positive independent sampling repeats required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Only authorized physical GPU1')
    import transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    if transformers.__version__!='4.57.1' or not torch.__version__.startswith('2.4.1'):
        raise ValueError('Pinned actual Qwen runtime required')
    torch.set_num_threads(1);torch.cuda.set_per_process_memory_fraction(.35);torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter();checkpoint=args.checkpoint.resolve();run=checkpoint.parent
    training=json.loads((run/'summary.json').read_text())
    if training['status']!='completed' or checkpoint.name not in ('best.pt','last.pt'):
        raise ValueError('Completed SFT run and its unchanged best/last checkpoint required')
    checkpoint_hash=sha256(checkpoint)
    if training['artifacts_sha256'].get(checkpoint.name)!=checkpoint_hash:
        raise ValueError('SFT checkpoint hash differs from completed-run receipt')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    config=saved['config']
    if saved['protocol']!=TRAINING_PROTOCOL or config['model_revision']!=QWEN_REVISION or config['horizon']!=24:
        raise ValueError('Pinned K1/K4 SFT checkpoint protocol required')
    if config!=json.loads((run/'config.json').read_text()):raise ValueError('Training config changed')
    source_hashes=json.loads((run/'data_source_hashes.json').read_text())
    if hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest()!=config['data_fingerprint']:
        raise ValueError('Training data source index changed')
    root=Path(__file__).resolve().parents[1]
    for name in ('routeset/vlm_sft_data.py','routeset/vlm_route_serialization.py','scripts/train_observed_lora.py'):
        if sha256(root/name)!=config['source_sha256'][name]:
            raise ValueError('Runtime SFT model/input helper differs from trained source: '+name)
    samples,input_hashes=generation_inputs(config,source_hashes)
    model_path=Path(config['model'])
    if sha256(model_path/'provenance.json')!=config['model_provenance_sha256']:
        raise ValueError('Pinned backbone provenance changed')
    processor=AutoProcessor.from_pretrained(model_path,local_files_only=True,
                                            min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
    model=Qwen3VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'})
    install_lora(model,rank=config['lora_rank'],alpha=config['lora_alpha']);load_adapters(model,saved['adapters'])
    model.requires_grad_(False);model.eval();del saved
    decoding=dict(do_sample=True,temperature=.7,top_p=.9,top_k=0,num_beams=1,num_return_sequences=1,
                  repetition_penalty=1.,use_cache=True)
    torch.cuda.synchronize();startup=time.perf_counter()-started
    @torch.inference_mode()
    def request(sample,k,decoding_seed,max_tokens):
        # Each invocation rereads both inputs, preprocesses and reencodes them.
        # KV caching is local to this one autoregressive request only.
        torch.cuda.synchronize();seed_all(decoding_seed)
        observed=read_observation(sample)
        prefix=prepare_prefix(processor,observed,k,24).to('cuda')
        prompt_tokens=prefix['input_ids'].shape[1]
        result=model.generate(**prefix,max_new_tokens=max_tokens,**decoding)
        output_ids=result[0,prompt_tokens:]
        text=processor.batch_decode(output_ids[None],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        output_tokens=len(output_ids)
        last_token=int(output_ids[-1]) if output_tokens else None
        torch.cuda.synchronize()
        return dict(text=text,tokens=dict(prompt_tokens=int(prompt_tokens),output_tokens=output_tokens,
            reached_token_limit=output_tokens>=max_tokens,last_output_token_id=last_token))
    groups=generate_comparison(samples,request,args.output,seed=args.seed,repeats=args.repeats,horizon=24)
    elapsed=time.perf_counter()-started
    files={str(path.relative_to(args.output)):dict(sha256=sha256(path),bytes=path.stat().st_size)
           for path in args.output.rglob('*') if path.is_file()}
    sources={name:sha256(root/name) for name in ('scripts/evaluate_vlm_route_sft.py','routeset/vlm_sft_evaluation.py',
        'routeset/vlm_sft_data.py','routeset/vlm_route_serialization.py','scripts/train_observed_lora.py')}
    summary=dict(protocol=EVALUATION_PROTOCOL,status='completed',code_commit=os.environ.get('CODE_COMMIT'),
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,checkpoint_step=training['best_step'] if checkpoint.name=='best.pt' else training['step'],
        training_summary_sha256=sha256(run/'summary.json'),config=config,decoding=decoding,
        export_manifest_sha256=sha256(Path(config['observations']).parent/'export_manifest.json'),
        generation_config=model.generation_config.to_dict(),seed=args.seed,repeats=args.repeats,
        candidate_pooling_between_repeats=False,examples=24,parents=8,groups=groups,
        startup_seconds=startup,elapsed_seconds=elapsed,gpu_hours_reserved=elapsed/3600,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),generation_input_sha256=input_hashes,
        source_sha256=sources,artifacts=files,
        scope='Same checkpoint, 4 complete independent K1 encodings versus one K4 encoding; all output slots charged. '
              'No references/targets/verification geometry in generation. Fixed-K metrics conservatively fail over-budget scenes. '
              'End-to-end generation excludes startup and separate later checker; no learned selection score. '
              'Repeats are sampling repeats, not training seeds.')
    write_json(args.output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('status','checkpoint_step','elapsed_seconds','gpu_hours_reserved','peak_cuda_allocated_bytes','groups')}),flush=True)


if __name__=='__main__':main()
