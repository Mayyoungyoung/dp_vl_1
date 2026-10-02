"""Standard constrained decoding; explicit TRAIN8 probe or full DEV comparison.

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
from routeset.vlm_route_grammar import (GRAMMAR_PROTOCOL, INITIAL, RouteJSONGrammar,
    RouteGrammarLogitsProcessor, TokenGrammarCache, tokenizer_fragments, fragment_fingerprint)


from scripts.evaluate_vlm_route_sft import generation_inputs
from routeset.vlm_sft_train_probe import (train_probe_inputs, generate_train_probe,
    decoding_kwargs, validate_greedy_scope, explicit_greedy_config)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--scope',choices=['train8_preflight','dev24_comparison'],required=True)
    parser.add_argument('--plan',type=Path,default=Path('configs/vlm_depth_interface_train8_v1.json'))
    parser.add_argument('--train8-greedy',action='store_true',
        help='Only fixed TRAIN8 K1, argmax with existing grammar, 180s request-loop boundary')
    args=parser.parse_args()
    validate_greedy_scope(args.train8_greedy,args.scope,args.seed,args.repeats)
    if args.output.exists():raise FileExistsError('Fresh autoregressive output required; do not overwrite or silently retry')
    if args.repeats<1:raise ValueError('Positive independent sampling repeats required')
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='1':raise ValueError('Only authorized physical GPU1')
    import transformers
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration,LogitsProcessorList
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
    if args.train8_greedy and checkpoint_hash!='675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f':
        raise ValueError('Greedy control requires the unchanged original SFT best checkpoint')
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
    if args.scope=='train8_preflight':
        if args.repeats!=1 or args.seed!=0: raise ValueError('Fixed seed0 single TRAIN probe required')
        samples,input_hashes=train_probe_inputs(config,source_hashes,json.loads(args.plan.read_text()))
    else:
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
    decoding=decoding_kwargs(args.train8_greedy)
    effective_generation_config = None
    if args.train8_greedy:
        # Use this explicit object in generate and serialize the same object,
        # instead of reporting unmodified model defaults as effective settings.
        effective_generation_config = explicit_greedy_config(model.generation_config)
    torch.cuda.synchronize();startup=time.perf_counter()-started
    fragments = None
    grammar_caches = {}
    grammar_vocabulary_receipt = {}
    @torch.inference_mode()
    def request(sample,k,decoding_seed,max_tokens):
        nonlocal fragments
        # Each invocation rereads both inputs, preprocesses and reencodes them.
        # KV caching is local to this one autoregressive request only.
        torch.cuda.synchronize();seed_all(decoding_seed)
        observed=read_observation(sample)
        prefix=prepare_prefix(processor,observed,k,24).to('cuda')
        prompt_tokens=prefix['input_ids'].shape[1]
        compilation_seconds = 0.
        if fragments is None:
            compile_start = time.perf_counter()
            fragments = tokenizer_fragments(processor.tokenizer)
            compilation_seconds = time.perf_counter()-compile_start
            grammar_vocabulary_receipt.update(fragment_count=len(fragments),sha256=fragment_fingerprint(fragments),
                compile_seconds_charged_to_first_request=compilation_seconds,cache_capacity=4096)
        eos_ids = model.generation_config.eos_token_id
        eos_ids = [eos_ids] if isinstance(eos_ids,int) else list(eos_ids)
        if k not in grammar_caches:
            grammar_caches[k] = TokenGrammarCache(RouteJSONGrammar(k,24),fragments,eos_ids)
        cache = grammar_caches[k]
        hits_before,misses_before = cache.hits,cache.misses
        constraint = RouteGrammarLogitsProcessor(cache,prompt_tokens)
        if args.train8_greedy:
            if max_tokens != 512: raise ValueError('Fixed greedy512 token budget required')
            result=model.generate(**prefix,logits_processor=LogitsProcessorList([constraint]),
                                  generation_config=effective_generation_config,use_model_defaults=False)
        else:
            result=model.generate(**prefix,max_new_tokens=max_tokens,logits_processor=LogitsProcessorList([constraint]),**decoding)
        output_ids=result[0,prompt_tokens:]
        text=processor.batch_decode(output_ids[None],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        output_tokens=len(output_ids)
        last_token=int(output_ids[-1]) if output_tokens else None
        final_state=INITIAL
        for token in output_ids.tolist():
            if token in eos_ids:
                break
            final_state=cache.grammar.advance(final_state,fragments.get(token,'')) if final_state is not None else None
        grammar_receipt=dict(protocol=GRAMMAR_PROTOCOL,complete=cache.grammar.complete(final_state),
            logits_processor_calls=constraint.calls,logits_processor_python_wall_seconds=constraint.wall_seconds,
            vocabulary_compile_seconds=compilation_seconds,cache_hits=cache.hits-hits_before,
            cache_misses=cache.misses-misses_before,posthoc_repair=False)
        torch.cuda.synchronize()
        return dict(text=text,tokens=dict(prompt_tokens=int(prompt_tokens),output_tokens=output_tokens,
            reached_token_limit=output_tokens>=max_tokens,last_output_token_id=last_token,grammar=grammar_receipt))
    if args.scope=='train8_preflight':
        groups=generate_train_probe(samples,request,args.output,seed=args.seed,greedy=args.train8_greedy)
    else:
        groups=generate_comparison(samples,request,args.output,seed=args.seed,repeats=args.repeats,horizon=24)
    elapsed=time.perf_counter()-started
    files={str(path.relative_to(args.output)):dict(sha256=sha256(path),bytes=path.stat().st_size)
           for path in args.output.rglob('*') if path.is_file()}
    sources={name:sha256(root/name) for name in ('scripts/evaluate_vlm_route_sft_constrained.py','scripts/evaluate_vlm_route_sft.py','routeset/vlm_route_grammar.py','routeset/vlm_sft_train_probe.py','routeset/vlm_sft_evaluation.py',
        'routeset/vlm_sft_data.py','routeset/vlm_route_serialization.py','scripts/train_observed_lora.py')}
    summary=dict(protocol=EVALUATION_PROTOCOL,status='completed',code_commit=os.environ.get('CODE_COMMIT'),
        evaluation_scope=args.scope,split='TRAIN' if args.scope=='train8_preflight' else 'DEV_MODEL',
        sampling_plan_sha256=sha256(args.plan) if args.scope=='train8_preflight' else None,
        syntax_constraint=dict(protocol=GRAMMAR_PROTOCOL,compact_json=True,k_exact=True,horizon=24,coordinate_integer_range=[-10000,10000],event_values=[0,1],
            vocabulary=grammar_vocabulary_receipt,all_constraint_cost_inside_measured_requests=True,posthoc_repair=False),
        checkpoint=str(checkpoint),checkpoint_sha256=checkpoint_hash,checkpoint_step=training['best_step'] if checkpoint.name=='best.pt' else training['step'],
        training_summary_sha256=sha256(run/'summary.json'),config=config,decoding=decoding,
        export_manifest_sha256=sha256(Path(config['observations']).parent/'export_manifest.json'),
        generation_config=model.generation_config.to_dict(),seed=args.seed,repeats=args.repeats,
        candidate_pooling_between_repeats=False,examples=len(samples),parents=8,groups=groups,
        startup_seconds=startup,elapsed_seconds=elapsed,gpu_hours_reserved=elapsed/3600,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),generation_input_sha256=input_hashes,
        source_sha256=sources,artifacts=files,
        scope=('Eight TRAIN target0 observations, one complete K1 request each; no DEV/geometry. ' if args.scope=='train8_preflight' else
               'Same checkpoint, 4 complete independent K1 encodings versus one K4 encoding; all output slots charged. ') +
              'No references/targets/verification geometry in generation. Fixed-K metrics conservatively fail over-budget scenes. '
              'End-to-end generation excludes startup and separate later checker; no learned selection score. '
              'All constraint compilation/masking is included in actual request timing. No output text/route repair. '
              'Repeats are sampling repeats, not training seeds.')
    if args.train8_greedy:
        summary.update(decoding_mode='greedy', effective_generation_config=effective_generation_config.to_dict(),
            generation_use_model_defaults=False,effective_generation_mode=effective_generation_config.get_generation_mode().value,
            inherited_model_generation_config_is_not_effective=True,
            requested_max_new_tokens_per_call=512, requested_calls=8, requested_slots=8,
            request_loop_limit_seconds=180.,
            scope=summary['scope']+' This is a single deterministic TRAIN8 greedy diagnostic, not a sampling repeat or a DEV result.')
    write_json(args.output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('status','checkpoint_step','elapsed_seconds','gpu_hours_reserved','peak_cuda_allocated_bytes','groups')}),flush=True)


if __name__=='__main__':main()
