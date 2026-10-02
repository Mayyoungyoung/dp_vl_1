"""One greedy TRAIN8 probe for a verified newly selected continuation checkpoint.

No reference paths, target coordinates or verification geometry are opened.
The original fixed-SHA same-checkpoint evaluation entry remains unchanged.
"""
import argparse
import json
import os
from pathlib import Path
import time

import torch

from routeset.common import seed_all, sha256, write_json
from routeset.vlm_sft_continued_eval import inspect_completed_run, validate_saved_checkpoint, PROBE_PROTOCOL
from routeset.vlm_sft_data import prepare_prefix, read_observation
from routeset.vlm_sft_train_probe import (train_probe_inputs, generate_train_probe,
    decoding_kwargs, explicit_greedy_config)
from routeset.vlm_route_grammar import (GRAMMAR_PROTOCOL, INITIAL, RouteJSONGrammar,
    RouteGrammarLogitsProcessor, TokenGrammarCache, tokenizer_fragments, fragment_fingerprint)
from scripts.train_observed_lora import install_lora, load_adapters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--plan',type=Path,default=Path('configs/vlm_depth_interface_train8_v1.json'))
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh TRAIN8 output required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1': raise ValueError('Only authorized physical GPU1')
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, LogitsProcessorList
    if transformers.__version__ != '4.57.1' or not torch.__version__.startswith('2.4.1'):
        raise ValueError('Pinned actual Qwen runtime required')
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.35)
    torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
    run = args.run.resolve()
    training,config,lineage,index,lineage_receipt = inspect_completed_run(run)
    saved = torch.load(run/'best.pt',map_location='cpu',weights_only=False)
    validate_saved_checkpoint(saved,training,config,lineage)
    root = Path(__file__).resolve().parents[1]
    shared = ('routeset/vlm_sft_data.py','routeset/vlm_route_serialization.py','scripts/train_observed_lora.py')
    for name in shared:
        if sha256(root/name) != config['source_sha256'][name]:
            raise ValueError('Trained model/input helper changed: '+name)
    samples,input_hashes = train_probe_inputs(config,index,json.loads(args.plan.read_text()))
    model_path = Path(config['model'])
    if sha256(model_path/'provenance.json') != config['model_provenance_sha256']:
        raise ValueError('Pinned backbone provenance changed')
    processor = AutoProcessor.from_pretrained(model_path,local_files_only=True,
        min_pixels=config['min_pixels'],max_pixels=config['max_pixels'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa',device_map={'':'cuda'})
    install_lora(model,rank=config['lora_rank'],alpha=config['lora_alpha'])
    load_adapters(model,saved['adapters']); model.requires_grad_(False); model.eval(); del saved
    effective = explicit_greedy_config(model.generation_config)
    torch.cuda.synchronize(); startup = time.perf_counter()-started
    fragments,cache = None,None
    vocabulary = {}

    @torch.inference_mode()
    def request(sample,k,decoding_seed,max_tokens):
        nonlocal fragments,cache
        if k != 1 or max_tokens != 512: raise ValueError('Only fixed K1/512 TRAIN probe')
        torch.cuda.synchronize(); seed_all(decoding_seed)
        observed = read_observation(sample)
        prefix = prepare_prefix(processor,observed,1,24).to('cuda')
        prompt_tokens = prefix['input_ids'].shape[1]
        compile_seconds = 0.
        eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else list(eos)
        if fragments is None:
            tic = time.perf_counter(); fragments = tokenizer_fragments(processor.tokenizer)
            compile_seconds = time.perf_counter()-tic
            vocabulary.update(fragment_count=len(fragments),sha256=fragment_fingerprint(fragments),
                compile_seconds_charged_to_first_request=compile_seconds,cache_capacity=4096)
            cache = TokenGrammarCache(RouteJSONGrammar(1,24),fragments,eos)
        hits,misses = cache.hits,cache.misses
        constraint = RouteGrammarLogitsProcessor(cache,prompt_tokens)
        result = model.generate(**prefix,logits_processor=LogitsProcessorList([constraint]),
            generation_config=effective,use_model_defaults=False)
        ids = result[0,prompt_tokens:]
        text = processor.batch_decode(ids[None],skip_special_tokens=True,clean_up_tokenization_spaces=False)[0]
        state = INITIAL
        for token in ids.tolist():
            if token in eos: break
            state = cache.grammar.advance(state,fragments.get(token,'')) if state is not None else None
        receipt = dict(protocol=GRAMMAR_PROTOCOL,complete=cache.grammar.complete(state),
            logits_processor_calls=constraint.calls,logits_processor_python_wall_seconds=constraint.wall_seconds,
            vocabulary_compile_seconds=compile_seconds,cache_hits=cache.hits-hits,cache_misses=cache.misses-misses,
            posthoc_repair=False)
        torch.cuda.synchronize()
        return dict(text=text,tokens=dict(prompt_tokens=int(prompt_tokens),output_tokens=len(ids),
            reached_token_limit=len(ids)>=512,last_output_token_id=int(ids[-1]) if len(ids) else None,grammar=receipt))

    groups = generate_train_probe(samples,request,args.output,seed=0,greedy=True)
    elapsed = time.perf_counter()-started
    sources = ('scripts/evaluate_vlm_sft_continued_train8.py','routeset/vlm_sft_continued_eval.py',
        'routeset/vlm_sft_continuation.py','routeset/vlm_sft_train_probe.py','routeset/vlm_route_grammar.py',
        'routeset/vlm_sft_evaluation.py')+shared
    summary = dict(protocol=PROBE_PROTOCOL,status='completed',code_commit=os.environ.get('CODE_COMMIT'),
        evaluation_scope='continued_train8_preflight',split='TRAIN',seed=0,repeats=1,examples=8,parents=8,
        checkpoint=str(run/'best.pt'),checkpoint_sha256=lineage_receipt['selected_checkpoint_sha256'],
        checkpoint_step=training['best_step'],training_summary_sha256=lineage_receipt['training_summary_sha256'],
        continuation_lineage=lineage_receipt,config=config,groups=groups,
        decoding=decoding_kwargs(True),decoding_mode='greedy',effective_generation_config=effective.to_dict(),
        generation_config=model.generation_config.to_dict(),inherited_model_generation_config_is_not_effective=True,
        generation_use_model_defaults=False,effective_generation_mode=effective.get_generation_mode().value,
        requested_max_new_tokens_per_call=512,requested_calls=8,requested_slots=8,request_loop_limit_seconds=180.,
        candidate_pooling_between_repeats=False,sampling_plan_sha256=sha256(args.plan),
        export_manifest_sha256=sha256(Path(config['observations']).parent/'export_manifest.json'),
        generation_input_sha256=input_hashes,startup_seconds=startup,elapsed_seconds=elapsed,
        gpu_hours_reserved=elapsed/3600,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        syntax_constraint=dict(protocol=GRAMMAR_PROTOCOL,compact_json=True,k_exact=True,horizon=24,
            coordinate_integer_range=[-10000,10000],event_values=[0,1],vocabulary=vocabulary,
            all_constraint_cost_inside_measured_requests=True,posthoc_repair=False),
        source_sha256={name:sha256(root/name) for name in sources},
        artifacts={str(p.relative_to(args.output)):dict(sha256=sha256(p),bytes=p.stat().st_size)
                   for p in args.output.rglob('*') if p.is_file()},
        scope='One fixed TRAIN8 K1 greedy probe after additional training; not same-checkpoint comparison or DEV evidence. '
              'No references/targets/verification geometry in generation. Each call rereads/preprocesses/reencodes inputs. '
              'All syntax masking/compilation is timed; startup and later checker are separate. '
              'Eight slots including failures; 180s checked only before requests, with overshoot reported. No retry or repair.')
    write_json(args.output/'summary.json',summary)
    print(json.dumps({key:summary[key] for key in ('status','checkpoint_step','elapsed_seconds','groups')}),flush=True)


if __name__ == '__main__': main()
