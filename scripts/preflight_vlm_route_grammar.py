"""CPU-only actual pinned tokenizer and grammar check using synthetic JSON."""
import argparse
import json
import os
from pathlib import Path
import time

from routeset.common import sha256, write_json
from routeset.vlm_route_grammar import (GRAMMAR_PROTOCOL, INITIAL, RouteJSONGrammar, TokenGrammarCache,
                                      fragment_fingerprint, tokenizer_fragments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh tokenizer preflight required')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('', '-1'): raise ValueError('CPU only; hide GPUs')
    import torch
    import transformers
    from transformers import AutoTokenizer
    if transformers.__version__ != '4.57.1' or not torch.__version__.startswith('2.4.1'):
        raise ValueError('Original runtime required')
    torch.set_num_threads(1); started = time.perf_counter()
    config = json.loads((args.training/'config.json').read_text())
    if config['model_revision'] != '89644892e4d85e24eaac8bacfd4f463576704203': raise ValueError('Pinned Qwen required')
    model = Path(config['model'])
    if sha256(model/'provenance.json') != config['model_provenance_sha256']: raise ValueError('Model provenance changed')
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    compile_start = time.perf_counter(); fragments = tokenizer_fragments(tokenizer)
    compile_seconds = time.perf_counter()-compile_start
    generation_config = json.loads((model/'generation_config.json').read_text())
    eos_ids = generation_config['eos_token_id']
    eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
    records = []
    for k in (1, 4):
        cache = TokenGrammarCache(RouteJSONGrammar(k, 24), fragments, eos_ids)
        for case in ('bounds', 'varying'):
            routes = []
            for route in range(k):
                points = [[10000, -10000, 0, 1] if case == 'bounds' else
                          [((route*1777+j*829+a*991)%20001)-10000 for a in range(3)]+[j%2] for j in range(24)]
                routes.append(points)
            text = json.dumps(routes, separators=(',', ':'))
            tokens = tokenizer.encode(text, add_special_tokens=False)
            state = INITIAL; lengths = []
            for token in tokens:
                allowed = cache.allowed(state); lengths.append(len(allowed))
                if token not in allowed: raise ValueError('Actual encoded legal text rejected by grammar')
                state = cache.grammar.advance(state, fragments[token])
            if not cache.grammar.complete(state) or cache.allowed(state) != tuple(sorted(eos_ids)):
                raise ValueError('Complete route must allow only EOS')
            if tokenizer.decode(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
                raise ValueError('Actual synthetic token round-trip changed')
            records.append(dict(k=k, horizon=24, case=case, tokens=len(tokens),
                legal_choices_min=min(lengths), legal_choices_max=max(lengths), complete=True,
                raw_text_sha256=__import__('hashlib').sha256(text.encode()).hexdigest(), cache_hits=cache.hits, cache_misses=cache.misses))
    args.output.mkdir(parents=True)
    write_json(args.output/'summary.json', dict(protocol=GRAMMAR_PROTOCOL, status='completed',
        code_commit=os.environ.get('CODE_COMMIT'), actual_tokenizer_class=type(tokenizer).__name__,
        fragments=len(fragments), fragment_sha256=fragment_fingerprint(fragments), eos_ids=eos_ids,
        compile_seconds=compile_seconds, elapsed_seconds=time.perf_counter()-started, cpu_threads=1, gpu_hours=0,
        model_forwards=0, observations_opened=0, synthetic_cases=records,
        source_sha256={str(path):sha256(path) for path in (Path(__file__),Path(__file__).parents[1]/'routeset/vlm_route_grammar.py')},
        scope='Actual tokenizer checks only; synthetic legal JSON is never a model condition or geometric result. Does not establish generation quality.'))
    print((args.output/'summary.json').read_text(), flush=True)


if __name__ == '__main__': main()
