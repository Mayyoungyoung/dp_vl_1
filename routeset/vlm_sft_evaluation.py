"""Finite, uncached independent-versus-whole generation accounting.

This module has no model or supervision reader. Its callback receives only a
whitelisted observation record, requested K, decoding seed and token limit.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .vlm_route_serialization import parse_paths


EVALUATION_PROTOCOL = 'vlm_sft_same_checkpoint_independent4_whole4_v1'
METHODS = ('independent4', 'whole4')
GENERATION_SAMPLE_KEYS = {'id','parent_id','split','instruction','image_path','observation_path'}


def safe_parse_paths(text,k,horizon):
    """Charge adversarially deep malformed output without ending the group.

    The successful training serializer stays unchanged and fingerprinted.
    This generation-only wrapper catches the JSON parser's recursion failure.
    """
    try:
        return parse_paths(text,k,horizon)
    except RecursionError:
        paths,events,receipt=parse_paths('',k,horizon)
        receipt['errors']=['JSON nesting exceeds parser recursion limit']
        return paths,events,receipt


def request_seed(seed, repeat, scene_id, method, request):
    value = json.dumps([seed,repeat,scene_id,method,request], separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], 'little')


def generate_comparison(samples, generate_request, output, seed=0, repeats=1, horizon=24):
    """Record each attempted request, never retry or pool sampling repeats.

    generate_request returns text and token metadata. Timing wraps its entire
    read/processor/encode/autoregressive/decode call plus strict parsing. Any
    exception is charged as a failed requested pool; raw errors remain visible.
    """
    if repeats < 1 or horizon < 2:
        raise ValueError('Positive repeats and valid horizon required')
    if not samples or len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Unique nonempty DEV observations required')
    for sample in samples:
        if set(sample) != GENERATION_SAMPLE_KEYS or sample['split'] != 'DEV_MODEL':
            raise ValueError('Generation accepts exact DEV observation metadata only')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    summaries = []
    for repeat in range(repeats):
        for method in METHODS:
            folder = output/f'repeat{repeat}'/method
            folder.mkdir(parents=True)
            strict_paths, strict_events, parsed_paths, parsed_events, rows = [], [], [], [], []
            for sample in samples:
                scene_start = time.perf_counter()
                paths, events, requests = [], [], []
                budgets = [1,1,1,1] if method == 'independent4' else [4]
                for number,k in enumerate(budgets):
                    decoding_seed = request_seed(seed,repeat,sample['id'],method,number)
                    started = time.perf_counter()
                    text, token_metadata, failure = '', {}, None
                    try:
                        result = generate_request(sample,k,decoding_seed,512*k)
                        text = result['text']; token_metadata = result['tokens']
                        if not isinstance(text,str):
                            raise TypeError('Generation must return raw decoded text')
                    except Exception as error:
                        failure = dict(type=type(error).__name__, message=str(error))
                        text = ''
                    xyz,opened,receipt = safe_parse_paths(text,k,horizon)
                    elapsed = time.perf_counter()-started
                    record = dict(scene_id=sample['id'],parent_id=sample['parent_id'],repeat=repeat,method=method,
                        request=number,k=k,decoding_seed=decoding_seed,max_new_tokens=512*k,
                        text=text,tokens=token_metadata,failure=failure,parse=receipt,
                        observed_input_to_decoded_routes_seconds=elapsed)
                    requests.append(record); paths.append(xyz); events.append(opened)
                    with (folder/'requests.jsonl').open('a',encoding='utf-8') as handle:
                        handle.write(json.dumps(record,allow_nan=False)+'\n'); handle.flush()
                xyz,opened = np.concatenate(paths),np.concatenate(events)
                charged = sum(r['parse']['charged_candidate_slots'] for r in requests)
                over_budget = charged > 4
                # An overgenerated request makes the entire scene ineligible as
                # a fixed-K4 result. Preserve nominal parses separately; do not
                # pick three successful routes from a five-route attempted pool.
                parsed_paths.append(xyz.copy()); parsed_events.append(opened.copy())
                if over_budget:
                    xyz[:]=np.nan; opened[:]=np.nan
                strict_paths.append(xyz); strict_events.append(opened)
                rows.append(dict(scene_id=sample['id'],parent_id=sample['parent_id'],repeat=repeat,method=method,
                    requested_candidate_slots=4,charged_candidate_slots=charged,
                    budget_exceeded=over_budget,strict_fixed_k_eligible=not over_budget,
                    requested_autoregressive_calls=len(budgets),completed_calls=sum(r['failure'] is None for r in requests),
                    format_valid_requested_slots=sum(r['parse']['format_valid_candidates'] for r in requests),
                    strict_finite_slots=int(np.isfinite(xyz).all(axis=(1,2)).sum()),
                    observed_input_to_routes_seconds=time.perf_counter()-scene_start,
                    request_seconds=sum(r['observed_input_to_decoded_routes_seconds'] for r in requests),
                    output_tokens=sum(r['tokens'].get('output_tokens',0) for r in requests),
                    prompt_tokens=sum(r['tokens'].get('prompt_tokens',0) for r in requests),
                    token_counts_incomplete=any(r['failure'] is not None for r in requests),
                    scope='Generation/read/processor/full encoding/decoding/parse; excludes independent later geometry checker, loading, and learned scoring (absent).'))
            np.savez_compressed(folder/'predictions.npz', scene_ids=np.asarray([s['id'] for s in samples]),
                parent_ids=np.asarray([s['parent_id'] for s in samples]), paths=np.stack(strict_paths),
                gripper_open=np.stack(strict_events), nominal_parsed_paths=np.stack(parsed_paths),
                nominal_parsed_gripper_open=np.stack(parsed_events),
                charged_candidate_slots=np.asarray([r['charged_candidate_slots'] for r in rows]),
                budget_exceeded=np.asarray([r['budget_exceeded'] for r in rows]))
            (folder/'per_scene.json').write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n',encoding='utf-8')
            result = dict(repeat=repeat,method=method,examples=len(samples),requested_candidate_slots=4*len(samples),
                charged_candidate_slots=sum(r['charged_candidate_slots'] for r in rows),
                budget_exceeded_examples=sum(r['budget_exceeded'] for r in rows),
                failed_requests=sum(r['requested_autoregressive_calls']-r['completed_calls'] for r in rows),
                strict_finite_slots=sum(r['strict_finite_slots'] for r in rows),
                total_generation_seconds=sum(r['observed_input_to_routes_seconds'] for r in rows),
                generation_ms_median=float(np.median([r['observed_input_to_routes_seconds'] for r in rows])*1000),
                generation_ms_p95=float(np.percentile([r['observed_input_to_routes_seconds'] for r in rows],95)*1000),
                first_request_in_this_group=rows[0],candidate_pooling_between_repeats=False,
                geometric_metrics=None)
            summaries.append(result)
            (folder/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return summaries
