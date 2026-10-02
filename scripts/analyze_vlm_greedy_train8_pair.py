"""Frozen same-checkpoint TRAIN8 sampling/greedy control, unchanged target check."""
import argparse
import json
from pathlib import Path

import numpy as np

from routeset.common import sha256, write_json
from routeset.vlm_sft_evaluation import safe_parse_paths
from routeset.vlm_sft_train_probe import PARENTS, decoding_kwargs

OLD_SUMMARY_SHA = '530c7621ff507669bef53252ccab31f8cf29d300347702c18722494869175e94'
CHECKPOINT_SHA = '675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f'
SUPERVISION_SHA = '37da38855af0357732ac48f125f140e8ed5d1a60008f1424fd5080c7a9421237'


def validate_pair_receipts(old, new):
    """Information, model and budget must match, before target metadata access."""
    for generated in (old, new):
        if (generated['status'] != 'completed' or generated['evaluation_scope'] != 'train8_preflight'
                or generated['split'] != 'TRAIN' or generated['examples'] != 8
                or generated['seed'] != 0 or generated['repeats'] != 1
                or generated['checkpoint_sha256'] != CHECKPOINT_SHA):
            raise ValueError('Complete fixed same-checkpoint TRAIN8 receipts required')
    for key in ('checkpoint_sha256','checkpoint_step','config','generation_input_sha256',
                'sampling_plan_sha256','training_summary_sha256','export_manifest_sha256'):
        if old[key] != new[key]:
            raise ValueError('Unpaired generation receipt: '+key)
    for key in ('protocol','compact_json','k_exact','horizon','coordinate_integer_range','event_values','posthoc_repair'):
        if old['syntax_constraint'][key] != new['syntax_constraint'][key]:
            raise ValueError('Syntax rule changed: '+key)
    if old['syntax_constraint']['vocabulary']['sha256'] != new['syntax_constraint']['vocabulary']['sha256']:
        raise ValueError('Actual tokenizer vocabulary changed')
    # The grammar and all trained observation/model helpers are unchanged.
    for name in ('routeset/vlm_route_grammar.py','routeset/vlm_sft_data.py',
                 'routeset/vlm_route_serialization.py','scripts/train_observed_lora.py'):
        if old['source_sha256'][name] != new['source_sha256'][name]:
            raise ValueError('Shared model/input/grammar source changed: '+name)
    if old['decoding'] != decoding_kwargs(False) or new['decoding'] != decoding_kwargs(True):
        raise ValueError('Only prespecified stochastic-versus-greedy decoding is allowed')
    if new.get('decoding_mode') != 'greedy' or new.get('request_loop_limit_seconds') != 180.:
        raise ValueError('Explicit greedy control and fixed request-loop cap required')
    if new.get('generation_use_model_defaults') is not False or new.get('effective_generation_mode') != 'greedy_search':
        raise ValueError('Model defaults must not override greedy mode')
    effective = new['effective_generation_config']
    for key, value in decoding_kwargs(True).items():
        if effective.get(key) != value:
            raise ValueError('Reported actual generation configuration differs: '+key)
    if effective.get('max_new_tokens') != 512:
        raise ValueError('Actual greedy configuration must retain max_new_tokens512')


def frozen_probe(path, expected_method):
    generated = json.loads((path/'summary.json').read_text())
    for name, receipt in generated['artifacts'].items():
        target = (path/name).resolve()
        if path.resolve() not in target.parents or sha256(target) != receipt['sha256']:
            raise ValueError('Frozen generation artifact changed')
    requests = [json.loads(line) for line in (path/'requests.jsonl').read_text().splitlines()]
    if len(requests) != 8 or [r['parent_id'] for r in requests] != PARENTS:
        raise ValueError('All eight requested slots must be frozen before label access')
    paths, events = [], []
    for r in requests:
        if (r['scene_id'] != r['parent_id']+'_target0' or r['split'] != 'TRAIN' or r['method'] != expected_method
                or r['k'] != 1 or r['horizon'] != 24 or r['max_new_tokens'] != 512 or r['repeat'] != 0 or r['request'] != 0):
            raise ValueError('Predeclared TRAIN8 request identity/budget changed')
        xyz, opened, receipt = safe_parse_paths(r['text'], 1, 24)
        if receipt != r['parse']:
            raise ValueError('Unmodified strict parse receipt changed')
        paths.append(xyz); events.append(opened)
    paths, events = np.stack(paths), np.stack(events)
    with np.load(path/'predictions.npz', allow_pickle=False) as saved:
        if (saved['scene_ids'].tolist() != [r['scene_id'] for r in requests] or saved['parent_ids'].tolist() != PARENTS
                or not np.array_equal(saved['paths'], paths, equal_nan=True)
                or not np.array_equal(saved['gripper_open'], events, equal_nan=True)):
            raise ValueError('Original prediction bytes and raw text disagree')
    return generated, requests, paths, events


def endpoint_metrics(paths, events, targets):
    """All eight slots contribute to success counts; missing errors stay null."""
    rows = []
    for i, target in enumerate(targets):
        finite = bool(np.isfinite(paths[i]).all() and np.isfinite(events[i]).all())
        error = nearest = None; within = correct = False
        if finite:
            distances = np.linalg.norm(np.asarray(target['centers'], dtype=np.float64)-paths[i,0,-1].astype(np.float64), axis=-1)
            error = float(distances[target['target_index']]); nearest = int(distances.argmin())
            within = error <= .03; correct = within and nearest == target['target_index']
        rows.append(dict(scene_id=PARENTS[i]+'_target0', strict_format_finite=finite,
            requested_target_distance_m=error, nearest_target_index=nearest,
            within_3cm=within, correct_requested_target=correct))
    errors = [r['requested_target_distance_m'] for r in rows if r['requested_target_distance_m'] is not None]
    all_eight_mean = float(np.mean(errors)) if len(errors) == 8 else None
    correct = sum(r['correct_requested_target'] for r in rows)
    within = sum(r['within_3cm'] for r in rows)
    return dict(requested_slots=8, strict_format_finite=len(errors),
        within_3cm_count=within, correct_requested_target_count=correct,
        endpoint_error_mean_m_over_finite_only=float(np.mean(errors)) if errors else None,
        endpoint_error_median_m_over_finite_only=float(np.median(errors)) if errors else None,
        all_eight_mean_endpoint_error_m=all_eight_mean,
        predeclared_useful_control=(within >= 4 and all_eight_mean is not None and all_eight_mean <= .15), rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sampling', type=Path, required=True)
    parser.add_argument('--greedy', type=Path, required=True)
    parser.add_argument('--supervision-metadata', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Fresh paired diagnostic required')
    if sha256(args.sampling/'summary.json') != OLD_SUMMARY_SHA:
        raise ValueError('Exactly the original frozen stochastic TRAIN8 required')
    old, old_requests, old_paths, old_events = frozen_probe(args.sampling, 'train8_constrained_k1')
    new, new_requests, new_paths, new_events = frozen_probe(args.greedy, 'train8_constrained_greedy_k1')
    validate_pair_receipts(old, new)
    if [r['decoding_seed'] for r in old_requests] != [r['decoding_seed'] for r in new_requests]:
        raise ValueError('Request seed identities changed; no extra trial allowed')
    if sha256(args.supervision_metadata) != SUPERVISION_SHA:
        raise ValueError('Original target metadata changed')
    # Only after both complete outputs and paired receipts are frozen.
    wanted = {p+'_target0' for p in PARENTS}
    labels = {r['id']:r for r in map(json.loads,args.supervision_metadata.read_text().splitlines()) if r['id'] in wanted}
    if set(labels) != wanted:
        raise ValueError('Exact TRAIN8 target join required')
    targets = []
    for parent in PARENTS:
        label = labels[parent+'_target0']
        if label['split'] != 'TRAIN' or label['parent_id'] != parent or label['semantic_targets']['tolerance'] != .03:
            raise ValueError('Unchanged TRAIN parent and 3cm target standard required')
        targets.append(label['semantic_targets'])
    sampling = endpoint_metrics(old_paths, old_events, targets)
    greedy = endpoint_metrics(new_paths, new_events, targets)
    pairs = [dict(scene_id=a['scene_id'], sampling=a, greedy=b,
        greedy_minus_sampling_distance_m=b['requested_target_distance_m']-a['requested_target_distance_m']
            if a['requested_target_distance_m'] is not None and b['requested_target_distance_m'] is not None else None)
        for a,b in zip(sampling['rows'],greedy['rows'])]
    result = dict(protocol='vlm_greedy_sampling_train8_pair_v1', split='TRAIN', status='completed',
        sampling_summary_sha256=OLD_SUMMARY_SHA, greedy_summary_sha256=sha256(args.greedy/'summary.json'),
        supervision_metadata_sha256=SUPERVISION_SHA, analyzer_sha256=sha256(__file__),
        checkpoint_sha256=CHECKPOINT_SHA, paired_receipts_verified=True,
        sampling=sampling, greedy=greedy, paired_scenes=pairs,
        new_attempted_autoregressive_calls=new['groups'][0]['attempted_autoregressive_calls'],
        new_budget_unattempted_slots=new['groups'][0]['budget_unattempted_slots'],
        new_elapsed_seconds=new['elapsed_seconds'], new_gpu_hours_reserved=new['gpu_hours_reserved'],
        new_peak_cuda_allocated_bytes=new['peak_cuda_allocated_bytes'],
        request_loop_seconds=new['groups'][0]['request_loop_seconds'],
        request_loop_budget_overshoot_seconds=new['groups'][0]['request_loop_budget_overshoot_seconds'],
        additional_model_calls=0, raw_reference_paths_opened=0, verification_geometry_opened=0,
        scope='One fixed TRAIN-only sampling realization versus greedy; no variance/generalization estimate, '
              'no repaired candidates or collision/execution claim. Failure slots retain their denominator.')
    args.output.mkdir(parents=True); write_json(args.output/'summary.json', result)
    print(json.dumps({key:result[key] for key in ('paired_receipts_verified','new_attempted_autoregressive_calls',
        'new_budget_unattempted_slots','new_elapsed_seconds')},indent=2))
    print(json.dumps(dict(sampling_correct=sampling['correct_requested_target_count'],
        greedy_correct=greedy['correct_requested_target_count'],
        greedy_mean_endpoint_m=greedy['all_eight_mean_endpoint_error_m'],
        predeclared_useful_control=greedy['predeclared_useful_control']),indent=2))


if __name__ == '__main__':
    main()
