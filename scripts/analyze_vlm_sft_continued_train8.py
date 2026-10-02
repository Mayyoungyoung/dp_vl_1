"""Compare frozen original/continued TRAIN8 greedy pools; additional training cost explicit."""
import argparse
import json
from pathlib import Path

from routeset.common import sha256, write_json
from routeset.vlm_sft_continuation import validate_continuation_config
from routeset.vlm_sft_continued_eval import inspect_completed_run, ORIGINAL_FILES, PROBE_PROTOCOL
from routeset.vlm_sft_train_probe import PARENTS, decoding_kwargs
from scripts.analyze_vlm_greedy_train8_pair import frozen_probe, endpoint_metrics, SUPERVISION_SHA

ORIGINAL_GREEDY_SUMMARY = '228506dbb94b80cad93ce5cc9d23c3841123447b43fbb52dade7963a5f1cc4c2'


def validate_pair(old, new, config, lineage_receipt):
    if (old['checkpoint_sha256'] != ORIGINAL_FILES['best.pt'] or old['checkpoint_step'] != 1500
            or old['evaluation_scope'] != 'train8_preflight'
            or new['evaluation_scope'] != 'continued_train8_preflight' or new['protocol'] != PROBE_PROTOCOL
            or new['continuation_lineage'] != lineage_receipt or new['config'] != config
            or new['checkpoint_sha256'] != lineage_receipt['selected_checkpoint_sha256']
            or new['checkpoint_step'] != lineage_receipt['selected_step']
            or new['training_summary_sha256'] != lineage_receipt['training_summary_sha256']):
        raise ValueError('Original/new checkpoint identity or verified continuation lineage differs')
    validate_continuation_config(new['config'],old['config'])
    for result in (old,new):
        if (result['status'] != 'completed' or result['split'] != 'TRAIN' or result['examples'] != 8
                or result['parents'] != 8 or result['seed'] != 0 or result['repeats'] != 1
                or result['requested_calls'] != 8 or result['requested_slots'] != 8
                or result['requested_max_new_tokens_per_call'] != 512
                or result['decoding'] != decoding_kwargs(True) or result['decoding_mode'] != 'greedy'
                or result['generation_use_model_defaults'] is not False
                or result['effective_generation_mode'] != 'greedy_search'
                or result['request_loop_limit_seconds'] != 180.):
            raise ValueError('Only complete unchanged-budget greedy TRAIN8 pools allowed')
    for key in ('generation_input_sha256','sampling_plan_sha256','export_manifest_sha256','effective_generation_config'):
        if old[key] != new[key]: raise ValueError('Shared input/decoder protocol changed: '+key)
    for key in ('protocol','compact_json','k_exact','horizon','coordinate_integer_range','event_values','posthoc_repair'):
        if old['syntax_constraint'][key] != new['syntax_constraint'][key]:
            raise ValueError('Syntax rules changed')
    if old['syntax_constraint']['vocabulary']['sha256'] != new['syntax_constraint']['vocabulary']['sha256']:
        raise ValueError('Actual tokenizer fragments changed')
    for name in ('routeset/vlm_route_grammar.py','routeset/vlm_sft_train_probe.py','routeset/vlm_sft_data.py',
                 'routeset/vlm_route_serialization.py','scripts/train_observed_lora.py'):
        if old['source_sha256'][name] != new['source_sha256'][name]:
            raise ValueError('Shared generation helper changed: '+name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--original-greedy',type=Path,required=True)
    parser.add_argument('--continued-greedy',type=Path,required=True)
    parser.add_argument('--supervision-metadata',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Fresh paired diagnostic required')
    if sha256(args.original_greedy/'summary.json') != ORIGINAL_GREEDY_SUMMARY:
        raise ValueError('Exactly the original frozen greedy result required')
    training,config,lineage,index,receipt = inspect_completed_run(args.run)
    old,orequests,opaths,oevents = frozen_probe(args.original_greedy,'train8_constrained_greedy_k1')
    new,nrequests,npaths,nevents = frozen_probe(args.continued_greedy,'train8_constrained_greedy_k1')
    validate_pair(old,new,config,receipt)
    if [r['decoding_seed'] for r in orequests] != [r['decoding_seed'] for r in nrequests]:
        raise ValueError('Request seed identity changed')
    if sha256(args.supervision_metadata) != SUPERVISION_SHA:
        raise ValueError('Original semantic target metadata changed')
    # No labels are opened until both complete pools and training lineage pass.
    wanted = {p+'_target0' for p in PARENTS}
    selected = [r for r in map(json.loads,args.supervision_metadata.read_text().splitlines()) if r['id'] in wanted]
    labels = {r['id']:r for r in selected}
    if len(selected) != 8 or set(labels) != wanted: raise ValueError('Exact eight TRAIN target join required')
    targets = []
    for p in PARENTS:
        label = labels[p+'_target0']
        if label['split'] != 'TRAIN' or label['parent_id'] != p or label['semantic_targets']['tolerance'] != .03:
            raise ValueError('Unchanged TRAIN role and 3cm standard required')
        targets.append(label['semantic_targets'])
    original = endpoint_metrics(opaths,oevents,targets)
    continued = endpoint_metrics(npaths,nevents,targets)
    pairs = [dict(scene_id=a['scene_id'],original=a,continued=b,
        continued_minus_original_distance_m=b['requested_target_distance_m']-a['requested_target_distance_m']
            if a['requested_target_distance_m'] is not None and b['requested_target_distance_m'] is not None else None)
        for a,b in zip(original['rows'],continued['rows'])]
    summary = dict(protocol='vlm_original_vs_continued_train8_greedy_v1',status='completed',split='TRAIN',
        original_summary_sha256=ORIGINAL_GREEDY_SUMMARY,continued_summary_sha256=sha256(args.continued_greedy/'summary.json'),
        continuation_lineage=receipt,supervision_metadata_sha256=SUPERVISION_SHA,analyzer_sha256=sha256(__file__),
        paired_receipts_verified=True,original=original,continued=continued,paired_scenes=pairs,
        continuation_training_accounting=training['continuation_accounting'],
        generation_elapsed_seconds=new['elapsed_seconds'],generation_gpu_hours_reserved=new['gpu_hours_reserved'],
        generation_peak_cuda_allocated_bytes=new['peak_cuda_allocated_bytes'],
        generation_group=new['groups'][0],additional_model_calls=0,raw_reference_paths_opened=0,verification_geometry_opened=0,
        scope='Original1500 versus DEV-token-NLL-selected checkpoint from bounded6000; additional training is the intervention. '
              'Eight fixed TRAIN observations, no generalization/robot collision/execution evidence or mechanism novelty. '
              'All requested slots retained. No repairs or post-hoc checkpoint selection.')
    args.output.mkdir(parents=True); write_json(args.output/'summary.json',summary)
    print(json.dumps(dict(original=original,continued=continued),indent=2))


if __name__ == '__main__': main()
