"""Exactly eight TRAIN reference-index0 forwards and one causal control, no training."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from routeset.common import sha256, write_json
from routeset.vlm_route_serialization import serialize_paths, parse_paths
from routeset.vlm_sft_data import TRAINING_PROTOCOL, prepare_teacher_forcing, read_observation, resolve
from routeset.vlm_sft_train_probe import PARENTS, train_probe_inputs
from routeset.vlm_teacher_audit import (PROTOCOL, BodyBudget, token_layout, category_statistics,
    scalar_readouts, causal_mutation, chunk_token_statistics, compare_prefix_logits)

CHECKPOINT_SHA = '675595f0f06eb523edfce1d783f61cf5726446b2a376e2e6cdf12f62c69be33f'
GENERATION_SUMMARY_SHA = '530c7621ff507669bef53252ccab31f8cf29d300347702c18722494869175e94'
CAUSAL_ATOL, CAUSAL_RTOL = 1e-3, 1e-4


def selected_references(config, index, samples):
    """Metadata join precedes raw reads; only selected eight index0 routes opened."""
    from routeset.observed_route_head import resample_event_segments
    manifest = Path(config['supervision']).resolve()
    if index.get(str(manifest)) != sha256(manifest):
        raise ValueError('Original SFT supervision metadata changed')
    wanted = {s['id'] for s in samples}
    selected = {}
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if row['id'] in wanted:
            if row['id'] in selected:
                raise ValueError('Duplicate selected reference identity')
            selected[row['id']] = row
    if set(selected) != wanted:
        raise ValueError('Missing predeclared TRAIN references')
    prepared, hashes = [], {str(manifest): sha256(manifest)}
    # Validate all eight identities/paths/hashes before any raw reference read.
    for sample in samples:
        label = selected[sample['id']]
        if label['split'] != 'TRAIN' or label['parent_id'] != sample['parent_id'] or not label['routes']:
            raise ValueError('TRAIN positive reference-index0 join changed')
        current = resolve(manifest.parent, label['observation']).resolve()
        route = resolve(manifest.parent, label['routes'][0]).resolve()
        if str(current) != sample['observation_path'] or sample['parent_id'] not in route.parts:
            raise ValueError('Reference or current input outside fixed parent')
        hashes[str(route)] = sha256(route)
        if index.get(str(route)) != hashes[str(route)]:
            raise ValueError('Original positive reference-index0 bytes changed')
        prepared.append((sample, route))
    results = []
    for sample, route in prepared:
        with np.load(route, allow_pickle=False) as archive:
            raw_xyz = archive['gripper_pose'][:, :3].copy()
            xyz, opened = resample_event_segments(archive['gripper_pose'], archive['gripper_open'], 24)
        observed = read_observation(sample)
        start_error = float(np.linalg.norm(xyz[0]-observed['current'][:3]))
        if start_error > .005 or not np.isfinite(raw_xyz).all():
            raise ValueError('Reference start/finite guard differs from training')
        xyz, opened = np.asarray(xyz, dtype=np.float32), np.asarray(opened, dtype=np.float32)
        answer = serialize_paths(xyz[None], opened[None])
        parsed, _, receipt = parse_paths(answer, 1, 24)
        per_axis_error = np.abs(parsed[0].astype(np.float64)-xyz.astype(np.float64)).max(axis=0)
        if receipt['format_valid_candidates'] != 1 or per_axis_error.max() > .0005001:
            raise ValueError('Integer-mm round-trip failed without repair')
        values = np.asarray(json.loads(answer))[0]
        interface = dict(reference_index=0, reference_path=str(route), reference_sha256=hashes[str(route)],
            raw_coordinate_unit='meter', serialized_coordinate_unit='integer millimeter',
            raw_coordinate_min_m=raw_xyz.min(axis=0).tolist(), raw_coordinate_max_m=raw_xyz.max(axis=0).tolist(),
            resampled_xyz_m=xyz.tolist(), serialized_xyz_mm=values[:, :3].tolist(),
            serialized_events=values[:, 3].tolist(),
            coordinate_histogram_mm_edges=[-10000,-1000,-500,-100,0,100,500,1000,10000],
            coordinate_histogram_mm=[np.histogram(values[:, axis], bins=[-10000,-1000,-500,-100,0,100,500,1000,10000])[0].tolist() for axis in range(3)],
            roundtrip_max_absolute_error_m_by_axis=per_axis_error.tolist(), current_to_reference_start_m=start_error)
        results.append(dict(sample=sample, observed=observed, xyz=xyz, opened=opened, answer=answer, interface=interface))
    return results, hashes


def frozen_free_endpoints(path):
    """Compare to already frozen eight free generations; no model calls or repair."""
    if sha256(path/'summary.json') != GENERATION_SUMMARY_SHA:
        raise ValueError('Exactly the completed original TRAIN8 generation required')
    summary = json.loads((path/'summary.json').read_text())
    if summary['status'] != 'completed' or summary['checkpoint_sha256'] != CHECKPOINT_SHA:
        raise ValueError('Same completed best-checkpoint free generation required')
    for name, receipt in summary['artifacts'].items():
        target = (path/name).resolve()
        if path.resolve() not in target.parents or sha256(target) != receipt['sha256']:
            raise ValueError('Frozen free-generation artifact changed')
    with np.load(path/'predictions.npz', allow_pickle=False) as source:
        if source['parent_ids'].tolist() != PARENTS or source['scene_ids'].tolist() != [p+'_target0' for p in PARENTS]:
            raise ValueError('Exact fixed TRAIN generation identities required')
        return source['paths'][:, 0, -1].copy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=Path('configs/vlm_depth_interface_train8_v1.json'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    result = dict(protocol=PROTOCOL, status='preflight', code_commit=os.environ.get('CODE_COMMIT'),
        split='TRAIN', reference_index=0, maximum_model_forwards=9, actual_model_forwards=0,
        backward_passes=0, optimizer_steps=0, autoregressive_requests=0, maximum_body_seconds=60.,
        deadline_policy='Check at operation boundaries; CUDA operations already launched cannot be interrupted',
        raw_reference_limit=8, dev_or_locked_raw_inputs=0, records=[], causal_control=None,
        scope='Teacher-forced conditional token/scalar readout; not a generated route or task success measure')
    torch = model = budget = None
    try:
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
            raise ValueError('Authorized physical GPU1 only')
        import torch
        import transformers
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from routeset.observed_route_head import QWEN_REVISION
        from routeset.vlm_sft_loss import causal_chunked_loss
        from scripts.train_observed_lora import install_lora, load_adapters
        if transformers.__version__ != '4.57.1' or not torch.__version__.startswith('2.4.1'):
            raise ValueError('Pinned actual Qwen runtime required')
        torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.35)
        torch.cuda.reset_peak_memory_stats()
        checkpoint = args.checkpoint.resolve(); run = checkpoint.parent
        training = json.loads((run/'summary.json').read_text())
        if sha256(checkpoint) != CHECKPOINT_SHA or training['artifacts_sha256'].get('best.pt') != CHECKPOINT_SHA or training['status'] != 'completed':
            raise ValueError('Exact completed SFT best checkpoint required')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False); config = saved['config']
        if saved['protocol'] != TRAINING_PROTOCOL or config['model_revision'] != QWEN_REVISION or config['horizon'] != 24 or config['chunk_size'] != 64:
            raise ValueError('Pinned actual training protocol required')
        if config != json.loads((run/'config.json').read_text()):
            raise ValueError('Actual training configuration changed')
        index = json.loads((run/'data_source_hashes.json').read_text())
        if hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest() != config['data_fingerprint']:
            raise ValueError('Original SFT data fingerprint changed')
        root = Path(__file__).resolve().parents[1]
        for name in ('routeset/vlm_sft_data.py','routeset/vlm_route_serialization.py','routeset/vlm_sft_loss.py',
                     'routeset/observed_route_head.py','scripts/train_observed_lora.py'):
            if sha256(root/name) != config['source_sha256'][name]:
                raise ValueError('Original actual training helper changed: '+name)
        samples, input_hashes = train_probe_inputs(config, index, json.loads(args.plan.read_text()))
        references, reference_hashes = selected_references(config, index, samples)
        free_endpoints = frozen_free_endpoints(args.generation)
        source_names = ('scripts/audit_vlm_sft_teacher_forcing.py','routeset/vlm_teacher_audit.py',
            'routeset/vlm_sft_train_probe.py','routeset/vlm_sft_data.py','routeset/vlm_route_serialization.py',
            'routeset/vlm_sft_loss.py','routeset/observed_route_head.py','scripts/train_observed_lora.py')
        result.update(checkpoint_sha256=CHECKPOINT_SHA, checkpoint_step=training['best_step'],
            training_summary_sha256=sha256(run/'summary.json'), config=config, input_sha256=input_hashes,
            reference_sha256=reference_hashes, sampling_plan_sha256=sha256(args.plan),
            free_generation_summary_sha256=GENERATION_SUMMARY_SHA,
            free_generation_predictions_sha256=sha256(args.generation/'predictions.npz'),
            source_sha256={name:sha256(root/name) for name in source_names},
            causal_tolerance=dict(atol=CAUSAL_ATOL, rtol=CAUSAL_RTOL),
            dtype_explanation='Base/head BF16 with FP32 LoRA as trained; logits are cast to FP32 only after head evaluation. '
                'Declared tolerance is small but nonzero for repeated CUDA arithmetic; actual exact equality is also reported. '
                'It does not exempt BF16-sized differences or prove all possible inputs causal.')
        write_json(args.output/'interface.json', [dict(scene_id=r['sample']['id'], **r['interface']) for r in references])
        model_path = Path(config['model'])
        if sha256(model_path/'provenance.json') != config['model_provenance_sha256']:
            raise ValueError('Pinned model provenance changed')
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
            min_pixels=config['min_pixels'], max_pixels=config['max_pixels'])
        # Processor mask/byte alignment is audited before any model is loaded.
        prepared = []
        for item in references:
            full, labels, receipt = prepare_teacher_forcing(processor, item['observed'], item['xyz'][None], item['opened'][None], 1, 24)
            tokenized = processor.tokenizer(item['answer'], add_special_tokens=False, return_offsets_mapping=True)
            layout = token_layout(item['answer'], tokenized['input_ids'], tokenized['offset_mapping'],
                full['input_ids'][0].numpy(), labels[0], receipt['prompt_tokens'])
            pieces = [processor.tokenizer.decode([token], clean_up_tokenization_spaces=False) for token in tokenized['input_ids']]
            if any(piece != item['answer'][a:b] for piece,(a,b) in zip(pieces, layout['answer_offsets'])):
                raise ValueError('Actual tokenizer decoded byte fragments differ from offsets')
            prepared.append((full, labels, receipt, layout))
        digit_ids = []
        for digit in '0123456789':
            ids = processor.tokenizer(digit, add_special_tokens=False)['input_ids']
            if len(ids) != 1 or processor.tokenizer.decode(ids) != digit:
                raise ValueError('One-token deterministic digit replacement required')
            digit_ids.append(ids[0])
        mutation = causal_mutation(prepared[0][0]['input_ids'][0].tolist(), prepared[0][3], digit_ids)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':'cuda'})
        install_lora(model, rank=config['lora_rank'], alpha=config['lora_alpha']); load_adapters(model, saved['adapters'])
        model.requires_grad_(False); model.eval(); del saved
        result['actual_lm_head_dtype'] = str(model.lm_head.weight.dtype)
        result['actual_parameter_dtypes'] = sorted({str(p.dtype) for p in model.parameters()})
        result['actual_trainable_parameter_count'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        torch.cuda.synchronize(); result['startup_seconds'] = time.perf_counter()-started
        budget = BodyBudget(60.)
        first_hidden = first_inputs = None
        with torch.inference_mode():
            for i, (item, (full, labels, receipt, layout)) in enumerate(zip(references, prepared)):
                budget.check()
                inputs = {key:value.to('cuda') for key,value in full.items()}
                target = torch.as_tensor(labels, dtype=torch.long, device='cuda')
                result['actual_model_forwards'] += 1
                hidden = model.model(**inputs, use_cache=False, return_dict=True).last_hidden_state
                torch.cuda.synchronize(); budget.check()
                stats = chunk_token_statistics(hidden, target, model.lm_head, 64, budget.check)
                budget.check()
                original_loss = float(causal_chunked_loss(hidden, target, model.lm_head, 64))
                mean_loss = float(np.mean(stats['nll']))
                if not np.isclose(mean_loss, original_loss, atol=2e-6, rtol=2e-6):
                    raise ValueError('Per-token loss does not reconcile with unchanged training loss')
                if stats['supervised_positions'] != layout['supervised_positions']:
                    raise ValueError('Actual next-token shift differs from interface receipt')
                top_pieces = [processor.tokenizer.decode([t], clean_up_tokenization_spaces=False) for t in stats['top_ids']]
                readout = scalar_readouts(item['answer'], layout, top_pieces)
                coordinates = [r for r in readout if r['category'] == 'coordinate']
                endpoint = [r['predicted_integer'] for r in coordinates[-3:]]
                endpoint_m = np.asarray(endpoint, dtype=float)/1000 if all(x is not None for x in endpoint) else None
                by_axis = []
                for axis in range(3):
                    valid = [r for r in coordinates if r['field'] == axis and r['predicted_integer'] is not None]
                    errors = [r['signed_integer_error']/1000 for r in valid]
                    by_axis.append(dict(axis='xyz'[axis], parsed_scalars=len(valid), total_scalars=24,
                        signed_error_m_mean=float(np.mean(errors)) if errors else None,
                        absolute_error_m_mean=float(np.mean(np.abs(errors))) if errors else None))
                record = dict(scene_id=item['sample']['id'], parent_id=item['sample']['parent_id'], split='TRAIN',
                    reference_index=0, answer=item['answer'], prompt_receipt=receipt,
                    full_input_ids=full['input_ids'][0].tolist(), labels=labels[0].tolist(),
                    layout=layout, token_statistics=stats, conditional_top1_token_fragments=top_pieces,
                    categories=category_statistics(layout['categories'], stats['nll'], stats['top_ids'], stats['target_ids']),
                    original_causal_chunked_loss=original_loss, per_token_nll_mean=mean_loss,
                    loss_absolute_difference=abs(original_loss-mean_loss),
                    scalars=readout, coordinate_errors_by_axis=by_axis,
                    conditional_endpoint_m=endpoint_m.tolist() if endpoint_m is not None else None,
                    conditional_endpoint_error_to_reference_m=float(np.linalg.norm(endpoint_m-item['xyz'][-1])) if endpoint_m is not None else None,
                    saved_free_endpoint_m=free_endpoints[i].tolist(),
                    saved_free_endpoint_error_to_same_reference_m=float(np.linalg.norm(free_endpoints[i]-item['xyz'][-1])),
                    caveat='Each top1 token receives true previous answer tokens; this is not a free-generation route or semantic success',
                    body_elapsed_seconds=budget.elapsed())
                result['records'].append(record)
                write_json(args.output/(item['sample']['id']+'.json'), record)
                if i == 0:
                    first_hidden, first_inputs = hidden, inputs
                del hidden, inputs, target
            budget.check()
            changed = dict(first_inputs); changed['input_ids'] = first_inputs['input_ids'].clone()
            changed['input_ids'][0, mutation['position']] = mutation['new_token_id']
            result['actual_model_forwards'] += 1
            second = model.model(**changed, use_cache=False, return_dict=True).last_hidden_state
            torch.cuda.synchronize(); budget.check()
            control = compare_prefix_logits(first_hidden, second, mutation['position'], model.lm_head,
                64, CAUSAL_ATOL, CAUSAL_RTOL, budget.check)
            result['causal_control'] = dict(**mutation, **control)
            result['status'] = 'completed' if control['passed'] else 'failed_causal_control'
    except Exception as error:
        result['status'] = 'budget_exhausted' if isinstance(error, TimeoutError) else 'failed'
        result['failure'] = dict(type=type(error).__name__, message=str(error))
    finally:
        if torch is not None and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
            except Exception as error:
                result['final_synchronize_error'] = str(error)
        result['elapsed_seconds'] = time.perf_counter()-started
        result['gpu_hours_reserved'] = result['elapsed_seconds']/3600
        result['body_seconds'] = budget.elapsed() if budget is not None else None
        result['body_budget_overshoot_seconds'] = max(0., result['body_seconds']-60.) if budget is not None else 0.
        result['completed_reference_forwards_with_statistics'] = len(result['records'])
        if result['records']:
            categories, nll, top_ids, target_ids = [], [], [], []
            for record in result['records']:
                categories.extend(record['layout']['categories'])
                nll.extend(record['token_statistics']['nll'])
                top_ids.extend(record['token_statistics']['top_ids'])
                target_ids.extend(record['token_statistics']['target_ids'])
            result['aggregate_token_categories'] = category_statistics(categories, nll, top_ids, target_ids)
            result['aggregate_supervised_tokens'] = len(nll)
            result['aggregate_token_nll'] = float(np.mean(nll))
        result['artifacts'] = {p.name:dict(sha256=sha256(p), bytes=p.stat().st_size)
            for p in args.output.iterdir() if p.is_file()}
        write_json(args.output/'summary.json', result)
        print(json.dumps({key:result.get(key) for key in ('status','actual_model_forwards','body_seconds','elapsed_seconds','failure','causal_control')}), flush=True)
    if result['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
