"""Pinned-Qwen observation-only K1/K4 route SFT; adapters-only checkpoints.

DEV teacher-forced token NLL selects checkpoints. It is not autoregressive
route quality. No frozen feature cache, goal coordinate, box, or mode label
enters this trainer's model inputs. Run server jobs from immutable exports.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from routeset.common import seed_all, sha256, write_json
from routeset.observed_route_head import QWEN_REVISION
from routeset.train_v2 import atomic_checkpoint, restore_rng, rng_state, synchronized_time
from routeset.vlm_sft_data import (TRAINING_PROTOCOL, alternating_k, fixed_development_plan,
    load_sft_data, parent_language_groups, prepare_teacher_forcing, read_observation,
    reference_indices, sample_parent_language, validate_resume_config,
    exclusive_training_output, validate_reserved_observation_manifest)
from routeset.vlm_sft_loss import causal_chunked_loss
from routeset.vlm_sft_continuation import (prepare_continuation_files, verify_continuation_source,
                                         continuation_accounting, validate_continuation_config)
from scripts.train_observed_lora import (adapter_state, adapters, audit_gradients, begin_audit,
    finish_audit, install_lora, load_adapters, tensor_hash)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def source_receipt():
    root = Path(__file__).resolve().parents[1]
    files = ['scripts/train_vlm_route_sft.py', 'routeset/vlm_sft_data.py', 'routeset/vlm_sft_loss.py',
             'routeset/vlm_route_serialization.py', 'scripts/train_observed_lora.py',
             'routeset/observed_route_head.py', 'routeset/train_v2.py', 'routeset/common.py',
             'routeset/vlm_sft_continuation.py']
    return {name: sha256(root / name) for name in files}


def item_loss(model, processor, sample, k, indices, horizon, chunk_size, device):
    observed = read_observation(sample)
    selected = np.asarray(indices, dtype=np.int64)
    full, labels, receipt = prepare_teacher_forcing(processor, observed, sample['paths'][selected],
                                                   sample['events'][selected], k, horizon)
    inputs = {key: value.to(device) for key, value in full.items()}
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    output = model.model(**inputs, use_cache=False, return_dict=True)
    loss = causal_chunked_loss(output.last_hidden_state, labels, model.lm_head, chunk_size)
    return loss, receipt


def _event_totals(events):
    totals = {split: dict(requests=0, candidate_slots=0, prompt_tokens=0, supervised_tokens=0,
                         sequence_tokens=0, completed_optimizer_steps=0) for split in ('TRAIN', 'DEV_MODEL')}
    for event in events:
        if event['kind'] not in ('train_request', 'dev_request'):
            continue
        split = 'TRAIN' if event['kind'] == 'train_request' else 'DEV_MODEL'
        row = totals[split]
        row['requests'] += 1; row['candidate_slots'] += event['k']
        for key in ('prompt_tokens', 'supervised_tokens', 'sequence_tokens'):
            row[key] += event[key]
        row['completed_optimizer_steps'] += event['kind'] == 'train_request'
    return totals


def _run_training_locked(model, processor, data, config, output, device='cuda', resume=False,
                 stop_after=None, startup_seconds=0., loss_function=item_loss, continue_from=None):
    """Actual shared training loop, also used by the tiny CPU resume test.

    A successful request is journaled before checkpointing. On resume, journal
    events beyond the saved optimizer step remain charged as replay overhead.
    Checkpoint trajectory state/history is restored from last.pt exactly.
    """
    output = Path(output)
    if resume and continue_from is not None:
        raise ValueError('Strict resume and explicit continuation are mutually exclusive')
    if resume and not (output / 'last.pt').is_file():
        raise FileNotFoundError('SFT resume requires original last.pt')
    if stop_after is not None and not 0 < stop_after <= config['steps']:
        raise ValueError('stop_after must be within the unchanged planned training steps')
    samples = data['samples']
    groups = parent_language_groups(samples)
    plan = fixed_development_plan(samples, config['dev_plan_seed'])
    if digest_json(plan) != config['dev_plan_sha256'] or data['fingerprint'] != config['data_fingerprint']:
        raise ValueError('Training data or fixed DEV plan differs from configuration')
    parameters = adapters(model)
    if not parameters or any(p.requires_grad for name, p in model.named_parameters() if name not in parameters):
        raise ValueError('Only genuine LoRA adapter parameters may train')
    if any(not p.requires_grad for p in parameters.values()):
        raise ValueError('Every adapter parameter must train')
    optimizer = torch.optim.AdamW(parameters.values(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _: 1.)
    seed_all(config['seed'])
    targets_rng = np.random.default_rng(config['seed'] + 100001)
    sampler = np.random.default_rng(config['seed'] + 100000)
    step, best_step, best_nll = 0, None, float('inf')
    history, events, gradient_audit = [], [], begin_audit(model)
    cumulative_before, checkpoint_event_count, segments = 0., 0, []
    lineage, best_is_inherited, added_gradient_audit = None, False, None
    restoring = resume or continue_from is not None
    base_hashes = {name: tensor_hash(value) for name, value in model.named_parameters() if '.base.weight' in name}
    if restoring:
        saved = torch.load(Path(continue_from) if continue_from is not None else output/'last.pt', map_location='cpu', weights_only=False)
        if continue_from is not None:
            lineage = prepare_continuation_files(continue_from, output, saved, config)
            best_is_inherited = True
        else:
            validate_resume_config(config, saved['config'])
            lineage = saved.get('continuation')
            best_is_inherited = saved.get('best_is_inherited', False)
        if saved['step'] >= config['steps']:
            raise ValueError('SFT run already reached its planned limit; do not restart completed training')
        if stop_after is not None and stop_after <= saved['step']:
            raise ValueError('stop_after must exceed the restored optimizer step')
        load_adapters(model, saved['adapters'])
        optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        restore_rng(saved['rng'], targets_rng); sampler.bit_generator.state = saved['sampler_state']
        step, best_step, best_nll = saved['step'], saved['best_step'], saved['best_nll']
        history, gradient_audit = saved['history'], saved['gradient_audit']
        cumulative_before, checkpoint_event_count = saved['elapsed_seconds'], saved['journal_event_count']
        if continue_from is not None:
            cumulative_before = lineage['inherited_elapsed_seconds']
        segments = saved['segments']
        if base_hashes != saved['frozen_lora_projection_sha256']:
            raise ValueError('Frozen base projections changed since checkpoint')
        raw_lines = (output / 'requests.jsonl').read_text(encoding='utf-8').splitlines()
        for index, line in enumerate(raw_lines):
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                raise ValueError('Incomplete request journal line; preserve/repair journal explicitly before resume: %d' % index)
        if len(events) < checkpoint_event_count:
            raise ValueError('SFT request journal is shorter than committed checkpoint')
        if digest_json(events[:checkpoint_event_count]) != saved['journal_prefix_sha256']:
            raise ValueError('Committed request journal changed')
        if not (output / 'best.pt').is_file() or sha256(output / 'best.pt') != saved['best_checkpoint_sha256']:
            raise ValueError('Selected best checkpoint missing or changed')
        if lineage is not None:
            added_gradient_audit = begin_audit(model) if continue_from is not None else saved['added_gradient_audit']
    else:
        write_json(output / 'config.json', config)
        write_json(output / 'data_source_hashes.json', data['source_sha256'])
        write_json(output / 'dataset_accounting.json', {key: data[key] for key in ('parents_by_split', 'unreferenced')})
        write_json(output / 'dev_plan.json', plan)
        torch.save(adapter_state(model), output / 'initial_adapters.pt')
    # Include already logged but uncheckpointed work in actual exposure/cost,
    # while restoring the original optimizer/RNG trajectory, never skipping it.
    replay_events = events[checkpoint_event_count:] if restoring else []
    replay_wall = sum(event['wall_seconds'] for event in replay_events)
    cumulative_before += replay_wall
    segment = dict(number=len(segments), restored_step=step, code_commit=os.environ.get('CODE_COMMIT'),
                   started_utc=datetime.now(timezone.utc).isoformat(), startup_seconds=startup_seconds,
                   extra_logged_replay_events=len(replay_events), extra_logged_replay_wall_seconds=replay_wall)
    segments = segments + [segment]
    model.eval()  # Deterministic LoRA forward with autograd; frozen backbone has no training dropout.
    segment_start = synchronized_time(device)
    def elapsed():
        return cumulative_before + startup_seconds + synchronized_time(device) - segment_start
    def log_request(kind, idx, k, chosen, metadata, loss, wall):
        event = dict(kind=kind, event_index=len(events), segment=segment['number'], step=step,
                     scene_id=samples[idx]['id'], parent_id=samples[idx]['parent_id'], k=k,
                     reference_indices=list(map(int, chosen)), loss=float(loss.detach()), wall_seconds=wall, **metadata)
        events.append(event)
        with (output / 'requests.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(event, allow_nan=False) + '\n'); handle.flush()
    def evaluate():
        total_nll, tokens = 0., 0
        with torch.no_grad():
            for record in plan:
                tic = synchronized_time(device)
                value, metadata = loss_function(model, processor, samples[record['index']], record['k'],
                    record['reference_indices'], config['horizon'], config['chunk_size'], device)
                if not torch.isfinite(value):
                    raise RuntimeError('Nonfinite DEV token NLL')
                amount = metadata['supervised_tokens']
                total_nll += float(value) * amount; tokens += amount
                log_request('dev_request', record['index'], record['k'], record['reference_indices'],
                            {key:val for key,val in metadata.items() if key != 'k'}, value,
                            synchronized_time(device)-tic)
        return total_nll / tokens
    def save_checkpoint(selected=False):
        payload = dict(protocol=TRAINING_PROTOCOL, adapters=adapter_state(model), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), rng=rng_state(targets_rng), sampler_state=sampler.bit_generator.state,
            step=step, config=config, best_step=best_step, best_nll=best_nll, history=history,
            gradient_audit=gradient_audit, exposure=_event_totals(events), elapsed_seconds=elapsed(),
            segments=segments, journal_event_count=len(events), journal_prefix_sha256=digest_json(events),
            frozen_lora_projection_sha256=base_hashes)
        if lineage is not None:
            payload.update(continuation=lineage, best_is_inherited=best_is_inherited,
                           added_gradient_audit=added_gradient_audit,
                           continuation_accounting=continuation_accounting(lineage,payload['exposure'],payload['elapsed_seconds']))
        if selected:
            atomic_checkpoint(output / 'best.pt', payload)
        payload['best_checkpoint_sha256'] = sha256(output / 'best.pt')
        atomic_checkpoint(output / 'last.pt', payload)
        write_json(output / 'history.json', history)
        write_json(output / 'progress.json', dict(step=step, planned_steps=config['steps'], best_step=best_step,
            best_dev_token_nll=best_nll, exposure=payload['exposure'], elapsed_seconds=payload['elapsed_seconds'],
            journal_event_count=len(events), source_commit=os.environ.get('CODE_COMMIT')))
    if not restoring:
        best_nll = evaluate(); best_step = 0
        history.append(dict(step=0, dev_token_nll=best_nll, selected=True))
        save_checkpoint(selected=True)
        print(json.dumps(dict(step=0, dev_token_nll=best_nll, selected=True)), flush=True)
    elif continue_from is not None:
        # A real recovery point in the new output exists before its first step;
        # do not spend another DEV evaluation or advance either sampler/RNG.
        save_checkpoint(selected=False)
    end = config['steps'] if stop_after is None else stop_after
    while step < end:
        next_step = step + 1
        k = alternating_k(next_step)
        idx = sample_parent_language(groups, sampler)
        chosen = reference_indices(len(samples[idx]['paths']), k, targets_rng)
        tic = synchronized_time(device)
        optimizer.zero_grad(set_to_none=True)
        loss, metadata = loss_function(model, processor, samples[idx], k, chosen, config['horizon'], config['chunk_size'], device)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite TRAIN causal loss')
        loss.backward(); audit_gradients(model, gradient_audit)
        if added_gradient_audit is not None:
            audit_gradients(model, added_gradient_audit)
        torch.nn.utils.clip_grad_norm_(list(parameters.values()), config['gradient_clip'])
        optimizer.step(); scheduler.step(); step = next_step
        wall = synchronized_time(device)-tic
        log_request('train_request', idx, k, chosen, {key:val for key,val in metadata.items() if key != 'k'}, loss.detach(), wall)
        row = dict(step=step, k=k, train_token_nll=float(loss.detach()))
        del loss
        selected = False
        if step % config['eval_every'] == 0 or step == config['steps']:
            nll = evaluate(); selected = nll < best_nll
            row.update(dev_token_nll=nll, selected=selected)
            if selected:
                best_nll, best_step = nll, step
                best_is_inherited = False
        history.append(row)
        if selected or step % config['checkpoint_every'] == 0 or step == end:
            save_checkpoint(selected)
        if step % config['log_every'] == 0 or 'dev_token_nll' in row or step == end:
            print(json.dumps(dict(row, train_candidate_slots=_event_totals(events)['TRAIN']['candidate_slots'])), flush=True)
    if base_hashes != {name: tensor_hash(value) for name, value in model.named_parameters() if '.base.weight' in name}:
        raise RuntimeError('Frozen LoRA projection weights changed')
    update_audit = finish_audit(model, gradient_audit, require_update=step >= 2)
    summary = dict(protocol=TRAINING_PROTOCOL, status='completed' if step == config['steps'] else 'stopped_early',
        step=step, planned_steps=config['steps'], best_step=best_step, best_dev_token_nll=best_nll,
        exposure=_event_totals(events), elapsed_seconds=elapsed(), segments=segments, gradient_audit=update_audit,
        frozen_lora_projection_hashes_unchanged=True, adapter_parameters=sum(p.numel() for p in parameters.values()),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0,
        code_commit=os.environ.get('CODE_COMMIT'), data_fingerprint=data['fingerprint'],
        artifacts_sha256={p.name:sha256(p) for p in output.glob('*.pt')},
        scope='Teacher-forced SFT and DEV token NLL only; autoregressive route quality requires separate evaluation. '
              'Journal includes successful replay requests beyond restored checkpoints; failed/in-flight requests remain job-log limitations.')
    summary['gpu_hours_reserved'] = summary['elapsed_seconds']/3600 if str(device).startswith('cuda') else 0.
    summary['gpu_hours_scope'] = 'Reserved wall time including startup and teacher-forced DEV; not autoregressive latency or utilization-normalized GPU hours.'
    if lineage is not None:
        verify_continuation_source(lineage)
        summary['continuation'] = lineage
        summary['continuation_accounting'] = continuation_accounting(lineage,summary['exposure'],summary['elapsed_seconds'])
        summary['added_gpu_hours_reserved'] = summary['continuation_accounting']['added_elapsed_seconds']/3600 if str(device).startswith('cuda') else 0.
        summary['inherited_gpu_hours_reserved'] = lineage['inherited_elapsed_seconds']/3600 if str(device).startswith('cuda') else 0.
        summary['original_source_files_unchanged'] = True
        summary['best_is_inherited'] = best_is_inherited
        summary['added_gradient_audit'] = finish_audit(model,added_gradient_audit,require_update=step > lineage['restored_step'])
        summary['selected_checkpoint_origin'] = str(Path(lineage['source_run'])/'best.pt') if best_is_inherited else str(output/'best.pt')
    write_json(output / 'summary.json', summary)
    return summary


def run_training(model, processor, data, config, output, device='cuda', resume=False,
                 stop_after=None, startup_seconds=0., loss_function=item_loss, continue_from=None):
    with exclusive_training_output(output, resume):
        return _run_training_locked(model, processor, data, config, output, device, resume,
                                    stop_after, startup_seconds, loss_function, continue_from)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('observations', 'supervision', 'model', 'output'):
        parser.add_argument('--'+key, type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dev-plan-seed', type=int, default=200000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--checkpoint-every', type=int, default=25)
    parser.add_argument('--log-every', type=int, default=25)
    parser.add_argument('--chunk-size', type=int, default=64)
    restore = parser.add_mutually_exclusive_group()
    restore.add_argument('--resume', action='store_true')
    restore.add_argument('--continue-from', type=Path,
        help='Explicitly extend completed original last.pt into a fresh output; all objective/data settings locked')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    if args.batch_size != 1 or min(args.steps,args.eval_every,args.checkpoint_every,args.log_every,args.chunk_size) < 1:
        raise ValueError('This protocol requires batch1 and positive step/checkpoint/chunk budgets')
    if args.lr <= 0 or args.weight_decay < 0:
        raise ValueError('Invalid optimizer configuration')
    if args.output.exists() and not args.resume:
        raise FileExistsError('Preserve previous SFT output')
    if args.resume and not (args.output/'last.pt').is_file():
        raise FileNotFoundError('Original last.pt required to resume')
    if args.continue_from is not None and (args.continue_from.name != 'last.pt' or not args.continue_from.is_file()):
        raise ValueError('Explicit continuation requires an existing original last.pt')
    validate_reserved_observation_manifest(args.observations)
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '1':
        raise ValueError('Only authorized physical GPU1 is allowed')
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    if transformers.__version__ != '4.57.1' or not torch.__version__.startswith('2.4.1'):
        raise ValueError('Pinned .venv-qwen runtime required')
    torch.set_num_threads(1); torch.cuda.set_per_process_memory_fraction(.35)
    torch.cuda.reset_peak_memory_stats(); seed_all(args.seed)
    started = time.perf_counter()
    model_path = args.model.resolve()
    provenance = json.loads((model_path/'provenance.json').read_text())
    if provenance.get('revision') != QWEN_REVISION or not provenance.get('all_hashes_verified'):
        raise ValueError('Verified pinned Qwen model provenance required')
    data = load_sft_data(args.observations, args.supervision, 24)
    config = dict(protocol=TRAINING_PROTOCOL, observations=str(args.observations.resolve()),
        supervision=str(args.supervision.resolve()), model=str(model_path), model_revision=QWEN_REVISION,
        processor_revision=QWEN_REVISION, model_provenance_sha256=sha256(model_path/'provenance.json'),
        steps=args.steps, batch_size=1, seed=args.seed, lr=args.lr, weight_decay=args.weight_decay,
        eval_every=args.eval_every, checkpoint_every=args.checkpoint_every, log_every=args.log_every,
        chunk_size=args.chunk_size, horizon=24, gradient_clip=1., scheduler='constant',
        min_pixels=65536, max_pixels=65536, lora_rank=8, lora_alpha=16., lora_layers='last2_q_v',
        parameter_dtype='bfloat16_base_float32_adapter', attn_implementation='sdpa',
        sampling='uniform_positive_parent_then_uniform_positive_language', k_schedule='odd1_even4',
        dev_plan_seed=args.dev_plan_seed, selection='DEV_token_NLL_total_sum_div_supervised_tokens',
        data_fingerprint=data['fingerprint'], source_sha256=source_receipt(),
        dependencies=dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__))
    config['dev_plan_sha256'] = digest_json(fixed_development_plan(data['samples'], config['dev_plan_seed']))
    if args.resume:
        validate_resume_config(config, json.loads((args.output/'config.json').read_text()))
    if args.continue_from is not None:
        preview = torch.load(args.continue_from, map_location='cpu', weights_only=False)
        validate_continuation_config(config,preview['config'])
        del preview
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, min_pixels=65536, max_pixels=65536)
    model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'':'cuda'})
    install_lora(model, rank=8, alpha=16.)
    summary = run_training(model, processor, data, config, args.output, resume=args.resume,
                           stop_after=args.stop_after, startup_seconds=time.perf_counter()-started,
                           continue_from=args.continue_from)
    print(json.dumps({key:summary[key] for key in ('status','step','best_step','best_dev_token_nll',
        'exposure','elapsed_seconds','peak_cuda_allocated_bytes')}), flush=True)


if __name__ == '__main__':
    main()
