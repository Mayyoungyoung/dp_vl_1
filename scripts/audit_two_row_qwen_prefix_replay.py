"""Fresh-only TRAIN6 serial Qwen prefix replay/LoRA technical gate (no DEV).

No quality experiment or training continuation is launched by this entry point.
The actual 10 full / 10 replay / 4 head / 4 optimizer call budget is immutable.
"""
import argparse
import copy
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
import traceback

from routeset.qwen_prefix_replay import (PROTOCOL, IDS, PARENTS, BUDGET, HEAD_SHA,
    REVISION, validate_policy, validate_rows)

PROJECT = Path(__file__).resolve().parents[1]
EXPORT_SHA = '04ba27290e382787d5f1764fc2645d4219b09457ea972d15f7d1f3baeac74fdc'
OFFICIAL_SHA = {
    'models/qwen3_vl/modeling_qwen3_vl.py': 'dd63ed3b124232735b3dca1bfa28f9d6b0d3f7182afcb75dde8f3e724b2b22da',
    'models/qwen3_vl/configuration_qwen3_vl.py': '177fd0a4dc1b08307c08ca72cf26b8d7dc028ab6ab5975bf650fc14ab8132a83',
    'integrations/sdpa_attention.py': 'dc5abe49a98dec3b9026739dfbf2e9a8f3e5272b2916b3c2d404727ac931a013',
    'masking_utils.py': '7a963feed8173b8265298dd24350c166222ce6b4719442fce48ab414d0478a6a'}
DEPENDENCIES = ('routeset/qwen_prefix_replay.py', 'scripts/train_observed_lora.py',
    'scripts/train_observed_geometry.py', 'routeset/observed_geometry.py',
    'routeset/observed_route_head.py', 'routeset/train_v2.py', 'routeset/common.py', 'routeset/models.py',
    'scripts/export_two_row_composite_observations.py', 'scripts/observation_cache_qwen.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temp.replace(path)


def exact_prefix(path):
    # Stop after six lines. Later TRAIN/DEV instructions and labels are not parsed.
    with Path(path).open(encoding='utf-8-sig') as stream:
        rows = [json.loads(next(stream)) for _ in IDS]
    if tuple(row.get('id') for row in rows) != IDS:
        raise ValueError('Original fixed TRAIN6 prefix changed')
    return rows


def checked_train_path(value, parent, manifest, hashes):
    path = Path(value)
    expected_root = Path(manifest['sources']['old']['source_dataset']) / 'parents' / 'TRAIN' / parent
    root = expected_root.resolve(strict=True)
    if expected_root.absolute() != root:
        raise ValueError('TRAIN parent root is a symlink')
    resolved = path.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError('Raw access outside the fixed TRAIN parent') from exc
    if path.absolute() != resolved:
        raise ValueError('TRAIN raw symlink rejected')
    value = sha(path)
    if manifest['source_files_sha256'].get(str(path)) != value:
        raise ValueError('TRAIN source differs from the sealed export: ' + str(path))
    hashes[str(path)] = value
    return path


def sealed_training_cache(data_root, rows, receipt_sha256, expected_config):
    """Open only six NPZs; bind each to the sealed old225 reuse receipt.

    Historical caches saved token counts, not token ID sequences or prompt text.
    Their row-derived filenames bind the original instruction/image identity.
    """
    import numpy as np
    cache = Path(data_root) / 'qwen_cache'
    receipt_path = cache / 'composite_cache_receipt.json'
    if sha(receipt_path) != receipt_sha256:
        raise ValueError('The head training cache receipt changed')
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('protocol') != 'two_row_composite108_old225_cache_reuse_v1'
            or receipt.get('source_export_manifest_sha256') != EXPORT_SHA
            or receipt.get('old_reused_count') != 225):
        raise ValueError('Historical cache reuse provenance mismatch')
    for name in ('cache_config.json', 'samples.jsonl'):
        if sha(cache / name) != receipt['artifact_sha256'][name]:
            raise ValueError('Sealed cache metadata changed: ' + name)
    config = json.loads((cache / 'cache_config.json').read_text())
    if (config != expected_config or config.get('revision') != REVISION
            or config.get('processor') != REVISION or config.get('model_trainable_parameter_count') != 0
            or config.get('manifest_sha256') != sha(Path(data_root) / 'observations.jsonl')):
        raise ValueError('Common-head frozen encoding contract changed')
    records = exact_prefix(cache / 'samples.jsonl')
    prior = receipt['old225_by_id'][:6]
    if tuple(r.get('id') for r in prior) != IDS:
        raise ValueError('Original six cache identity order changed')
    expected_fields = {'mean_hidden','last_hidden','id','parent_id','split','image_sha256','input_tokens'}
    features, audit = {}, []
    for row, record, reused in zip(validate_rows(rows), records, prior):
        filename = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20] + '.npz'
        path = cache / filename
        if (record['file'] != filename or reused['file'] != filename or reused['split'] != 'TRAIN'
                or not reused['npz_bytes_equal'] or not reused['feature_arrays_equal']
                or not (sha(path) == record['sha256'] == reused['sha256'] == receipt['artifact_sha256'][filename])):
            raise ValueError('Selected TRAIN cache bytes/input-row identity changed')
        with np.load(path, allow_pickle=False) as archive:
            if (set(archive.files) != expected_fields or
                    any(str(archive[k].item()) != row[k] for k in ('id','parent_id','split')) or
                    str(archive['image_sha256'].item()) != sha(row['image']) or
                    str(archive['image_sha256'].item()) != record['image_sha256'] or
                    int(archive['input_tokens']) != record['input_tokens'] or
                    any(archive[k].shape != (2048,) or archive[k].dtype != np.float32 or not np.isfinite(archive[k]).all()
                        for k in ('mean_hidden','last_hidden'))):
                raise ValueError('Selected TRAIN cached feature schema/source changed')
            features[row['id']] = dict(feature=np.concatenate([archive['mean_hidden'],archive['last_hidden']])[None],
                                      input_tokens=int(archive['input_tokens']))
        audit.append(dict(id=row['id'], sha256=record['sha256'], input_row_key=filename,
                          input_tokens=record['input_tokens'], image_sha256=record['image_sha256']))
    return features, dict(receipt_sha256=receipt_sha256, config_sha256=sha(cache/'cache_config.json'),
        samples_sha256=sha(cache/'samples.jsonl'), selected=audit, opened_npz_count=6,
        original_prompt_and_token_ids_available=False,
        prompt_provenance='Exact original observation row key + pinned processor/extractor; old NPZ records token count only')


def compare_historical_feature(actual, tokens, historical):
    import numpy as np
    expected = historical['feature']
    if actual.shape != expected.shape or actual.dtype != expected.dtype or tokens != historical['input_tokens']:
        raise ValueError('Historical feature shape/dtype/token count mismatch')
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    return dict(exact=bool(np.array_equal(actual, expected)),
                max_abs=float(np.abs(delta).max()), rms=float(np.sqrt(np.mean(delta**2))),
                input_tokens=tokens, prompt_and_token_sequence_equality_claimed=False)


class BudgetLedger:
    def __init__(self, output):
        self.output = Path(output)
        self.records = []

    def call(self, kind, identity, function):
        key = {'full': 'full_features', 'replay': 'replay_features', 'head': 'head_calls',
               'optimizer_full': 'full_optimizer_steps', 'optimizer_replay': 'replay_optimizer_steps'}[kind]
        if sum(r['kind'] == kind for r in self.records) >= BUDGET[key]:
            raise ValueError('Predeclared call budget exhausted: ' + kind)
        record = dict(kind=kind, identity=identity, state='issued',
                      candidate_path_states=4 if kind == 'head' else 0)
        self.records.append(record)
        write(self.output / 'call_ledger.json', self.records)
        started = time.perf_counter()
        try:
            result = function()
            record['state'] = 'completed'
            return result
        except BaseException as exc:
            record.update(state='failed', exception=repr(exc))
            raise
        finally:
            record['elapsed_seconds'] = time.perf_counter() - started
            write(self.output / 'call_ledger.json', self.records)

    def counts(self):
        def n(kind): return sum(r['kind'] == kind for r in self.records)
        return dict(full_features=n('full'), replay_features=n('replay'), head_calls=n('head'),
            candidate_path_states=4*n('head'), full_optimizer_steps=n('optimizer_full'),
            replay_optimizer_steps=n('optimizer_replay'), optimizer_steps_total=n('optimizer_full')+n('optimizer_replay'))


def load_selected(data_root, manifest, output):
    import numpy as np
    from scripts.train_observed_geometry import read_geometry
    from routeset.observed_route_head import resample_event_segments
    hashes = {}
    for name in ('observations.jsonl', 'supervision.jsonl'):
        path = data_root / name
        if sha(path) != manifest['output_files_sha256'][name]:
            raise ValueError('Sealed export manifest changed: ' + name)
        hashes[str(path)] = sha(path)
    observations = validate_rows(exact_prefix(data_root / 'observations.jsonl'))
    labels = exact_prefix(data_root / 'supervision.jsonl')
    samples = []
    for i, (row, label) in enumerate(zip(observations, labels)):
        if (label.get('parent_id'), label.get('split')) != (row['parent_id'], 'TRAIN'):
            raise ValueError('TRAIN label identity mismatch')
        image = checked_train_path(row['image'], row['parent_id'], manifest, hashes)
        observation = checked_train_path(label['observation'], row['parent_id'], manifest, hashes)
        with np.load(observation, allow_pickle=False) as archive:
            current = np.r_[archive['gripper_pose'].reshape(7), archive['gripper_open'].reshape(1)].astype(np.float32)
        item = dict(row=row, image=image, current=current)
        # Only the two predeclared micro-update examples need route supervision.
        if i in (0, 3):
            item['points'] = read_geometry(image, observation, 2)
            paths, events = [], []
            for source in label['routes']:
                path = checked_train_path(source, row['parent_id'], manifest, hashes)
                with np.load(path, allow_pickle=False) as archive:
                    xyz, opened = resample_event_segments(archive['gripper_pose'], archive['gripper_open'], 24)
                if np.linalg.norm(xyz[0] - current[:3]) > .005:
                    raise ValueError('Recorded route current-state mismatch')
                paths.append(xyz); events.append(opened)
            if not paths:
                raise ValueError('Fixed update example has no known positive; no replacement allowed')
            item.update(paths=np.stack(paths), events=np.stack(events))
        samples.append(item)
    for start in (0, 3):
        if len({str(s['image']) for s in samples[start:start+3]}) != 1 or any(
                not np.array_equal(samples[start]['current'], s['current']) for s in samples[start:start+3]):
            raise ValueError('Same-parent three instructions require identical current observation')
    write(output / 'selected_observations.json', observations)
    write(output / 'selected_raw_source_sha256.json', hashes)
    return samples


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # A failed attempt cannot be retried in this pool.
    started = time.perf_counter()
    ledger = BudgetLedger(out)
    gpu_requested = False
    summary = dict(protocol=PROTOCOL, status='running', planned_budget=BUDGET,
                   dev_inputs_opened=0, reserved_raw_opened=0, batched_replay_verified=False)
    write(out / 'status.json', summary)
    try:
        import numpy as np
        import torch
        import transformers
        from PIL import Image
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from routeset import qwen_prefix_replay as replay
        from routeset.common import seed_all
        from routeset.observed_geometry import ObservedGeometryRouteHead, positive_endpoint_attention_loss
        from routeset.train_v2 import positive_assignment_loss
        from scripts.train_observed_lora import install_lora
        from scripts.export_two_row_composite_observations import _ready
        policy = validate_policy(json.loads(Path(args.config).read_text()))
        write(out / 'policy.json', policy)
        if (os.environ.get('CUDA_VISIBLE_DEVICES') != '1' or not torch.cuda.is_available()
                or torch.__version__.split('+')[0] != '2.4.1' or transformers.__version__ != '4.57.1'
                or not hasattr(os, 'sched_getaffinity') or len(os.sched_getaffinity(0)) != 1):
            raise ValueError('Pinned Torch/transformers, GPU1, and one CPU affinity required')
        torch.set_num_threads(1)
        torch.cuda.set_per_process_memory_fraction(.35, 0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        installed = Path(inspect.getfile(transformers)).parent
        for relative, expected in OFFICIAL_SHA.items():
            if sha(installed / relative) != expected:
                raise ValueError('Official installed source changed: ' + relative)
        model_path, data_root, checkpoint_path = map(Path, (args.model, args.data, args.head))
        if sha(checkpoint_path) != HEAD_SHA:
            raise ValueError('The fixed composite last12000 head checkpoint changed')
        if sha(data_root / 'export_manifest.json') != EXPORT_SHA:
            raise ValueError('Fixed composite export changed')
        manifest = json.loads((data_root / 'export_manifest.json').read_text())
        _, gate = _ready(manifest['sources']['old']['source_dataset'],
                         manifest['sources']['extension']['source_dataset'], manifest['selection'])
        provenance = json.loads((model_path / 'provenance.json').read_text())
        if (provenance.get('model_id') != 'Qwen/Qwen3-VL-2B-Instruct'
                or provenance.get('revision') != REVISION or not provenance.get('all_hashes_verified')):
            raise ValueError('Pinned model provenance required')
        source_hashes = {name: sha(PROJECT / name) for name in DEPENDENCIES}
        source_hashes['scripts/audit_two_row_qwen_prefix_replay.py'] = sha(__file__)
        write(out / 'preflight.json', dict(source_sha256=source_hashes, official_source_sha256=OFFICIAL_SHA,
            policy_sha256=sha(args.config), head_checkpoint_sha256=HEAD_SHA, export_sha256=EXPORT_SHA,
            model_provenance_sha256=sha(model_path/'provenance.json'), mechanical_only_gate=gate,
            code_commit=os.environ.get('CODE_COMMIT'), pid=os.getpid(), cpu_affinity=sorted(os.sched_getaffinity(0))))
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        cfg = checkpoint['config']
        required = dict(steps=12000, feature_dim=4096, horizon=24, candidates=4, width=128, depth=2,
            point_width=64, anchor_mode='straight_through_peak', endpoint_mode='surface_anchor',
            endpoint_residual_bound=.05, refinement_mode='none', pooling='both', grounding_weight=.02,
            grounding_sigma=.025, event_scale=.2, objective='saturation')
        if checkpoint['step'] != 12000 or any(cfg.get(k) != v for k, v in required.items()):
            raise ValueError('Fixed ordinary head architecture/step mismatch')
        samples = load_selected(data_root, manifest, out)
        historical_features, historical_guard = sealed_training_cache(data_root,
            [sample['row'] for sample in samples], cfg['composite_cache_sha256'], cfg['cache_config'])
        write(out / 'historical_cache_guard.json', historical_guard)
        if sha(PROJECT/'scripts/observation_cache_qwen.py') != '9b8121ec8c95e4e95d89e191c0fae3b9251db120c7b621bed18dad7fc820d4bf':
            raise ValueError('Historical prompt/extractor source changed')
        seed_all(policy['seed'])
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa')
        gpu_requested = True
        model = model.to('cuda').eval().requires_grad_(False)
        text = model.model.language_model
        if len(text.layers) != 28 or text.config.hidden_size != 2048 or text.config.attention_dropout != 0:
            raise ValueError('Actual pinned Qwen shape/dropout mismatch')
        processor = AutoProcessor.from_pretrained(model_path, local_files_only=True,
            min_pixels=policy['min_pixels'], max_pixels=policy['max_pixels'])
        write(out / 'model_assets.json', {p.name: sha(p) for p in model_path.iterdir() if p.is_file()})
        (out / 'prefix_cache').mkdir()
        payloads, features, inputs = {}, {}, {}
        prompts, historical_comparisons = [], []
        for sample in samples:
            row = sample['row']; identity = row['id']
            with Image.open(sample['image']) as image:
                rgb = image.convert('RGB')
            messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': row['instruction']}]}]
            prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            values = processor(text=[prompt], images=[rgb], return_tensors='pt').to('cuda')
            if 'labels' in values:
                raise ValueError('Answer tokens cannot enter Qwen conditioning')
            inputs[identity] = values
            def capture():
                result = replay.capture_prefix(model, values)
                torch.cuda.synchronize()
                return result
            payload, feature = ledger.call('full', identity + ':capture', capture)
            comparison = compare_historical_feature(feature.numpy(), int(values['attention_mask'].sum()), historical_features[identity])
            historical_comparisons.append(dict(id=identity, **comparison))
            write(out / 'historical_feature_comparisons.json', historical_comparisons)
            if not comparison['exact']:
                # Retain the actual mismatching feature even though no replay is issued.
                torch.save(feature, out / (identity + '_historical_mismatch.pt'))
                raise ValueError('Initial official feature differs from common-head frozen cache; no retry')
            path = out / 'prefix_cache' / (identity + '.pt')
            torch.save(payload, path)
            # The gate uses the actual serialized/reloaded cache, not only RAM.
            payloads[identity] = torch.load(path, map_location='cpu', weights_only=False)
            features[identity] = feature
            prompts.append(dict(id=identity, message_roles=['user'], prompt=prompt,
                token_ids=values['input_ids'].detach().cpu().tolist(), token_count=int(values['attention_mask'].sum()),
                prefix_sha256=sha(path), prefix_tensor_shapes={k:list(v.shape) for k,v in payload.items() if torch.is_tensor(v)},
                source_image_sha256=sha(sample['image'])))
        write(out / 'input_and_prefix_audit.json', prompts)
        torch.save(features, out / 'official_initial_features.pt')
        seed_all(policy['seed'] + 200000)
        modules = install_lora(model, rank=policy['rank'], alpha=policy['alpha'])
        model.eval()
        adapters = replay.adapter_parameters(model)
        if len(adapters) != 8 or sum(p.numel() for p in adapters.values()) != 114688:
            raise ValueError('Exactly last-two q/v LoRA eight tensors required')
        if {n for n,p in model.named_parameters() if p.requires_grad} != set(adapters):
            raise ValueError('Unexpected trainable base parameter')
        initial_adapters = replay.cpu_copy(adapters)
        frozen_before = replay.parameter_hashes(model, exclude_adapters=True)
        write(out / 'frozen_before_sha256.json', frozen_before)

        def feature_call(kind, identity, stage):
            def function():
                value = (replay.official_feature(model, inputs[identity]) if kind == 'full'
                         else replay.replay_feature(model, payloads[identity]))
                torch.cuda.synchronize()
                return value
            return ledger.call(kind, identity + ':' + stage, function)

        initial_comparisons = []
        for identity in IDS:
            with torch.no_grad(): value = feature_call('replay', identity, 'zero_adapter')
            result = replay.tensor_difference(features[identity], value.cpu())
            initial_comparisons.append(dict(id=identity, **result))
            write(out / 'initial_feature_comparisons.json', initial_comparisons)
            if not result['exact']:
                raise ValueError('Initial official/replay feature mismatch; no retry')
        full_head = ObservedGeometryRouteHead(4096, 24, 4, 128, 2, 64, .05,
            anchor_mode='straight_through_peak', endpoint_mode='surface_anchor').to('cuda')
        full_head.load_state_dict(checkpoint['model'], strict=True)
        full_head.requires_grad_(True)
        replay_head = copy.deepcopy(full_head)
        initial_head_hashes = replay.parameter_hashes(full_head)
        updater = replay.PairedTailUpdater(model, full_head, replay_head, policy['head_lr'], policy['lora_lr'])
        del checkpoint
        step_reports = []
        for step, index in enumerate((0, 3), 1):
            sample = samples[index]; identity = sample['row']['id']
            def loss_function(head, feature, branch):
                model_inputs = dict(features=feature, current=torch.as_tensor(sample['current'][None], device='cuda'))
                model_inputs.update({k:torch.as_tensor(v[None], device='cuda') for k,v in sample['points'].items()})
                def head_forward():
                    value = head(**model_inputs); torch.cuda.synchronize(); return value
                xyz, opened, details = ledger.call('head', identity + ':' + branch + ':step' + str(step), head_forward)
                target = torch.as_tensor(sample['paths'][None], device='cuda')
                events = torch.as_tensor(sample['events'][None], device='cuda')
                mask = np.ones((1, len(sample['paths'])), bool)
                prediction = torch.cat([xyz[:,:,1:], opened[:,:,1:,None]*.2], -1)
                wanted = torch.cat([target[:,:,1:], events[:,:,1:,None]*.2], -1)
                path_loss = positive_assignment_loss(prediction, wanted, mask, 'saturation', np.random.default_rng(0))
                grounding = positive_endpoint_attention_loss(details['attention'], model_inputs['world_xyz'],
                    model_inputs['valid_mask'], target[:,:,-1], torch.as_tensor(mask, device='cuda'), .025)
                return path_loss + .02*grounding, dict(paths=xyz, gripper_open=opened, path_loss=path_loss, grounding_loss=grounding)
            def optimizer_call(branch, function):
                def execute():
                    result = function(); torch.cuda.synchronize(); return result
                return ledger.call('optimizer_' + branch, identity + ':step' + str(step), execute)
            result = updater.step({name:(lambda name=name:feature_call(name, identity, 'gradient_step'+str(step)))
                                   for name in ('full','replay')}, loss_function, optimizer_call)
            torch.save(result, out / ('step%d_full_replay.pt' % step))
            report = dict(step=step, id=identity, exact=result['exact'], differences=result['differences'],
                full_gradient_audit=result['branches']['full']['gradient_audit'],
                replay_gradient_audit=result['branches']['replay']['gradient_audit'],
                loss=float(result['branches']['full']['loss']))
            step_reports.append(report); write(out / 'step_reports.json', step_reports)
            if not result['exact']:
                raise ValueError('Paired loss/gradient/optimizer/update mismatch; no retry')
            with torch.no_grad():
                full = feature_call('full', identity, 'post_update'+str(step))
                cut = feature_call('replay', identity, 'post_update'+str(step))
            post = replay.tensor_difference(full, cut)
            report['post_update_feature'] = post
            report['feature_changed_from_initial'] = not replay.tensor_difference(features[identity], full.cpu())['exact']
            torch.save(dict(full=full.cpu(), replay=cut.cpu()), out / ('step%d_post_features.pt' % step))
            write(out / 'step_reports.json', step_reports)
            torch.save(dict(paired=updater.state_dict(), torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), policy=policy, policy_sha256=sha(args.config),
                sources=source_hashes, budget=ledger.counts()), out / ('step%d_state.pt' % step))
            if not post['exact']:
                raise ValueError('Updated official/replay feature mismatch; no retry')
        changed = {n:not torch.equal(initial_adapters[n], p.detach().cpu()) for n,p in adapters.items()}
        final_gradients = step_reports[-1]['full_gradient_audit']
        adapter_grad_ok = all(final_gradients[n]['present'] and final_gradients[n]['finite'] and
                              final_gradients[n]['nonzero'] for n in adapters)
        head_grad_ok = all(v['present'] and v['finite'] for n,v in final_gradients.items() if n.startswith('head.'))
        frozen_after = replay.parameter_hashes(model, exclude_adapters=True)
        write(out / 'frozen_after_sha256.json', frozen_after)
        final_audit = dict(adapter_modules=modules, adapter_parameters=114688, adapter_changed=changed,
            common_head_historical_features_exact=all(r['exact'] for r in historical_comparisons),
            second_step_adapter_gradient_gate=adapter_grad_ok, all_head_parameters_trainable=all(p.requires_grad for p in full_head.parameters()),
            second_step_head_gradients_present_finite=head_grad_ok,
            head_changed={n:initial_head_hashes[n] != h for n,h in replay.parameter_hashes(full_head).items()},
            frozen_base_unchanged=frozen_before == frozen_after,
            initial_feature_exact=all(r['exact'] for r in initial_comparisons),
            two_branch_steps_exact=all(r['exact'] for r in step_reports),
            post_update_features_exact=all(r['post_update_feature']['exact'] for r in step_reports),
            at_least_one_updated_feature_changed=any(r['feature_changed_from_initial'] for r in step_reports))
        write(out / 'final_audit.json', final_audit)
        if not (all(changed.values()) and adapter_grad_ok and head_grad_ok and frozen_before == frozen_after
                and final_audit['at_least_one_updated_feature_changed'] and ledger.counts() == BUDGET):
            raise ValueError('Actual update/frozen-weight/budget gate failed; do not expand training')
        summary.update(status='completed', gate_passed=True, exit_code=0, final_audit=final_audit)
    except BaseException as exc:
        summary.update(status='failed', gate_passed=False, exit_code=1, exception=repr(exc))
        (out / 'exception.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise
    finally:
        elapsed = time.perf_counter()-started
        summary.update(actual_issued_budget=ledger.counts(), elapsed_seconds=elapsed,
            peak_cuda_memory_allocated_bytes=int(torch.cuda.max_memory_allocated()) if gpu_requested else None,
            peak_cuda_memory_reserved_bytes=int(torch.cuda.max_memory_reserved()) if gpu_requested else None,
            gpu_hours_reserved=elapsed/3600 if gpu_requested else 0., gpu_model_transfer_requested=gpu_requested,
            gpu_cost_scope='Whole job wall time after a GPU model transfer was requested; includes CPU hashing/preflight, not utilization-derived compute time',
            interpretation='Technical serial replay gate only; no method/quality result, no DEV or candidate selection',
            resume_policy='fresh-only no retry; intermediate states are diagnostic artifacts, not permission to replay budget')
        write(out / 'status.json', summary)
        write(out / 'artifact_index.json', {str(p.relative_to(out)):dict(sha256=sha(p), bytes=p.stat().st_size)
              for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'artifact_index.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'model', 'data', 'head', 'output'):
        parser.add_argument('--' + name, required=True)
    run(parser.parse_args())
