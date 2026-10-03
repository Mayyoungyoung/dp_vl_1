"""Auditable serial common-head continuation; no model or data-policy changes."""
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path

PROTOCOL = 'two_row_common_head_serial_lora_3000_v1'
HEAD_SHA = 'ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3'
EXPORT_SHA = '04ba27290e382787d5f1764fc2645d4219b09457ea972d15f7d1f3baeac74fdc'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'
POLICY = dict(protocol=PROTOCOL, head_sha256=HEAD_SHA, export_sha256=EXPORT_SHA,
    prefix_manifest_sha256='4a3152db319e67c158f692a4b88e8575fd5e00b7ebe483ebfe4fafec1552bf15',
    model_revision=REVISION, steps=3000, accumulation=32, micro_batch=1,
    candidates=4, horizon=24, seed=0, sampler_seed=100000, adapter_seed=200000,
    head_lr=.0003, lora_lr=.00001, head_weight_decay=.0001, lora_weight_decay=0.,
    rank=8, alpha=16., eval_every=250, checkpoint_every=25, train_inputs=285,
    dev_inputs=36, requested_train_inputs=288, observation_draws=96000,
    candidate_path_states=384000, selection_opportunities=12,
    grad_clip=1., new_dev_allowed=False, automatic_stage_chaining=False)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False).encode() + b'\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))


def validate_policy(policy):
    if policy != POLICY:
        raise ValueError('Only the registered common-head serial3000 setting is supported')
    return policy


def draw_plan(ids, steps=3000, accumulation=32, seed=100000):
    import numpy as np
    ids = list(map(str, ids))
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('Unique ordered TRAIN input IDs required')
    rng = np.random.default_rng(seed)
    initial = rng.bit_generator.state
    draws = rng.choice(len(ids), size=(steps, accumulation), replace=True).tolist()
    body = dict(protocol='sealed_common_head_draws_v1', ids=ids, steps=steps,
        accumulation=accumulation, seed=seed, indices=draws,
        initial_sampler_state=initial, final_sampler_state=rng.bit_generator.state)
    return dict(body, fingerprint=hashlib.sha256(canonical(body)).hexdigest())


def seal_draw_plan(path, expected):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != expected:
            raise ValueError('Shared draw plan differs; never regenerate it')
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(canonical(expected) + b'\n')
            stream.flush(); os.fsync(stream.fileno())
    return digest(path)


class RequestJournal:
    """Durable issued-call ledger; checkpoint restoration cannot roll it back.

    Each tail/head/optimizer call is issued immediately before invocation. A
    raised call stays charged. Duplicate keys and partial/corrupt lines fail.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.records, self.keys, self.counts = [], set(), {}
        self.chain = '0' * 64
        if self.path.exists():
            with self.path.open('rb') as stream:
                for raw in stream:
                    if not raw.endswith(b'\n'):
                        raise ValueError('Partial request ledger; no automatic replay')
                    record = json.loads(raw)
                    self._accept(record)

    def _accept(self, record):
        body = {k: v for k, v in record.items() if k != 'sha256'}
        if (body.get('sequence') != len(self.records) or body.get('previous') != self.chain
                or record.get('sha256') != hashlib.sha256(canonical(body)).hexdigest()):
            raise ValueError('Request ledger hash/order mismatch')
        key = (body['kind'], body['key'])
        if key in self.keys:
            raise ValueError('A previously issued request cannot be repeated')
        self.keys.add(key); self.records.append(record); self.chain = record['sha256']
        self.counts[body['kind']] = self.counts.get(body['kind'], 0) + 1

    def issue(self, kind, key):
        if (kind, str(key)) in self.keys:
            raise ValueError('A previously issued request cannot be repeated')
        body = dict(sequence=len(self.records), previous=self.chain, kind=kind, key=str(key))
        record = dict(body, sha256=hashlib.sha256(canonical(body)).hexdigest())
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open('ab') as stream:
            stream.write(canonical(record) + b'\n')
            stream.flush(); os.fsync(stream.fileno())
        self._accept(record)

    def snapshot(self):
        return dict(records=len(self.records), sha256=self.chain, counts=dict(self.counts))

    def require_boundary(self, saved):
        if self.snapshot() != saved:
            raise ValueError('Uncheckpointed issued calls exist; no silent replay')


@contextmanager
def exclusive_lock(path):
    path = Path(path)
    descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    try:
        yield
    finally:
        path.unlink()


@contextmanager
def composite_evaluator(ordinary, verifier):
    """Change only the export verifier; no fresh-initialization or loop hooks."""
    previous = ordinary.verify_export
    ordinary.verify_export = verifier
    try:
        yield
    finally:
        ordinary.verify_export = previous


def check_config(current, saved):
    if current != saved:
        raise ValueError('Continuation identity/config differs, including source/cache/arm/budget')


def recovered_elapsed(checkpoint_seconds, sealed_pool_seconds):
    if any(not math.isfinite(x) or x<0 for x in (checkpoint_seconds,sealed_pool_seconds)):
        raise ValueError('Finite nonnegative recovery costs required')
    return checkpoint_seconds+sealed_pool_seconds


def should_pause(step, stop_after, total_steps):
    """Administrative boundary, never part of the immutable training setting."""
    if stop_after is None:return False
    if not isinstance(stop_after,int) or not 1<=stop_after<=total_steps or step>stop_after:
        raise ValueError('stop-after must be an unreached absolute logical step within the total budget')
    return step==stop_after and step<total_steps


class SerialTrainer:
    """Shared tested B1 accumulation loop. The supplied loss performs one draw."""
    def __init__(self, head, adapter_parameters, config, plan):
        import numpy as np
        import torch
        self.head, self.adapters, self.config, self.plan = head, adapter_parameters, config, plan
        self.rng = np.random.default_rng(config['seed'])
        groups = [dict(params=list(head.parameters()), lr=config['head_lr'],
                       weight_decay=config['head_weight_decay'])]
        if adapter_parameters:
            groups.append(dict(params=list(adapter_parameters.values()), lr=config['lora_lr'],
                               weight_decay=config['lora_weight_decay']))
        self.optimizer = torch.optim.AdamW(groups, betas=(.9, .999), eps=1e-8)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.)
        self.step, self.history, self.best = 0, [], None
        self.gradient_audit = {}
        self.recovered_sealed_evaluation_seconds = 0.

    def train_step(self, loss_function, journal):
        import torch
        step = self.step + 1
        if step > self.config['steps']:
            raise ValueError('Training budget exhausted')
        self.head.train(); self.optimizer.zero_grad(set_to_none=True)
        total = 0.
        for micro, slot in enumerate(self.plan['indices'][self.step]):
            loss = loss_function(self.plan['ids'][slot], step, micro, self.rng)
            if loss.ndim or not bool(torch.isfinite(loss)):
                raise ValueError('Finite scalar loss required')
            (loss / self.config['accumulation']).backward()
            total += float(loss.detach()) / self.config['accumulation']
        parameters = dict(('head.' + n, p) for n, p in self.head.named_parameters())
        parameters.update(self.adapters)
        if step<=2 or step%self.config['checkpoint_every']==0:
            self.gradient_audit = {n: dict(present=p.grad is not None,
                nonzero=p.grad is not None and bool(p.grad.count_nonzero()),
                norm=float(p.grad.double().norm()) if p.grad is not None else None)
                for n, p in parameters.items()}
        norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), self.config['grad_clip'], error_if_nonfinite=True)
        journal.issue('optimizer', step)
        self.optimizer.step(); self.scheduler.step(); self.step = step
        self.optimizer.zero_grad(set_to_none=True)
        return dict(step=step, loss=total, gradient_norm=float(norm),
                    learning_rates=[g['lr'] for g in self.optimizer.param_groups])

    def state_dict(self, journal, elapsed_seconds):
        from routeset.qwen_prefix_replay import cpu_copy
        from routeset.train_v2 import rng_state
        return dict(protocol=PROTOCOL, config=self.config, model=cpu_copy(self.head.state_dict()),
            adapters=cpu_copy(self.adapters), optimizer=cpu_copy(self.optimizer.state_dict()),
            scheduler=self.scheduler.state_dict(), rng=rng_state(self.rng), step=self.step,
            history=self.history, best=self.best, gradient_audit=self.gradient_audit,
            draw_fingerprint=self.plan['fingerprint'], draw_position=self.step*self.config['accumulation'],
            observation_draws=self.step*self.config['accumulation'],
            candidate_path_states=self.step*self.config['accumulation']*self.config['candidates'],
            journal=journal.snapshot(), elapsed_seconds=elapsed_seconds, pending_gradients=False,
            recovered_sealed_evaluation_seconds=self.recovered_sealed_evaluation_seconds)

    def load_state_dict(self, state, journal, allow_sealed_evaluation=False):
        from routeset.qwen_prefix_replay import restore_parameters
        from routeset.train_v2 import restore_rng
        check_config(self.config, state['config'])
        step = state['step']; draws = step*self.config['accumulation']
        if (state.get('protocol') != PROTOCOL or state.get('pending_gradients') is not False or
                not 0 <= step <= self.config['steps'] or state['draw_fingerprint'] != self.plan['fingerprint'] or
                state['draw_position'] != draws or state['observation_draws'] != draws or
                state['candidate_path_states'] != draws*self.config['candidates']):
            raise ValueError('Incomplete or inconsistent boundary checkpoint')
        if not allow_sealed_evaluation:
            journal.require_boundary(state['journal'])
        self.head.load_state_dict(state['model'], strict=True)
        restore_parameters(self.adapters, state['adapters'])
        self.optimizer.load_state_dict(state['optimizer']); self.scheduler.load_state_dict(state['scheduler'])
        expected_lrs=[self.config['head_lr']]+([self.config['lora_lr']] if self.adapters else [])
        if ([g['lr'] for g in self.optimizer.param_groups]!=expected_lrs or
                self.scheduler.base_lrs!=expected_lrs or self.scheduler.last_epoch!=step):
            raise ValueError('Saved optimizer/scheduler violates constant LR or step')
        restore_rng(state['rng'], self.rng)
        self.step, self.history, self.best = step, state['history'], state['best']
        self.gradient_audit = state['gradient_audit']
        self.recovered_sealed_evaluation_seconds=state['recovered_sealed_evaluation_seconds']


def verify_pool(folder, identity):
    folder = Path(folder)
    receipt = json.loads((folder/'pool_receipt.json').read_text())
    if receipt['identity'] != identity:
        raise ValueError('Saved evaluation pool identity differs')
    required = {'tail_features.npz','predictions.npz','per_scene.json','metrics.json'}
    if not required.issubset(receipt['artifacts']) or set(receipt['artifacts'])-required-{'paired_language_predictions.npz'}:
        raise ValueError('Incomplete or unexpected evaluation artifacts')
    for name, value in receipt['artifacts'].items():
        if Path(name).name != name or digest(folder/name) != value:
            raise ValueError('Saved evaluation artifact changed')
    if json.loads((folder/'metrics.json').read_text()) != receipt['metrics']:
        raise ValueError('Evaluation receipt metrics changed')
    return receipt


def reconcile_sealed_pool(journal, saved_boundary, receipt):
    """Only a complete sealed evaluation may extend a training checkpoint."""
    if receipt['journal_before'] != saved_boundary or receipt['journal_after'] != journal.snapshot():
        raise ValueError('No exact sealed-evaluation journal span')
    extra = journal.records[saved_boundary['records']:]
    n = receipt['identity']['requests']
    step = receipt['identity']['step']
    if len(extra) != 2*n or [r['kind'] for r in extra] != ['dev_tail']*n + ['dev_head']*n:
        raise ValueError('Unsealed/non-evaluation calls after checkpoint')
    expected = [str(step)+':'+identifier for identifier in receipt['identity']['ids']]
    if [r['key'] for r in extra] != expected + expected:
        raise ValueError('Sealed evaluation IDs changed')


def atomic_torch_save(path, value):
    import torch
    path = Path(path); temporary = path.with_name(path.name+'.tmp')
    with temporary.open('wb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(str(temporary), str(path))
