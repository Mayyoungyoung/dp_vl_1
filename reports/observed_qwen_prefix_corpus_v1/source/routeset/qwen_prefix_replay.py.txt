"""Frozen Qwen3-VL prefix capture and serial last-two-layer replay.

This is an efficiency/ordinary-LoRA baseline utility, not a route mechanism.
Torch imports are lazy so protocol/manifest tests do not require a GPU runtime.
"""
import copy
import hashlib

PROTOCOL = 'qwen3vl_last_two_prefix_replay_probe_v1'
PARENTS = ('two_row_reach_283200', 'two_row_reach_283201')
IDS = tuple(p + '_target' + str(t) for p in PARENTS for t in range(3))
BUDGET = dict(full_features=10, replay_features=10, head_calls=4,
              candidate_path_states=16, full_optimizer_steps=2,
              replay_optimizer_steps=2, optimizer_steps_total=4)
HEAD_SHA = 'ce0b186b1f73beab2bd09b0582622e1d1b012fc88b892ab909479dd7deb250b3'
REVISION = '89644892e4d85e24eaac8bacfd4f463576704203'


def validate_policy(policy):
    expected = dict(protocol=PROTOCOL, parents=list(PARENTS), ids=list(IDS),
                    update_ids=[IDS[0], IDS[3]], head_checkpoint_sha256=HEAD_SHA,
                    model_revision=REVISION, cut_layer=26, language_layers=28,
                    rank=8, alpha=16.0, head_lr=0.0003, lora_lr=0.00001,
                    seed=0, candidates=4, micro_batch=1, logical_steps=2,
                    budget=BUDGET, feature_comparison='exact',
                    branch_comparison='exact', retry=False, resume=False,
                    allowed_split='TRAIN', min_pixels=65536, max_pixels=262144)
    if policy != expected:
        raise ValueError('The fixed six-TRAIN technical probe policy changed')
    return policy


def validate_rows(rows):
    if len(rows) != 6 or tuple(r.get('id') for r in rows) != IDS:
        raise ValueError('Only the exact first two registered TRAIN parents/three targets are allowed')
    for row, parent in zip(rows, [p for p in PARENTS for _ in range(3)]):
        if (set(row) != {'id', 'parent_id', 'split', 'image', 'instruction'}
                or row['parent_id'] != parent or row['split'] != 'TRAIN'
                or not isinstance(row['instruction'], str) or not row['instruction'].strip()):
            raise ValueError('Invalid observation-only TRAIN row')
    return rows


def map_tensors(value, function):
    import torch
    if torch.is_tensor(value):
        return function(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        return {k: map_tensors(v, function) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(v, function) for v in value)
    raise TypeError('Unsupported mutable/non-tensor replay value: ' + type(value).__name__)


def cpu_copy(value):
    return map_tensors(value, lambda x: x.detach().cpu().clone())


def tensor_hash(value):
    import torch
    data = value.detach().cpu().contiguous()
    # Flatten first: scalar parameters cannot directly view a larger element dtype.
    return hashlib.sha256(data.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def tensor_difference(a, b):
    import torch
    if a.shape != b.shape or a.dtype != b.dtype:
        return dict(exact=False, shape_dtype_match=False, max_abs=None, rms=None)
    delta = a.detach().cpu().double() - b.detach().cpu().double()
    return dict(exact=bool(torch.equal(a.detach().cpu(), b.detach().cpu())),
                shape_dtype_match=True, max_abs=float(delta.abs().max()) if delta.numel() else 0.,
                rms=float(delta.square().mean().sqrt()) if delta.numel() else 0.)


def differences(a, b, prefix=''):
    """Compare all state keys, including optimizer steps; missing != missing proof."""
    import torch
    result = {}
    if torch.is_tensor(a) and torch.is_tensor(b):
        d = tensor_difference(a, b)
        if not d['exact']:
            result[prefix] = d
    elif isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            result[prefix + ':keys'] = dict(left=sorted(map(str, a)), right=sorted(map(str, b)))
        for key in set(a) & set(b):
            result.update(differences(a[key], b[key], prefix + '/' + str(key)))
    elif isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        if len(a) != len(b):
            result[prefix + ':length'] = [len(a), len(b)]
        for i, (x, y) in enumerate(zip(a, b)):
            result.update(differences(x, y, prefix + '/' + str(i)))
    elif type(a) != type(b) or a != b:
        result[prefix] = dict(left=repr(a), right=repr(b))
    return result


def pool_hidden(hidden, valid_mask):
    if hidden.ndim != 3 or hidden.shape[0] != 1 or valid_mask.shape != hidden.shape[:2]:
        raise ValueError('Replay is deliberately serial, preserving original kernel shapes')
    selected = hidden[0, valid_mask[0].bool()].float()
    if not len(selected):
        raise ValueError('No valid input tokens')
    import torch
    return torch.cat((selected.mean(0), selected[-1]))[None]


def official_feature(backbone, inputs):
    result = backbone.model(**inputs, use_cache=False, return_dict=True)
    return pool_hidden(result.last_hidden_state, inputs['attention_mask'])


def capture_prefix(backbone, inputs, cut_layer=26):
    """One real official full forward; capture exactly the decoder call boundary."""
    import torch
    if torch.is_inference_mode_enabled():
        raise ValueError('Use no_grad, not inference_mode, for autograd-compatible cache tensors')
    if any(p.requires_grad for p in backbone.parameters()):
        raise ValueError('Capture before installing adapters; every base parameter must be frozen')
    layer = backbone.model.language_model.layers[cut_layer]
    captured = []

    def hook(module, args, kwargs):
        if len(args) != 1 or not torch.is_tensor(args[0]):
            raise ValueError('Unexpected official decoder positional arguments')
        required = {'position_embeddings', 'attention_mask', 'position_ids', 'cache_position', 'past_key_values'}
        if not required.issubset(kwargs) or kwargs['past_key_values'] is not None or kwargs.get('use_cache', False):
            raise ValueError('Only uncached official prefill is supported')
        if not isinstance(kwargs['position_embeddings'], tuple) or len(kwargs['position_embeddings']) != 2:
            raise ValueError('Actual multiaxis cos/sin tuple required')
        captured.append(dict(hidden=cpu_copy(args[0]), kwargs=cpu_copy(kwargs),
                             pooling_mask=cpu_copy(inputs['attention_mask']), cut_layer=cut_layer))

    handle = layer.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        with torch.no_grad():
            feature = official_feature(backbone, inputs)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise ValueError('Capture must execute the cut exactly once')
    return captured[0], cpu_copy(feature)


def replay_feature(backbone, payload):
    language = backbone.model.language_model
    device = next(language.layers[payload['cut_layer']].parameters()).device
    values = map_tensors(payload, lambda x: x.to(device))
    hidden, kwargs = values['hidden'], values['kwargs']
    if kwargs.get('past_key_values') is not None or kwargs.get('use_cache', False):
        raise ValueError('Past KV reuse would change the probe')
    # The installed Qwen3-VL injects DeepStack only after early text layers.
    # The cut already contains those additions; never invoke TextModel again.
    for layer in language.layers[payload['cut_layer']:]:
        hidden = layer(hidden, **kwargs)
    return pool_hidden(language.norm(hidden), values['pooling_mask'])


def adapter_parameters(backbone):
    return {n: p for n, p in backbone.named_parameters()
            if n.endswith('.lora_A') or n.endswith('.lora_B')}


def parameter_hashes(module, exclude_adapters=False):
    return {n: tensor_hash(p) for n, p in module.named_parameters()
            if not exclude_adapters or not (n.endswith('.lora_A') or n.endswith('.lora_B'))}


def restore_parameters(parameters, state):
    import torch
    if set(parameters) != set(state):
        raise ValueError('Adapter parameter identity changed')
    with torch.no_grad():
        for name, value in parameters.items():
            value.copy_(state[name].to(value))


class PairedTailUpdater:
    """Two independent optimizer/head branches sharing immutable base storage.

    The adapter parameters are temporarily restored to the pre-step state before
    the replay branch. Adam moments are never shared. Every state is compared.
    """
    def __init__(self, backbone, full_head, replay_head, head_lr, lora_lr):
        import torch
        self.adapters = adapter_parameters(backbone)
        self.heads = {'full': full_head, 'replay': replay_head}
        self.optimizers = {name: torch.optim.AdamW([
            dict(params=list(head.parameters()), lr=head_lr, weight_decay=1e-4),
            dict(params=list(self.adapters.values()), lr=lora_lr, weight_decay=0.)])
            for name, head in self.heads.items()}
        self.step_number = 0
        if differences(full_head.state_dict(), replay_head.state_dict()):
            raise ValueError('Paired heads must have identical actual initial tensors')

    def step(self, features, loss_function, on_optimizer=None):
        import torch
        before = cpu_copy(self.adapters)
        rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        branch = {}
        for name in ('full', 'replay'):
            restore_parameters(self.adapters, before)
            torch.set_rng_state(rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            head, optimizer = self.heads[name], self.optimizers[name]
            head.train()
            optimizer.zero_grad(set_to_none=True)
            feature = features[name]()
            if not feature.requires_grad:
                raise ValueError('Tail feature lost adapter autograd')
            loss, artifacts = loss_function(head, feature, name)
            if not bool(torch.isfinite(loss)):
                raise ValueError('Nonfinite probe loss')
            loss.backward()
            params = dict(self.adapters)
            params.update({'head.' + n: p for n, p in head.named_parameters()})
            gradients = {n: None if p.grad is None else cpu_copy(p.grad) for n, p in params.items()}
            gradient_audit = {n: dict(present=g is not None,
                finite=bool(torch.isfinite(g).all()) if g is not None else False,
                nonzero=bool(g.count_nonzero()) if g is not None else False,
                norm=float(g.double().norm()) if g is not None else None) for n, g in gradients.items()}
            torch.nn.utils.clip_grad_norm_(list(params.values()), 1.)
            if on_optimizer is None:
                optimizer.step()
            else:
                on_optimizer(name, optimizer.step)
            branch[name] = dict(feature=cpu_copy(feature), loss=cpu_copy(loss),
                gradients=gradients, gradient_audit=gradient_audit,
                adapters=cpu_copy(self.adapters), head=cpu_copy(head.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()), artifacts=cpu_copy(artifacts),
                rng=torch.get_rng_state().clone(),
                cuda_rng=cpu_copy(torch.cuda.get_rng_state_all()) if cuda_rng is not None else None)
        self.step_number += 1
        diff = differences(branch['full'], branch['replay'])
        return dict(step=self.step_number, exact=not diff, differences=diff, branches=branch)

    def state_dict(self):
        import torch
        return dict(step=self.step_number, adapters=cpu_copy(self.adapters),
                    heads={n: cpu_copy(h.state_dict()) for n, h in self.heads.items()},
                    optimizers={n: cpu_copy(o.state_dict()) for n, o in self.optimizers.items()},
                    torch_rng=torch.get_rng_state().clone(),
                    cuda_rng=cpu_copy(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else None)

    def load_state_dict(self, state):
        import torch
        if set(state) != {'step', 'adapters', 'heads', 'optimizers', 'torch_rng', 'cuda_rng'} or set(state['heads']) != set(self.heads) or set(state['optimizers']) != set(self.optimizers):
            raise ValueError('Complete paired state required')
        restore_parameters(self.adapters, state['adapters'])
        for name in self.heads:
            self.heads[name].load_state_dict(state['heads'][name], strict=True)
            self.optimizers[name].load_state_dict(state['optimizers'][name])
        self.step_number = state['step']
        torch.set_rng_state(state['torch_rng'].cpu())
        if state['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda_rng']])
