"""Bounded teacher-forcing diagnostics; conditional readouts are not rollouts."""
import re
import time

import numpy as np

CATEGORIES = ('syntax', 'coordinate', 'event', 'mixed', 'termination')
PROTOCOL = 'vlm_fixed_train8_teacher_forcing_v1'


class BodyBudget:
    """Cooperative boundary checks; an in-flight CUDA operation is not cancelled."""
    def __init__(self, seconds=60., clock=time.perf_counter):
        if not 0 < seconds <= 60:
            raise ValueError('At most 60 seconds diagnostic body permitted')
        self.seconds, self.clock = seconds, clock
        self.started = clock()

    def check(self):
        if self.clock() - self.started >= self.seconds:
            raise TimeoutError('Diagnostic body budget reached; no further operation started')

    def elapsed(self):
        return self.clock() - self.started


def token_layout(answer, answer_ids, offsets, full_ids, labels, prompt_tokens):
    """Require exact processor IDs, mask, next-token shift, and ASCII partition."""
    full_ids, labels = np.asarray(full_ids), np.asarray(labels)
    answer_ids, offsets = list(answer_ids), [list(map(int, x)) for x in offsets]
    if (full_ids.ndim != 1 or labels.shape != full_ids.shape or not 0 < prompt_tokens < len(labels)
            or not answer_ids or len(answer_ids) != len(offsets)):
        raise ValueError('Nonempty one-dimensional actual prompt/answer required')
    if not np.all(labels[:prompt_tokens] == -100) or not np.array_equal(labels[prompt_tokens:], full_ids[prompt_tokens:]):
        raise ValueError('Actual supervision mask differs from exact assistant-only labels')
    if not np.array_equal(full_ids[prompt_tokens:prompt_tokens+len(answer_ids)], answer_ids):
        raise ValueError('Standalone answer token IDs do not match actual processor sequence')
    if not answer.isascii() or offsets[0][0] != 0 or offsets[-1][1] != len(answer):
        raise ValueError('Exact compact ASCII answer offsets required')
    cursor = 0
    for begin, end in offsets:
        if begin != cursor or end <= begin or end > len(answer):
            raise ValueError('Answer offsets must form an exact disjoint character partition')
        cursor = end
    spans = []
    char_class = ['syntax'] * len(answer)
    for i, match in enumerate(re.finditer(r'-?(?:0|[1-9][0-9]*)', answer)):
        category = 'event' if i % 4 == 3 else 'coordinate'
        spans.append(dict(begin=match.start(), end=match.end(), value=int(match.group()),
                          point=i//4, field=i%4, category=category))
        char_class[match.start():match.end()] = [category] * len(match.group())
    if len(spans) != 24*4:
        raise ValueError('This fixed diagnostic requires exactly one H24 route')
    categories = []
    for begin, end in offsets:
        found = set(char_class[begin:end])
        categories.append(next(iter(found)) if len(found) == 1 else 'mixed')
    categories += ['termination'] * (len(full_ids)-prompt_tokens-len(answer_ids))
    if len(categories) != int(np.sum(labels != -100)):
        raise ValueError('Every supervised token must have exactly one category')
    positions = np.flatnonzero(labels != -100).tolist()
    return dict(categories=categories, answer_offsets=offsets, scalar_spans=spans,
                supervised_positions=positions, hidden_positions=[p-1 for p in positions],
                answer_token_count=len(answer_ids), prompt_tokens=int(prompt_tokens))


def category_statistics(categories, nll, top_ids, target_ids):
    nll = np.asarray(nll, dtype=np.float64)
    correct = np.asarray(top_ids) == np.asarray(target_ids)
    if len(categories) != len(nll) or correct.shape != nll.shape or not np.isfinite(nll).all():
        raise ValueError('Finite per-token statistics with matching lengths required')
    if any(c not in CATEGORIES for c in categories):
        raise ValueError('Unrecognized token category')
    result = {}
    for category in CATEGORIES:
        mask = np.asarray([c == category for c in categories]); count = int(mask.sum())
        result[category] = dict(tokens=count, nll_sum=float(nll[mask].sum()),
            nll_mean=float(nll[mask].mean()) if count else None,
            top1_correct=int(correct[mask].sum()), top1_accuracy=float(correct[mask].mean()) if count else None)
    return result


def scalar_readouts(answer, layout, decoded_top_pieces):
    """Read a scalar only if tokens touching it preserve both surrounding spans.

    Mixed tokens are not sliced by predicted character length. All adjacent
    original characters inside their token spans must be reproduced exactly;
    otherwise the scalar is unparseable. This can conservatively reject good
    numbers when punctuation or another scalar in the same token is wrong.
    """
    offsets = layout['answer_offsets']
    if len(decoded_top_pieces) < len(offsets):
        raise ValueError('Missing conditional top1 token fragments')
    result = []
    for span in layout['scalar_spans']:
        overlapping = [i for i, (a, b) in enumerate(offsets) if a < span['end'] and b > span['begin']]
        first, last = overlapping[0], overlapping[-1]
        prefix = answer[offsets[first][0]:span['begin']]
        suffix = answer[span['end']:offsets[last][1]]
        joined = ''.join(decoded_top_pieces[i] for i in overlapping)
        value = None
        if joined.startswith(prefix) and joined.endswith(suffix) and len(joined) >= len(prefix)+len(suffix):
            number = joined[len(prefix):len(joined)-len(suffix) if suffix else None]
            if re.fullmatch(r'-?(?:0|[1-9][0-9]*)', number):
                trial = int(number)
                if (span['category'] == 'event' and trial in (0, 1)) or (span['category'] == 'coordinate' and abs(trial) <= 10000):
                    value = trial
        result.append(dict(**span, token_indices=overlapping, conditional_top1_text=joined,
                           predicted_integer=value, signed_integer_error=None if value is None else value-span['value']))
    return result


def causal_mutation(full_ids, layout, replacement_ids):
    """Last pure coordinate token, deterministic different single-digit token."""
    candidates = [i for i, c in enumerate(layout['categories'][:layout['answer_token_count']]) if c == 'coordinate']
    if not candidates:
        raise ValueError('No pure coordinate token available for declared causal control')
    position = layout['prompt_tokens'] + candidates[-1]
    original = int(full_ids[position])
    choices = [int(t) for t in replacement_ids if int(t) != original]
    if not choices:
        raise ValueError('A different one-token coordinate replacement is required')
    return dict(position=position, old_token_id=original, new_token_id=choices[0],
                compared_prefix_positions=position, comparison='all positions [0, position), all vocabulary logits')


def chunk_token_statistics(hidden, labels, lm_head, chunk_size=64, check=lambda: None):
    """Use the same t->t+1 states/labels as the unchanged training objective."""
    import torch
    from torch.nn import functional as F
    if hidden.ndim != 3 or hidden.shape[0] != 1 or labels.shape != hidden.shape[:2] or chunk_size < 1:
        raise ValueError('One actual sequence and matching labels required')
    if any(p.requires_grad for p in lm_head.parameters()):
        raise ValueError('Diagnostic vocabulary head must be frozen')
    positions = torch.nonzero(labels[0, 1:] != -100).flatten() + 1
    if not len(positions):
        raise ValueError('No supervised answer tokens')
    top, nll, targets = [], [], []
    for begin in range(0, len(positions), chunk_size):
        check(); p = positions[begin:begin+chunk_size]
        logits = lm_head(hidden[0, p-1]).float()
        target = labels[0, p]
        if not bool(torch.isfinite(logits).all()):
            raise ValueError('Nonfinite real model logits')
        nll.extend(F.cross_entropy(logits, target, reduction='none').cpu().tolist())
        top.extend(logits.argmax(-1).cpu().tolist()); targets.extend(target.cpu().tolist())
    return dict(supervised_positions=positions.cpu().tolist(), nll=nll, top_ids=top, target_ids=targets)


def compare_prefix_logits(first, changed, position, lm_head, chunk_size=64,
                          atol=1e-3, rtol=1e-4, check=lambda: None):
    """No logits are skipped or persisted; float32 compare after original head."""
    import torch
    if first.shape != changed.shape or first.ndim != 3 or first.shape[0] != 1 or not 0 < position < first.shape[1]:
        raise ValueError('Matching complete hidden states and interior mutation required')
    compared = violations = 0
    max_abs = max_scaled = 0.
    for begin in range(0, position, chunk_size):
        check(); end = min(position, begin+chunk_size)
        x, y = lm_head(first[0, begin:end]).float(), lm_head(changed[0, begin:end]).float()
        if not bool(torch.isfinite(x).all() and torch.isfinite(y).all()):
            raise ValueError('Nonfinite prefix logits')
        delta = (x-y).abs(); tolerance = atol+rtol*x.abs()
        max_abs = max(max_abs, float(delta.max())); max_scaled = max(max_scaled, float((delta/tolerance).max()))
        violations += int((delta > tolerance).sum()); compared += x.numel()
    return dict(atol=atol, rtol=rtol, compared_positions=position, compared_logit_values=compared,
        max_absolute_difference=max_abs, max_difference_over_tolerance=max_scaled,
        values_outside_tolerance=violations, passed=violations == 0,
        hidden_max_absolute_difference=float((first[:, :position].float()-changed[:, :position].float()).abs().max()),
        hidden_at_or_after_mutation_max_difference=float((first[:, position:].float()-changed[:, position:].float()).abs().max()),
        interpretation='Actual BF16 head outputs compared in FP32 with predeclared tolerances; not quantization equivalence or free-generation accuracy')
