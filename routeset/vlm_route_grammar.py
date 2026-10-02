"""Standard finite-state JSON logit constraint for the existing route protocol.

Only syntax/counts and the parser's universal numeric bounds are constrained.
Coordinates still come from model logits. No route repair or geometry access.
"""
from collections import OrderedDict
import hashlib
import json
import time

GRAMMAR_PROTOCOL = 'vlm_compact_exact_k_h24_integer_json_v1'
ALPHABET = frozenset('[],-0123456789')
INITIAL = ('outer_open', 0, 0, 0, '')


class RouteJSONGrammar:
    def __init__(self, k, horizon=24):
        if k not in (1, 4) or horizon < 1:
            raise ValueError('This baseline supports K1/K4 and a positive horizon')
        self.k, self.horizon = k, horizon

    def advance(self, state, fragment):
        """Consume every character of an arbitrary token fragment, or reject it."""
        if not fragment or any(c not in ALPHABET for c in fragment):
            return None
        for char in fragment:
            phase, route, point, field, number = state
            if phase == 'outer_open':
                if char != '[': return None
                state = ('route_open', route, point, field, '')
            elif phase == 'route_open':
                if char != '[': return None
                state = ('point_open', route, 0, 0, '')
            elif phase == 'point_open':
                if char != '[': return None
                state = ('integer', route, point, 0, '')
            elif phase == 'integer':
                if char == '-' and not number and field < 3:
                    state = (phase, route, point, field, '-'); continue
                if char.isdigit():
                    proposed = number+char
                    unsigned = proposed.lstrip('-')
                    if (len(unsigned) > 1 and unsigned[0] == '0') or len(unsigned) > 5:
                        return None
                    if field == 3 and (number or char not in '01'):
                        return None
                    if abs(int(proposed)) > 10000:
                        return None
                    state = (phase, route, point, field, proposed); continue
                if not number or number == '-': return None
                if field < 3 and char == ',':
                    state = (phase, route, point, field+1, '')
                elif field == 3 and char == ']':
                    state = ('after_point', route, point, 0, '')
                else: return None
            elif phase == 'after_point':
                if point+1 < self.horizon and char == ',':
                    state = ('point_open', route, point+1, 0, '')
                elif point+1 == self.horizon and char == ']':
                    state = ('after_route', route, point, 0, '')
                else: return None
            elif phase == 'after_route':
                if route+1 < self.k and char == ',':
                    state = ('route_open', route+1, 0, 0, '')
                elif route+1 == self.k and char == ']':
                    state = ('complete', route, point, 0, '')
                else: return None
            else:
                return None
        return state

    @staticmethod
    def complete(state):
        return state is not None and state[0] == 'complete'


def tokenizer_fragments(tokenizer):
    """Exact ASCII ByteLevel tokens, including multi-character punctuation.

Filtering vocabulary spellings is valid only for the checked ByteLevel decoder.
Each retained token is independently decoded and checked. Added/special tokens
are excluded; EOS is handled separately. Compact JSON matches training targets.
"""
    backend = json.loads(tokenizer.backend_tokenizer.to_str())
    if backend.get('decoder', {}).get('type') != 'ByteLevel':
        raise ValueError('Only the pinned Qwen ByteLevel decoder is supported')
    excluded = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    candidates = sorted((index, text) for text, index in tokenizer.get_vocab().items()
                        if index not in excluded and text and all(c in ALPHABET for c in text))
    fragments = {}
    for index, text in candidates:
        decoded = tokenizer.decode([index], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        if decoded != text:
            raise ValueError('ByteLevel token spelling/decode mismatch')
        fragments[index] = text
    if not ALPHABET.issubset(set(fragments.values())):
        raise ValueError('Individual grammar characters must all have model tokens')
    # With ASCII ByteLevel fragments, concatenation is context-independent.
    identifiers = list(fragments)
    if tokenizer.decode(identifiers, skip_special_tokens=False, clean_up_tokenization_spaces=False) != ''.join(fragments.values()):
        raise ValueError('Tokenizer fragments are not concatenation-stable')
    return fragments


class TokenGrammarCache:
    """Bounded CPU transition-mask cache, reused within the same K protocol."""
    def __init__(self, grammar, fragments, eos_token_ids, capacity=4096):
        self.grammar, self.fragments = grammar, dict(fragments)
        self.eos_token_ids = tuple(sorted(set(eos_token_ids)))
        if not self.eos_token_ids or any(index in fragments for index in self.eos_token_ids):
            raise ValueError('Separate nonempty EOS token set required')
        if capacity < 1:
            raise ValueError('Positive cache capacity required')
        self.capacity, self.cache = capacity, OrderedDict()
        self.hits = self.misses = 0

    def allowed(self, state):
        if state in self.cache:
            self.hits += 1; self.cache.move_to_end(state); return self.cache[state]
        self.misses += 1
        ids = self.eos_token_ids if self.grammar.complete(state) else tuple(
            index for index, fragment in self.fragments.items() if self.grammar.advance(state, fragment) is not None)
        if not ids:
            raise ValueError('Grammar dead end: no legal model-vocabulary continuation')
        self.cache[state] = ids
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return ids


class RouteGrammarLogitsProcessor:
    """Transformers-compatible batch1 callable; preserve all allowed logits.

The Transformers 4.57.1 pipeline applies this before temperature/top-p. Callers
must keep the original sampler and max_new_tokens and count all execution time.
"""
    def __init__(self, cache, prompt_length):
        self.cache, self.prompt_length = cache, int(prompt_length)
        self.state, self.consumed = INITIAL, ()
        self.calls, self.wall_seconds = 0, 0.
        if self.prompt_length < 1:
            raise ValueError('Actual positive prompt token length required')

    def advance_ids(self, generated):
        generated = tuple(generated)
        if generated[:len(self.consumed)] != self.consumed:
            raise ValueError('Single-request token history changed or rewound')
        for token in generated[len(self.consumed):]:
            if token in self.cache.eos_token_ids:
                raise ValueError('Generation must stop after EOS; EOS cannot continue')
            fragment = self.cache.fragments.get(token)
            self.state = self.cache.grammar.advance(self.state, fragment) if fragment else None
            if self.state is None:
                raise ValueError('Generated token violates grammar; no repair is permitted')
        self.consumed = generated
        return self.cache.allowed(self.state)

    def __call__(self, input_ids, scores):
        import torch
        started = time.perf_counter()
        if input_ids.ndim != 2 or scores.ndim != 2 or input_ids.shape[0] != 1 or scores.shape[0] != 1:
            raise ValueError('Grammar baseline requires one sample and one beam per request')
        if input_ids.shape[1] < self.prompt_length:
            raise ValueError('Input shorter than the registered prompt')
        allowed = self.advance_ids(input_ids[0, self.prompt_length:].tolist())
        if max(allowed) >= scores.shape[1] or min(allowed) < 0:
            raise ValueError('Grammar token is outside the actual model vocabulary')
        indices = torch.tensor(allowed, dtype=torch.long, device=scores.device)
        values = scores[0, indices]
        if torch.isnan(values).any().item() or torch.isposinf(values).any().item() or not torch.isfinite(values).any().item():
            raise ValueError('No finite allowed model logit; reject instead of inventing a token')
        masked = torch.full_like(scores, -float('inf'))
        masked[0, indices] = values
        self.calls += 1; self.wall_seconds += time.perf_counter()-started
        return masked


def fragment_fingerprint(fragments):
    return hashlib.sha256(json.dumps(sorted(fragments.items()), separators=(',', ':')).encode()).hexdigest()
