"""Pure grammar/token-fragment tests; no installed Torch/tokenizer required."""
import json
import pytest

from routeset.vlm_route_grammar import (ALPHABET, INITIAL, RouteJSONGrammar, RouteGrammarLogitsProcessor,
                                      TokenGrammarCache, tokenizer_fragments)


def answer(k=1, h=2):
    return json.dumps([[[10000, -10000, 0, 1], [-1, 1, -0, 0]][:h]]*k, separators=(',', ':'))


def test_every_two_fragment_split_and_many_multichar_chunks_preserve_exact_shape():
    for k in (1, 4):
        grammar = RouteJSONGrammar(k, 2); text = answer(k)
        for cut in range(1, len(text)):
            state = grammar.advance(INITIAL, text[:cut])
            assert state is not None and not grammar.complete(state)
            assert grammar.complete(grammar.advance(state, text[cut:]))
        for width in range(1, 12):
            state = INITIAL
            for start in range(0, len(text), width):
                state = grammar.advance(state, text[start:start+width])
                assert state is not None
            assert grammar.complete(state)


@pytest.mark.parametrize('bad', [
    '[[[10001,0,0,1]]]', '[[[-10001,0,0,1]]]', '[[[01,0,0,1]]]', '[[[0.1,0,0,1]]]',
    '[[[+1,0,0,1]]]', '[[[0,0,0,2]]]', '[[[0,0,0,-1]]]', '[[[0,0,0,true]]]',
    '[[[0,0,0,1,1]]]', '[[[0,0,0]]]', '[[[0,0,0,1],[0,0,0,1]]]', '[[[0,0,0,1]]][]'])
def test_range_integer_event_counts_and_extra_json_rejected(bad):
    assert RouteJSONGrammar(1, 1).advance(INITIAL, bad) is None


def test_prefix_sign_multichar_tokens_eos_and_dead_end():
    grammar = RouteJSONGrammar(1, 1)
    fragments = {0: '[[[', 1: '-', 2: '10000', 3: ',0,0,1]]]', 4: '[', 5: '10001'}
    cache = TokenGrammarCache(grammar, fragments, [99], capacity=2)
    constraint = RouteGrammarLogitsProcessor(cache, 10)
    assert 99 not in constraint.advance_ids([])
    assert 1 in constraint.advance_ids([0]) and 99 not in cache.allowed(constraint.state)
    assert 2 in constraint.advance_ids([0, 1])
    assert 99 not in constraint.advance_ids([0, 1, 2])
    assert constraint.advance_ids([0, 1, 2, 3]) == (99,)
    assert len(cache.cache) <= 2
    with pytest.raises(ValueError, match='history'):
        constraint.advance_ids([0])
    with pytest.raises(ValueError, match='stop after EOS'):
        constraint.advance_ids([0, 1, 2, 3, 99])
    with pytest.raises(ValueError, match='dead end'):
        TokenGrammarCache(grammar, {0: '9'}, [99]).allowed(INITIAL)


def test_fixed_max_token_prefix_remains_incomplete_no_padding_or_repair():
    grammar = RouteJSONGrammar(1, 24)
    state = grammar.advance(INITIAL, '[[[0,0,0,1]')
    assert state is not None and not grammar.complete(state)
    with pytest.raises(json.JSONDecodeError):
        json.loads('[[[0,0,0,1]')
    assert grammar.advance(state, ']]') is None


class FakeTokenizer:
    all_special_ids = [99]
    def __init__(self):
        self.vocab = {char: index for index, char in enumerate(sorted(ALPHABET))}
        self.vocab.update({'[[[': 50, '],[': 51, '100': 52, 'explanation': 53, 'Ġ1': 54, '<eos>': 99})
        self.backend_tokenizer = self
    def to_str(self): return json.dumps(dict(decoder=dict(type='ByteLevel')))
    def get_added_vocab(self): return {'<eos>': 99}
    def get_vocab(self): return self.vocab
    def decode(self, identifiers, **kwargs):
        reverse = {index: text for text, index in self.vocab.items()}
        return ''.join(reverse[index] for index in identifiers)


def test_tokenizer_compiler_preserves_multichar_fragments_and_rejects_unsupported_decoder():
    tokenizer = FakeTokenizer(); fragments = tokenizer_fragments(tokenizer)
    assert fragments[50] == '[[[' and fragments[51] == '],[' and fragments[52] == '100'
    assert 53 not in fragments and 54 not in fragments and 99 not in fragments
    tokenizer.to_str = lambda: json.dumps(dict(decoder=dict(type='WordPiece')))
    with pytest.raises(ValueError, match='ByteLevel'):
        tokenizer_fragments(tokenizer)
