"""Actual CPU tensor mask tests; no model download/GPU/transformers required."""
import pytest
torch = pytest.importorskip('torch')
from routeset.vlm_route_grammar import RouteJSONGrammar, RouteGrammarLogitsProcessor, TokenGrammarCache


def test_mask_keeps_model_logits_exact_and_never_chooses_a_replacement_value():
    cache = TokenGrammarCache(RouteJSONGrammar(1, 1), {0: '[[[', 1: '0,0,0,1]]]', 2: 'wrong'}, [3])
    processor = RouteGrammarLogitsProcessor(cache, 2)
    logits = torch.tensor([[.123, .456, 50., -.987]])
    masked = processor(torch.tensor([[7, 8]]), logits)
    assert masked[0, 0] == logits[0, 0] and torch.isneginf(masked[0, 1:]).all()
    masked = processor(torch.tensor([[7, 8, 0]]), logits)
    assert masked[0, 1] == logits[0, 1] and torch.isneginf(masked[0, [0, 2, 3]]).all()
    masked = processor(torch.tensor([[7, 8, 0, 1]]), logits)
    assert masked[0, 3] == logits[0, 3] and torch.isneginf(masked[0, :3]).all()
    torch.testing.assert_close(logits, torch.tensor([[.123, .456, 50., -.987]]), rtol=0, atol=0)
    assert processor.calls == 3


def test_all_allowed_nonfinite_scores_or_batch_expansion_rejected():
    cache = TokenGrammarCache(RouteJSONGrammar(1, 1), {0: '[[['}, [1])
    processor = RouteGrammarLogitsProcessor(cache, 1)
    with pytest.raises(ValueError, match='finite'):
        processor(torch.tensor([[7]]), torch.tensor([[-float('inf'), 3.]]))
    with pytest.raises(ValueError, match='one sample'):
        processor(torch.tensor([[7], [8]]), torch.zeros(2, 2))
    two_legal = TokenGrammarCache(RouteJSONGrammar(1, 1), {0: '[', 1: '[['}, [2])
    with pytest.raises(ValueError, match='finite'):
        RouteGrammarLogitsProcessor(two_legal, 1)(torch.tensor([[7]]), torch.tensor([[float('inf'), 0., 1.]]))
