"""Small actual tensor checks; these do not replace real Qwen causal audit."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from routeset.vlm_sft_loss import causal_chunked_loss
from routeset.vlm_teacher_audit import chunk_token_statistics, compare_prefix_logits


def test_actual_per_token_causal_shift_matches_original_loss():
    torch.manual_seed(81)
    hidden = torch.randn(1,19,7)
    head = torch.nn.Linear(7,31, bias=False).requires_grad_(False)
    labels = torch.randint(31,(1,19)); labels[:,:5] = -100
    with torch.inference_mode():
        stats = chunk_token_statistics(hidden, labels, head, 3)
        reference = causal_chunked_loss(hidden, labels, head, 3)
        dense = torch.nn.functional.cross_entropy(head(hidden[0,4:-1]), labels[0,5:], reduction='none')
    assert stats['supervised_positions'] == list(range(5,19))
    np.testing.assert_allclose(stats['nll'], dense.numpy(), rtol=1e-6)
    assert abs(np.mean(stats['nll'])-float(reference)) < 1e-6
    assert stats['top_ids'] == head(hidden[0,4:-1]).argmax(-1).tolist()
    labels[:] = -100
    with pytest.raises(ValueError): chunk_token_statistics(hidden, labels, head)


def test_complete_prefix_all_vocab_detects_interior_non_top1_change():
    # Identity head makes the lower-logit coordinate differ without top1 change.
    first = torch.zeros(1,11,4); first[:,:,0] = 100.
    changed = first.clone(); changed[:,8:,0] += 5.
    head = torch.nn.Linear(4,4, bias=False).requires_grad_(False)
    head.weight.copy_(torch.eye(4))
    control = compare_prefix_logits(first, changed, 8, head, chunk_size=3)
    assert control['passed'] and control['compared_logit_values'] == 8*4
    assert control['hidden_at_or_after_mutation_max_difference'] == 5.
    changed[0,2,3] = .1
    assert torch.equal(head(first[:,:8]).argmax(-1), head(changed[:,:8]).argmax(-1))
    control = compare_prefix_logits(first, changed, 8, head, chunk_size=3)
    assert not control['passed'] and control['values_outside_tolerance'] == 1
    assert control['max_absolute_difference'] == pytest.approx(.1)
