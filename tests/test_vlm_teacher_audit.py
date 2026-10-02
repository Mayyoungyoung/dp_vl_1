"""Pure alignment/readout tests; no Qwen or new data accessed."""
import json

import numpy as np
import pytest

from routeset.vlm_teacher_audit import (BodyBudget, token_layout, category_statistics,
    scalar_readouts, causal_mutation)


def fixture():
    answer = json.dumps([[[-123,456,789,1]]*24], separators=(',', ':'))
    # Include tokens spanning syntax and a numeric sign, and multi-digit tokens.
    pieces = []
    i = 0
    while i < len(answer):
        size = 2 if answer[i:i+2] == ',-' else 1
        pieces.append(answer[i:i+size]); i += size
    vocabulary = {piece:i+10 for i,piece in enumerate(dict.fromkeys(pieces))}
    ids = [vocabulary[p] for p in pieces]
    offsets, position = [], 0
    for piece in pieces:
        offsets.append([position, position+len(piece)]); position += len(piece)
    full = np.asarray([5,6,7]+ids+[1,2]); labels = full.copy(); labels[:3] = -100
    layout = token_layout(answer, ids, offsets, full, labels, 3)
    return answer, pieces, ids, offsets, full, labels, layout


def test_actual_mask_shift_partition_and_roundtrip_readout():
    answer,pieces,ids,offsets,full,labels,layout = fixture()
    assert layout['hidden_positions'][0] == 2
    assert layout['supervised_positions'] == list(range(3,len(full)))
    assert layout['categories'][-2:] == ['termination']*2
    assert len(layout['categories']) == len(full)-3
    readout = scalar_readouts(answer, layout, pieces+['<eos>','\n'])
    assert len(readout) == 96 and all(r['signed_integer_error'] == 0 for r in readout)
    stats = category_statistics(layout['categories'], [1.]*(len(full)-3), full[3:], full[3:])
    assert sum(v['tokens'] for v in stats.values()) == len(full)-3
    assert sum(v['nll_sum'] for v in stats.values()) == len(full)-3
    assert all(v['top1_correct'] == v['tokens'] for v in stats.values())


@pytest.mark.parametrize('fault', ['mask','label','answer_ids','offsets'])
def test_reject_silent_mask_token_or_offset_drift(fault):
    answer,pieces,ids,offsets,full,labels,layout = fixture()
    if fault == 'mask': labels[1] = full[1]
    elif fault == 'label': labels[5] += 1
    elif fault == 'answer_ids': ids[4] += 100
    else: offsets[3][0] -= 1
    with pytest.raises(ValueError):
        token_layout(answer, ids, offsets, full, labels, 3)


def test_mixed_token_has_no_invented_character_slicing():
    answer = json.dumps([[[-12,3,4,1]]*24], separators=(',', ':'))
    # Deliberately combine an entire coordinate plus its two delimiters.
    pieces = [answer[:2], answer[2:7]]+list(answer[7:])
    ids = list(range(10,10+len(pieces))); cursor = 0; offsets = []
    for piece in pieces:
        offsets.append([cursor,cursor+len(piece)]);cursor += len(piece)
    full = np.asarray([2,3]+ids+[1]); labels = full.copy();labels[:2] = -100
    layout = token_layout(answer, ids, offsets, full, labels, 2)
    assert layout['categories'][1] == 'mixed'
    # The fixed '[' and ',' survive while integer width changes: valid scalar.
    altered = pieces.copy(); altered[1] = '[-123,'
    assert scalar_readouts(answer, layout, altered)[0]['predicted_integer'] == -123
    altered[1] = 'x-123,'
    assert scalar_readouts(answer, layout, altered)[0]['predicted_integer'] is None
    altered[1] = '[-01,'
    assert scalar_readouts(answer, layout, altered)[0]['predicted_integer'] is None


def test_causal_control_mutates_late_answer_only_and_body_is_bounded():
    answer,pieces,ids,offsets,full,labels,layout = fixture()
    control = causal_mutation(full, layout, [800,801])
    p = control['position']
    assert p > .9*len(full) and p < len(full)-2
    assert control['old_token_id'] == full[p] and control['new_token_id'] != full[p]
    assert layout['categories'][p-3] == 'coordinate'
    clock = [0.]
    budget = BodyBudget(60., lambda:clock[0]); clock[0] = 59.99; budget.check()
    clock[0] = 60.
    with pytest.raises(TimeoutError): budget.check()
    with pytest.raises(ValueError): BodyBudget(61.)


def test_invalid_conditional_scalars_remain_unparseable():
    answer,pieces,ids,offsets,full,labels,layout = fixture()
    edited = pieces.copy()
    first = layout['scalar_spans'][0]
    affected = [i for i,(a,b) in enumerate(offsets) if a < first['end'] and b > first['begin']]
    edited[affected[0]] = 'true'
    assert scalar_readouts(answer, layout, edited)[0]['predicted_integer'] is None
    event = layout['scalar_spans'][3]
    index = next(i for i,(a,b) in enumerate(offsets) if a == event['begin'])
    edited[index] = '2'
    assert scalar_readouts(answer, layout, edited)[3]['predicted_integer'] is None
