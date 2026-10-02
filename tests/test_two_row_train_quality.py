import numpy as np
import json
from pathlib import Path
import pytest
from scripts.audit_two_row_train_reference_quality import endpoint_support,trajectory_statistics,registered_train_scope,train_rows


def test_axis_support_is_not_confused_with_euclidean_bound_or_invalid_pixels():
    xyz=np.array([[.049,.049,.049],[0.,0.,0.]])
    result=endpoint_support(np.zeros(3),xyz,[True,False])
    assert result['axis_bound_representable'] and result['nearest_L2_m']>.05
    assert not endpoint_support(np.zeros(3),[[.051,0,0]],[True])['axis_bound_representable']
    assert not endpoint_support(np.zeros(3),xyz,[False,False])['axis_bound_representable']


def test_long_high_arc_is_described_not_shortened_or_filtered():
    path=np.array([[0,0,.865],[.2,0,1.8],[.4,0,.84]])
    original=path.copy();result=trajectory_statistics(path)
    assert result['maximum_world_z_m']==1.8 and result['length_m']>1.8
    assert np.array_equal(path,original)


@pytest.mark.parametrize('count',[16,32,64])
def test_quality_covers_every_registered_train_input_and_keeps_zero_refs(count):
    selection=json.loads((Path(__file__).resolve().parents[1]/f'configs/observed_two_row_prefix{count+12}_selection_v1.json').read_text())
    indices,parents=registered_train_scope(selection)
    assert indices==list(range(count)) and len(parents)==count
    rows=[dict(id=parent+'_target0',parent_id=parent,split='TRAIN',routes=[]) for parent in sorted(parents)]
    dev=dict(id='two_row_reach_283264_target0',parent_id='two_row_reach_283264',split='DEV_MODEL',routes=['not opened'])
    inputs,labels=train_rows(selection,rows+[dev],rows+[dev])
    assert len(inputs)==len(labels)==count and all(not r['routes'] for r in labels)
    with pytest.raises(ValueError,match='every recorded input'):train_rows(selection,rows,rows[:-1])
    with pytest.raises(ValueError,match='every recorded input'):train_rows(selection,rows+rows[:1],rows)
    forbidden=dict(id='two_row_reach_283300_target0',parent_id='two_row_reach_283300',split='TRAIN',routes=[])
    with pytest.raises(ValueError,match='Fixed TRAIN'):train_rows(selection,rows+[forbidden],rows+[forbidden])
