import numpy as np
import pytest
from scripts.research_v3_frequency import canonical_targets


def test_original_witness_selection_preserves_all_groups_and_rejects_broken_blocks():
    tags=np.repeat(np.array(['right','over','left']),5)
    chosen=canonical_targets(tags)
    assert chosen.tolist()==[0,5,10]
    assert tags[chosen].tolist()==['right','over','left']
    with pytest.raises(ValueError):canonical_targets(tags[:-1])
    tags[1]='left'
    with pytest.raises(ValueError):canonical_targets(tags)
