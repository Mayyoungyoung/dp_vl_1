import copy
import numpy as np
from scripts.research_v3_frequency import sampled_distinct_targets


def test_stratified_sampler_has_exact_budget_and_no_impossible_mode_coverage():
    rng=np.random.default_rng(41)
    for count in (2,6,7,8,9):
        tags=np.repeat(np.array(['m%d'%i for i in range(count)]),5)
        chosen=sampled_distinct_targets(tags,rng)
        assert len(chosen)==8 and len(set(tags[chosen]))==min(count,8)
        assert ((chosen>=0)&(chosen<len(tags))).all()


def test_new_sampler_resume_preserves_every_actual_target():
    tags=np.repeat(np.array(['m%d'%i for i in range(9)]),5)
    rng=np.random.default_rng(17)
    for _ in range(5):sampled_distinct_targets(tags,rng)
    state=copy.deepcopy(rng.bit_generator.state)
    expected=[sampled_distinct_targets(tags,rng).tolist() for _ in range(5)]
    restored=np.random.default_rng();restored.bit_generator.state=state
    assert [sampled_distinct_targets(tags,restored).tolist() for _ in range(5)]==expected
