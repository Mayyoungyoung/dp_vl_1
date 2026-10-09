import pytest
from scripts.summarize_verified_replicates import paired_summary


def row(i,f,v):return dict(id=i,family=f,raw={'distinct':v})


def test_families_not_slots_or_repeated_seeds_are_independent_units():
    a=[row('a','A',2),row('b','A',2),row('c','B',0)]
    b=[row('a','A',0),row('b','A',0),row('c','B',0)]
    result=paired_summary([a,a],[b,b],'distinct')
    assert result['mean_difference']==1
    assert result['seed_differences']==[1,1]
    assert result['families']==2
    assert result['fixed_seed_family_CI95']==[0,2]


def test_reject_different_seed_populations_and_duplicate_ids():
    a=[row('a','A',1)];b=[row('b','A',1)]
    with pytest.raises(ValueError,match='seed populations'):
        paired_summary([a,b],[a,b],'distinct')
    with pytest.raises(ValueError,match='Duplicate'):
        paired_summary([a+a],[a+a],'distinct')
