import pytest
from scripts.analyze_verified_set import contrast


def row(ident,family,value):
    return dict(id=ident,family=family,raw=dict(distinct=value))


def test_family_weighting_and_prespecified_subset():
    a=[row('a','f1',1),row('b','f1',1),row('c','f2',0)]
    b=[row('a','f1',0),row('b','f1',0),row('c','f2',0)]
    result=contrast(a,b,'distinct')
    assert result['difference']==.5 and result['families']==2 and result['requests']==3
    subset=contrast(a,b,'distinct',{'a','b'})
    assert subset['difference']==1 and subset['family_bootstrap95']==[1,1]


def test_missing_failed_requests_cannot_be_silently_dropped():
    with pytest.raises(AssertionError):
        contrast([row('a','f1',1)],[row('b','f1',0)],'distinct')
