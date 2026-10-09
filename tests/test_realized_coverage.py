import numpy as np
import torch
from research_realized_coverage_v1.core import variants,replace_query
from research_realized_coverage_v1.allocator import SetUtility,SuccessHead,allocate
from routeset.mode_geometry import ModeGeometryHead

def test_replace_preserves_all_companion_pairs():
    m=np.array([1,1,2,3,4,5,6,7]);v=variants(m)
    a,b=replace_query(m,v,0,8)
    assert np.array_equal(a[1:],m[1:]) and np.array_equal(b[1:],v[1:])
    assert b[1]==1  # Do not recanonicalize the untouched repeated query.

def test_explicit_variants_match_old_default_and_preserve_geometry():
    torch.manual_seed(9);model=ModeGeometryHead(feature_dim=12,width=16,point_width=8,horizon=24,max_candidates=8)
    c=torch.randn(1,16);a=torch.randn(1,3);s=torch.randn(1,8);m=torch.tensor([[1,1,2,3,4,5,6,7]])
    v=torch.tensor(variants(m[0].numpy())[None]);p=model.decode(c,a,s,m)[0];q=model.decode(c,a,s,m,variant_ids=v)[0]
    torch.testing.assert_close(p,q,rtol=0,atol=0)

def test_set_utility_permutation_and_difference():
    torch.manual_seed(9);head=SetUtility();c=torch.randn(2,128);m=torch.arange(8).repeat(2,1);v=torch.zeros_like(m)
    p=head(c,m,v);order=torch.randperm(8)
    torch.testing.assert_close(p,head(c,m[:,order],v[:,order]),rtol=1e-5,atol=1e-5)
    assert torch.equal(p-p,torch.zeros_like(p))
    assert torch.isfinite(p).all()

def test_allocation_is_query_only_and_eight():
    c=torch.randn(1,128);m=torch.arange(8)[None];v=torch.zeros_like(m)
    for kind,head in [('success',SuccessHead()),('net',SetUtility())]:
        with torch.no_grad():a,b,cost=allocate(kind,head,c,m,v)
        assert a.shape==b.shape==(1,8) and cost['replacements']<=2

def test_net_counts_losses_not_just_additions():
    before={0,1,2};after={2,3}
    added=after-before;lost=before-after
    assert len(added)==1 and len(after)-len(before)==len(added)-len(lost)==-1
