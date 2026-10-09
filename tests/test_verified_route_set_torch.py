"""Server runtime checks; local runtime has no torch and must report a skip."""
import numpy as np
import pytest
torch = pytest.importorskip('torch')
from routeset.verified_route_set import build_targets
from scripts.research_v3_frequency import match_loss


@pytest.mark.parametrize('groups', [2, 8, 10])
def test_ordinary_detached_targets_match_existing_loss_and_gradient(groups):
    rng = np.random.default_rng(5)
    p = rng.normal(size=(8,24,3)).astype('float32')
    e = rng.uniform(size=(8,24)).astype('float32')
    r = rng.normal(size=(groups*3,24,3)).astype('float32')
    re = rng.uniform(size=(groups*3,24)).astype('float32')
    tags = np.repeat(['g%d'%i for i in range(groups)], 3)
    def check(p,e): return np.ones(len(p),bool), [None]*len(p)
    tp,te,_ = build_targets(p,e,r,re,tags,np.zeros((len(r),24)),np.zeros(8,bool),[None]*8,
                            'ordinary',np.random.default_rng(10),check)
    a=torch.tensor(np.concatenate((p[:,1:],.2*e[:,1:,None]),-1)[None],requires_grad=True)
    b=a.detach().clone().requires_grad_(True)
    truth=torch.tensor(np.concatenate((r[:,1:],.2*re[:,1:,None]),-1)[None])
    targets=torch.tensor(np.concatenate((tp[:,1:],.2*te[:,1:,None]),-1)[None])
    expected=match_loss(a,truth,tags[None],np.random.default_rng(10))
    actual=(b-targets).square().mean()
    torch.testing.assert_close(expected,actual)
    expected.backward(); actual.backward()
    torch.testing.assert_close(a.grad,b.grad)
