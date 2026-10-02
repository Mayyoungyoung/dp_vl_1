"""Pure strict configuration and inherited/added cost guards."""
from copy import deepcopy

import pytest

from routeset.vlm_sft_continuation import (TRAINER,HELPER,validate_continuation_config,
                                         continuation_accounting)


def configs():
    old = dict(steps=1500,lr=.0001,data_fingerprint='original',eval_every=250,seed=0,
        source_sha256={TRAINER:'old','loss.py':'unchanged','model.py':'unchanged'})
    new = deepcopy(old);new['steps']=6000
    new['source_sha256'].update({TRAINER:'new',HELPER:'new'})
    return old,new


def test_only_explicit_limit_and_registered_continuation_source_may_change():
    old,new = configs();validate_continuation_config(new,old)
    for key,value in [('lr',.0002),('data_fingerprint','other'),('eval_every',500),('seed',1),('steps',1500)]:
        bad=deepcopy(new);bad[key]=value
        with pytest.raises(ValueError):validate_continuation_config(bad,old)
    bad=deepcopy(new);bad['extra_setting']=1
    with pytest.raises(ValueError):validate_continuation_config(bad,old)
    for key in ('loss.py','model.py'):
        bad=deepcopy(new);bad['source_sha256'][key]='other'
        with pytest.raises(ValueError):validate_continuation_config(bad,old)
    bad=deepcopy(new);del bad['source_sha256'][HELPER]
    with pytest.raises(ValueError):validate_continuation_config(bad,old)


def test_added_cost_subtracts_inherited_once_and_never_swallows_failures():
    original = dict(TRAIN=dict(requests=1500,candidate_slots=3750),DEV_MODEL=dict(requests=322,candidate_slots=805))
    lineage = dict(inherited_elapsed_seconds=556.8,inherited_exposure=original)
    cumulative = dict(TRAIN=dict(requests=6001,candidate_slots=15001),DEV_MODEL=dict(requests=1150,candidate_slots=2875))
    accounting = continuation_accounting(lineage,cumulative,2226.8)
    assert accounting['added_elapsed_seconds'] == pytest.approx(1670.)
    # One hypothetical replay request remains charged, not erased to a planned count.
    assert accounting['added_exposure']['TRAIN'] == dict(requests=4501,candidate_slots=11251)
    assert accounting['cumulative_exposure']['TRAIN']['candidate_slots'] == 15001
    with pytest.raises(ValueError):continuation_accounting(lineage,cumulative,500.)
    bad=deepcopy(cumulative);bad['TRAIN']['requests']=1499
    with pytest.raises(ValueError):continuation_accounting(lineage,bad,2226.8)
