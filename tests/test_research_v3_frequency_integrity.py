import numpy as np
from scripts.research_v3_frequency_integrity import replay_exposure


def test_exposure_replay_is_deterministic_excludes_dev_and_counts_full_sets():
    support=dict(ids=np.array(['train0','dev0']),splits=np.array(['TRAIN','DEV_MODEL']),
        mask=np.ones((2,4),bool),modes=np.array([['gap0|gap0']*2+['over|over']*2]*2))
    cfg=dict(seed=0,steps=3,batch_size=2)
    a=replay_exposure(support,cfg,'empirical_uniform')
    b=replay_exposure(support,cfg,'balanced')
    assert a==b and sum(a['global_counts'].values())==48
    assert [r['id'] for r in a['rows']]==['train0']
    full=replay_exposure(support,cfg,'set_matching')
    assert full['global_counts']=={'gap0|gap0':12,'over|over':12}
    assert full['unique_routes']==4 and full['unexposed_minority_request_modes']==0
