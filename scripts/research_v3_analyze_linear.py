"""Matched additional-training control with an unchanged complete scorer."""
import argparse
import numpy as np
from scripts.research_v3_frequency import RUN
from scripts.research_v3_analyze_frequency import paired
from scripts.run_observed_probability import read,write,sha


def replay():
    root=RUN/'matched_q_paired_v1'
    parent=RUN/'safety_mean/evaluation_matched_q_v1'
    old=root/'evaluation/paired_dev/mean_seed0/pool.npz'
    cal=root/'reliability/mean_seed0/calibration_seed0/predictions.npz'
    with np.load(parent/'pool.npz') as new,np.load(old) as pool,np.load(cal) as pred:
        for k in ('paths','events','labels','ids'):np.testing.assert_array_equal(new[k],pool[k])
        np.testing.assert_allclose(new['q'],pred['q'],rtol=0,atol=1e-6)
        error=float(np.max(np.abs(new['q']-pred['q'])))
    return dict(paths_events_labels_ids_exact=True,q_max_abs_error=error,
        parent_pool_sha256=sha(parent/'pool.npz'),original_pool_sha256=sha(old),calibrated_sha256=sha(cal))


def main(output):
    replayed=replay();out=RUN/output;out.mkdir(exist_ok=False)
    metrics={};rows={};training={};configs={}
    for arm in ('mean','linear'):
        root=RUN/('margin_'+arm);training[arm]=read(root/'summary.json');configs[arm]=read(root/'config.json')
        ev=root/'evaluation_matched_q_v1';metrics[arm]=read(ev/'metrics.json');rows[arm]=read(ev/'rows.json')
        assert metrics[arm]['generator_sha256']==training[arm]['last_sha256']==sha(root/'last.pt')
        assert metrics[arm]['pool_sha256']==sha(ev/'pool.npz')
        assert training[arm]['steps']==1200
    a,b=(training[k] for k in ('mean','linear'))
    assert a['sampler']==b['sampler'] and a['initial_sha256']==b['initial_sha256']
    assert a['mode_exposure']==b['mode_exposure'] and a['unique_routes']==b['unique_routes']
    assert configs['mean']['initial_checkpoint_sha256']==configs['linear']['initial_checkpoint_sha256']==sha(RUN/'safety_mean/last.pt')
    assert metrics['mean']['scorer_bundle_sha256']==metrics['linear']['scorer_bundle_sha256']=='2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e'
    delta=paired(rows['mean'],rows['linear'])
    gate=(delta['valid']['delta']>=.02 and delta['valid']['CI95'][0]>0 and delta['modes8']['delta']>=.15
        and delta['rare8']['delta']>=-.02 and delta['oracle']['delta']>=-.01 and delta['brier']['delta']<=.01)
    initial=read(RUN/'safety_mean/evaluation_matched_q_v1/rows.json')
    result=dict(metrics=metrics,training=training,configs=configs,contrast=delta,
        versus_parent={a:paired(initial,rows[a]) for a in rows},ordinary_linear_gate=bool(gate),
        actual_initial_input_target_exposure_equal=True,parent_replay=replayed,
        scope='One generator seed, standard loss controls, same extra1200 updates and fixed complete new-domain scorer; not same-pool scoring improvement or novelty.')
    write(out/'RESULTS.json',result);print(dict(contrast=delta,gate=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output');p.add_argument('--replay-only',action='store_true');a=p.parse_args()
    if a.replay_only:print(replay(),flush=True)
    else:main(a.output)
