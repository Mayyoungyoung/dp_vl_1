"""Prespecified mean-versus-path-bottleneck continuation comparison."""
import argparse
from scripts.research_v3_frequency import RUN
from scripts.research_v3_analyze_frequency import paired
from scripts.run_observed_probability import read,write,sha


def main(output):
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    metrics={};rows={};training={};configs={}
    for arm in ('mean','worst'):
        root=RUN/('safety_'+arm);training[arm]=read(root/'summary.json');configs[arm]=read(root/'config.json')
        ev=root/'evaluation_fixed_q_v2';metrics[arm]=read(ev/'metrics.json');rows[arm]=read(ev/'rows.json')
        assert metrics[arm]['generator_sha256']==training[arm]['last_sha256']==sha(root/'last.pt')
        assert metrics[arm]['pool_sha256']==sha(ev/'pool.npz')
    a,b=(training[k] for k in ('mean','worst'))
    assert a['sampler']==b['sampler'] and a['mode_exposure']==b['mode_exposure'] and a['unique_routes']==b['unique_routes']
    assert configs['mean']['initial_checkpoint_sha256']==configs['worst']['initial_checkpoint_sha256']
    assert metrics['mean']['scorer_bundle_sha256']==metrics['worst']['scorer_bundle_sha256']
    delta=paired(rows['mean'],rows['worst'])
    gate=(delta['valid']['delta']>=.02 and delta['valid']['CI95'][0]>0
        and delta['modes8']['delta']>=.15 and delta['rare8']['delta']>=-.02)
    initial=read(RUN/'frequency_set_matching/evaluation_fixed_q_v2/rows.json')
    write(out/'RESULTS.json',dict(metrics=metrics,training=training,configs=configs,contrast=delta,
        versus_parent={a:paired(initial,rows[a]) for a in rows},ordinary_safety_gate=gate,
        actual_input_and_target_exposure_equal=True,
        scope='single-seed development ordinary safety objective; same initialization, data and1200-step budget; no novelty claim'))
    print(dict(contrast=delta,gate=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
