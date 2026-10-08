"""Registered anchor intervention versus its exact unchanged parent model."""
import argparse
from scripts.research_v3_frequency import RUN
from scripts.research_v3_analyze_frequency import paired
from scripts.run_observed_probability import read,write,sha


def main(output):
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    root=RUN/'safety_mean';evs={'peak':'evaluation_fixed_q_v2','mass':'evaluation_anchor_mass_v1'}
    metrics={k:read(root/v/'metrics.json') for k,v in evs.items()}
    rows={k:read(root/v/'rows.json') for k,v in evs.items()}
    for k,v in evs.items():
        assert metrics[k]['generator_sha256']==sha(root/'last.pt')
        assert metrics[k]['pool_sha256']==sha(root/v/'pool.npz')
    assert metrics['peak']['input_sha256']==metrics['mass']['input_sha256']
    assert metrics['peak']['scorer_bundle_sha256']==metrics['mass']['scorer_bundle_sha256']
    delta=paired(rows['peak'],rows['mass'])
    gate=(delta['valid']['delta']>=.01 and delta['valid']['CI95'][0]>0
        and delta['rare8']['delta']>=-.01 and delta['oracle']['delta']>=0)
    result=dict(metrics=metrics,contrast=delta,gate=gate,
        scope='Ordinary fixed inference aggregation on same seed0 weights, complete scorer and all inputs; no retraining or novel-method claim')
    write(out/'RESULTS.json',result);print(dict(contrast=delta,gate=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
