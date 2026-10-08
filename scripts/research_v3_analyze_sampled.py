"""Evaluate the preregistered equal-target-budget ordinary control."""
import argparse
import numpy as np
from pathlib import Path
from scripts.run_observed_probability import read,write,sha
from scripts.research_v3_analyze_frequency import paired
from scripts.research_v3_frequency_integrity import replay_exposure,initial_rows
from scripts.research_v3_frequency import RUN


def main(output):
    out=RUN/output;out.mkdir(parents=True,exist_ok=False)
    arm='set_sampled';folder=RUN/('frequency_'+arm);ev=folder/'evaluation_fixed_q_v2'
    summary=read(folder/'summary.json');cfg=read(folder/'config.json');metrics=read(ev/'metrics.json');rows=read(ev/'rows.json')
    assert summary['sampler']==read(RUN/'frequency_balanced/summary.json')['sampler']
    assert summary['last_sha256']==metrics['generator_sha256']==sha(folder/'last.pt')
    assert metrics['pool_sha256']==sha(ev/'pool.npz')
    with np.load(RUN/'frequency_support_v1/support.npz') as z:support={k:z[k] for k in z.files}
    exposure=replay_exposure(support,cfg['policy'],arm)
    assert exposure['global_counts']==summary['mode_exposure']
    assert exposure['unique_routes']==summary['unique_routes']
    assert sum(summary['mode_exposure'].values())==307200
    write(out/'exposure.json',exposure)
    contrasts={a:paired(read(RUN/('frequency_'+a)/'evaluation_fixed_q_v2/rows.json'),rows) for a in ('balanced','set_matching')}
    contrasts['initial']=paired(initial_rows(RUN),rows)
    benefit=contrasts['balanced']['rare8'];valid=contrasts['balanced']['valid']
    gate=benefit['delta']>=.05 and benefit['CI95'][0]>0 and valid['delta']>=-.02
    write(out/'RESULTS.json',dict(metrics=metrics,training=summary,contrasts=contrasts,ordinary_control_gate=gate,
        same_initialization_and_inputs=True,reference_slots=307200,scope='single-seed development conventional control; no novelty or multi-seed claim'))
    print(dict(raw=metrics['raw'],rare8=metrics['rare_recall8'],contrasts=contrasts,ordinary_control_gate=gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
