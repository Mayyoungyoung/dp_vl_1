"""Matched ordinary optimizer continuation; reuse the sealed fresh control."""
import argparse
from scripts.research_v3_frequency import RUN
from scripts.research_v3_analyze_frequency import paired
from scripts.run_observed_probability import read,write,sha,torch_setup


def main():
    torch=torch_setup();out=RUN/'optimizer_analysis_v1';out.mkdir(exist_ok=False)
    training={};metrics={};rows={};configs={};steps={}
    for name in ('margin_mean','optimizer_restored'):
        root=RUN/name;ev=root/'evaluation_matched_q_v1'
        training[name]=read(root/'summary.json');configs[name]=read(root/'config.json')
        metrics[name]=read(ev/'metrics.json');rows[name]=read(ev/'rows.json')
        assert training[name]['steps']==1200
        assert metrics[name]['generator_sha256']==training[name]['last_sha256']==sha(root/'last.pt')
        assert metrics[name]['pool_sha256']==sha(ev/'pool.npz')
        assert configs[name]['initial_checkpoint_sha256']==sha(RUN/'safety_mean/last.pt')
        assert metrics[name]['scorer_bundle_sha256']=='2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e'
        checkpoint=torch.load(root/'last.pt',map_location='cpu',weights_only=False)
        steps[name]=sorted({int(s['step']) for s in checkpoint['optimizer']['state'].values()})
        assert all(g['lr']==.0003 and g['weight_decay']==.0001 for g in checkpoint['optimizer']['param_groups'])
        del checkpoint
    assert steps==dict(margin_mean=[1200],optimizer_restored=[2400])
    a,b=(training[k] for k in ('margin_mean','optimizer_restored'))
    for key in ('sampler','initial_sha256','mode_exposure','unique_routes'):assert a[key]==b[key]
    delta=paired(rows['margin_mean'],rows['optimizer_restored'])
    gate=(delta['valid']['delta']>=.02 and delta['valid']['CI95'][0]>0 and delta['modes8']['delta']>=.15
        and delta['rare8']['delta']>=-.02 and delta['oracle']['delta']>=-.01 and delta['brier']['delta']<=.01)
    parent=read(RUN/'safety_mean/evaluation_matched_q_v1/rows.json')
    result=dict(metrics=metrics,training=training,configs=configs,contrast=delta,optimizer_final_steps=steps,
        versus_parent=paired(parent,rows['optimizer_restored']),ordinary_optimizer_gate=bool(gate),
        actual_initial_input_target_exposure_equal=True,
        scope='Single generator seed, reused sealed fresh control, only optimizer moments and time index differ. No novelty or final-test claim.')
    write(out/'RESULTS.json',result);print(dict(contrast=delta,gate=gate),flush=True)


if __name__=='__main__':main()
