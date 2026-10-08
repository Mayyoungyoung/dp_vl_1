"""Attribute full-set matching to coverage versus within-mode references."""
from scripts.research_v3_frequency import RUN
from scripts.research_v3_analyze_frequency import paired
from scripts.run_observed_probability import read,write,sha,torch_setup


def main():
    torch=torch_setup();out=RUN/'canonical_analysis_v1';out.mkdir(exist_ok=False)
    names=['frequency_set_matching','frequency_set_sampled','frequency_set_canonical']
    training={};metrics={};rows={};configs={};rngs={}
    for name in names:
        root=RUN/name;ev=root/'evaluation_fixed_q_v2'
        training[name]=read(root/'summary.json');configs[name]=read(root/'config.json')
        metrics[name]=read(ev/'metrics.json');rows[name]=read(ev/'rows.json')
        assert training[name]['steps']==1200
        assert metrics[name]['generator_sha256']==training[name]['last_sha256']==sha(root/'last.pt')
        assert metrics[name]['pool_sha256']==sha(ev/'pool.npz')
        saved=torch.load(root/'last.pt',map_location='cpu',weights_only=False);rngs[name]=saved['loss_rng'];del saved
    full,sub,canonical=(training[n] for n in names)
    for key in ('sampler','initial_sha256'):assert full[key]==canonical[key]==sub[key]
    assert rngs[names[0]]==rngs[names[2]]
    assert full['unique_routes']==5*canonical['unique_routes']
    assert full['mode_exposure']=={k:5*v for k,v in canonical['mode_exposure'].items()}
    for key in ('initial_checkpoint_sha256','support_sha256','input_ids_sha256'):
        assert len({configs[n][key] for n in names})==1
    assert len({metrics[n]['scorer_bundle_sha256'] for n in names})==1
    delta=paired(rows[names[2]],rows[names[0]])
    gate=delta['rare8']['delta']>=.02 and delta['rare8']['CI95'][0]>0 and delta['valid']['delta']>=-.02
    result=dict(metrics=metrics,training=training,configs=configs,full_minus_canonical=delta,
        canonical_minus_sampled=paired(rows[names[1]],rows[names[2]]),ordinary_augmentation_gate=bool(gate),
        actual_initial_sampler_equal=True,matching_group_rng_final_equal=True,
        scope='One generator seed, same input/update budget, target slots differ by5x. Ordinary augmentation attribution only; original fixed complete q, not paired-domain q.')
    write(out/'RESULTS.json',result);print(dict(full_minus_canonical=delta,gate=gate),flush=True)


if __name__=='__main__':main()
