"""Actual stochastic pool replay and all-allowed preference equivalence."""
import json,os
import numpy as np
from scripts.run_observed_probability import ROOT,SOURCE,read,write,sha,lines,torch_setup
from research_route_portfolio_v1.evaluate import RUN,POLICY


def main():
    torch=torch_setup()
    from routeset.mode_geometry import load_fixed_scored_mode_planner
    from research_route_portfolio_v1.planner import PreferenceRoutePlanner
    from scripts.mode_geometry_experiment import Q
    from scripts.evaluate_paired_modes import inputs_for
    out=RUN/'REPLAY_V1';out.mkdir(parents=True,exist_ok=False)
    checkpoint=ROOT/'runs/mode_geometry_v1/canonical_C_seed0/last.pt'
    planner=load_fixed_scored_mode_planner(checkpoint,Q,'cuda').requires_grad_(False)
    spec=read(POLICY)['datasets']['shift'];data=ROOT/spec['relative_path']
    rows=lines(data/'export/observations.jsonl');labels={r['id']:r for r in lines(data/'export/supervision.jsonl')}
    file=RUN/'shift_C_seed0_ordinary_71239/pool.npz'
    assert sha(file)==read(file.parent/'PREDICTION_SEAL.json')['predictions_sha256']
    with np.load(file) as z:pool={k:z[k] for k in z.files}
    torch.manual_seed(71239)
    before=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
    hashes={}
    for i,row in enumerate(rows):
        inp=inputs_for(row,labels[row['id']],data/'export/qwen_cache',torch,hashes)
        with torch.inference_mode():result=planner(**inp,sampling='ordinary')
        for key in ('paths','events','q','mode_ids','selected_indices'):
            np.testing.assert_array_equal(result[key][0].cpu().numpy(),pool[key][i])
    after=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all())
    torch.save(dict(before=before,after=after,inference_seed=71239,requests=len(rows),
                    generator_sha256=sha(checkpoint),pool_sha256=sha(file)),out/'sampling_rng.pt')
    with np.load(RUN/'shift_C_seed0/pool.npz') as z:base={k:z[k] for k in z.files}
    pref=PreferenceRoutePlanner(planner);maxq=0.
    for i,row in enumerate(rows[:32]):
        inp=inputs_for(row,labels[row['id']],data/'export/qwen_cache',torch,hashes)
        observed=pref.observe(**inp)
        result=pref.propose(observed,torch.ones_like(observed['allocation_logits'],dtype=torch.bool))
        for key in ('paths','events','mode_ids','selected_indices'):
            np.testing.assert_array_equal(result[key][0].cpu().numpy(),base[key][i])
        err=float(np.max(np.abs(result['q'][0].cpu().numpy()-base['q'][i])))
        maxq=max(maxq,err)
        np.testing.assert_allclose(result['q'][0].cpu().numpy(),base['q'][i],atol=1e-6,rtol=1e-6)
    report=dict(source_commit=os.environ.get('CODE_COMMIT'),source_sha256=sha(SOURCE/'research_route_portfolio_v1/replay.py'),
        stochastic_replay_requests=336,stochastic_paths_events_q_modes_indices_exact=True,
        all_allowed_preference_requests=32,preference_paths_events_modes_indices_exact=True,
        preference_q_max_abs_error=maxq,rng_state_sha256=sha(out/'sampling_rng.pt'),
        historical_checkpoint_sha256=sha(checkpoint),scorer_sha256=sha(Q),locked_access=False)
    write(out/'RESULTS.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':main()
