"""Actual deployment replay and unchanged candidates across scorer repair."""
import argparse
import copy
import numpy as np
from scripts.run_observed_probability import ROOT,torch_setup,write,sha
from scripts.paired_modes_data import RUN as OLD_RUN
from scripts.research_v3_frequency import fixed_path_scores


def main(output):
    torch=torch_setup()
    from routeset.observed_probability import load_scored_planner
    bundle=OLD_RUN/'reliability/R1_seed0/deployment_seed0/planner.pt'
    planner=load_scored_planner(bundle,'cuda')
    inputs=torch.load(bundle.parent/'example_observed_inputs.pt',map_location='cuda',weights_only=False)
    with torch.inference_mode():
        reference=planner(**inputs,return_k=4)
        q=fixed_path_scores(planner,reference['paths'],reference['events'],inputs)
        torch.testing.assert_close(q,reference['q'],rtol=0,atol=0)
        initial={k:v.clone() for k,v in planner.state_dict().items()}
        other=copy.deepcopy(planner.generator)
        state=torch.load(ROOT/'runs/research_v3_v1/frequency_empirical_uniform/last.pt',map_location='cpu',weights_only=False)
        other.load_state_dict(state['model']);new_paths,events,_=other(**inputs)
        assert not torch.equal(new_paths,reference['paths'])
        repeat=fixed_path_scores(planner,reference['paths'],reference['events'],inputs)
        torch.testing.assert_close(repeat,q,rtol=0,atol=0)
        for k,v in planner.state_dict().items():torch.testing.assert_close(initial[k],v,rtol=0,atol=0)
    write(output,dict(exact_deployment_q=True,unchanged_q_function_after_generator_replacement=True,
        original_bundle_sha256=sha(bundle),external_generator_sha256=sha(ROOT/'runs/research_v3_v1/frequency_empirical_uniform/last.pt')))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
