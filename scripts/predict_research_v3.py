"""Observed-only V3 route generator with an independently frozen full scorer."""
import argparse
import copy
from pathlib import Path
from scripts.run_observed_probability import read,write,sha,torch_setup
from scripts.observation_cache_qwen import read_manifest,REVISION
from scripts.evaluate_paired_modes import inputs_for
from scripts.research_v3_frequency import fixed_path_scores


def predict(bundle,checkpoint,manifest,identifier,observation,cache,output,k):
    torch=torch_setup()
    from routeset.observed_probability import load_scored_planner,select_route_indices
    manifest,cache,output=map(Path,(manifest,cache,output))
    if output.exists():raise FileExistsError('Use a fresh prediction output')
    if not 1<=k<=8:raise ValueError('Returned K must be between1 and8; internal budget is always8')
    cc=read(cache/'cache_config.json')
    assert cc['manifest_sha256']==sha(manifest) and cc['revision']==REVISION
    assert cc['model_trainable_parameter_count']==0
    row=next(r for r in read_manifest(manifest) if r['id']==identifier)
    if row['split']=='TEST_LOCKED':raise ValueError('No reserved TEST_LOCKED inference in development CLI')
    hashes={};inp=inputs_for(row,{'observation':observation},cache,torch,hashes)
    scorer=load_scored_planner(bundle,'cuda');generator=copy.deepcopy(scorer.generator)
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)
    generator.load_state_dict(state['model']);generator.eval();generator.requires_grad_(False)
    with torch.inference_mode():
        paths,events,_=generator(**inp);q=fixed_path_scores(scorer,paths,events,inp)
        selected=select_route_indices(paths,q,k)
    result=dict(paths=paths[0].cpu().tolist(),events=events[0].cpu().tolist(),q=q[0].cpu().tolist(),
        selected_indices=selected[0].cpu().tolist(),selected_paths=paths[0,selected[0]].cpu().tolist(),selected_q=q[0,selected[0]].cpu().tolist(),
        id=identifier,instruction=row['instruction'],internal_candidates=8,returned_candidates=k,
        generator_checkpoint_sha256=sha(checkpoint),complete_scorer_bundle_sha256=sha(bundle),input_sha256=hashes,
        scope='Learned probability of registered high-level validity; independent sigmoid routes, not robot execution success')
    write(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--manifest',required=True);p.add_argument('--id',required=True);p.add_argument('--observation',required=True)
    p.add_argument('--qwen-cache',required=True);p.add_argument('--output',required=True);p.add_argument('--k',type=int,default=4)
    a=p.parse_args();predict(a.bundle,a.checkpoint,a.manifest,a.id,a.observation,a.qwen_cache,a.output,a.k)
