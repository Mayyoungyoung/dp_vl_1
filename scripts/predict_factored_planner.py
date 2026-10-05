"""Observed inputs only: frozen Qwen feature, RGB-D, cameras, current state."""
import argparse
import json
from pathlib import Path
import numpy as np
from scripts.run_observed_probability import read,write,sha,torch_setup
from scripts.observation_cache_qwen import read_manifest,REVISION
from scripts.evaluate_paired_modes import inputs_for


def predict(bundle,manifest,identifier,observation,cache,output,k):
    torch=torch_setup()
    from routeset.factored_q import load_factored_planner
    manifest,cache,output=map(Path,(manifest,cache,output))
    if output.exists():raise FileExistsError('Use a fresh prediction output')
    cc=read(cache/'cache_config.json');assert cc['manifest_sha256']==sha(manifest)
    assert cc['revision']==REVISION and cc['model_trainable_parameter_count']==0
    rows=read_manifest(manifest);row=next(r for r in rows if r['id']==identifier)
    if row['split']=='TEST_LOCKED':raise ValueError('This research demo cannot open reserved TEST_LOCKED')
    inp=inputs_for(row,{'observation':observation},cache,torch,{})
    planner=load_factored_planner(bundle,'cuda')
    with torch.inference_mode():pred=planner(**inp,return_k=k)
    # The inherited internal mass head is untrained and intentionally not exposed.
    result={key:pred[key][0].cpu().tolist() for key in ('paths','events','q_task','q_feas','q','selected_indices','selected_paths','selected_q')}
    result.update(factorization='P(task|observation,path) * P(geometry|task,observation,path)', id=identifier,instruction=row['instruction'],internal_candidates=8,returned_candidates=k,
        bundle_sha256=sha(bundle),scope='Learned high-level path validity probability, not whole-robot execution probability')
    write(output,result)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--manifest',required=True)
    p.add_argument('--id',required=True);p.add_argument('--observation',required=True);p.add_argument('--qwen-cache',required=True)
    p.add_argument('--output',required=True);p.add_argument('--k',type=int,default=4)
    a=p.parse_args();predict(a.bundle,a.manifest,a.id,a.observation,a.qwen_cache,a.output,a.k)
