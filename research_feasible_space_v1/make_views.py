"""Bind each postprocessed decoder to its own hash before fresh success feedback."""
import argparse,json
import torch
from research_feasible_space_v1.prepare import RUN
from scripts.run_observed_probability import sha,write,torch_setup
from routeset.train_v2 import atomic_checkpoint

def main(name,parent,kind):
    torch_setup();out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(parent,map_location='cpu',weights_only=False);ck['settings']=dict(ck['settings'],arm=kind,view_parent_sha256=sha(parent),view_kind=kind)
    # No parameter/optimizer changes or additional route training; only mapping differs.
    atomic_checkpoint(out/'last.pt',ck)
    report=dict(parent_sha256=sha(parent),decoder_sha256=sha(out/'last.pt'),kind=kind,weight_tensors_unchanged=True)
    write(out/'VIEW.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--parent',type=type(RUN),required=True);p.add_argument('--kind',choices=['center','projection'],required=True);main(**vars(p.parse_args()))
