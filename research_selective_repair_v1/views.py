"""Freeze exact inference settings into a uniquely hashed decoder checkpoint."""
import argparse
import torch
from research_selective_repair_v1.core import *
from routeset.train_v2 import atomic_checkpoint

def main(name,parent,prototype,threshold=.5,scale=1.):
    out=RUN/name;out.mkdir(parents=True,exist_ok=False)
    ck=torch.load(parent,map_location='cpu',weights_only=False)
    ck['view']=dict(parent_sha256=sha(parent),prototype_sha256=sha(prototype),threshold=threshold,scale=scale)
    atomic_checkpoint(out/'last.pt',ck);write(out/'VIEW.json',dict(**ck['view'],decoder_sha256=sha(out/'last.pt'),weight_tensors_unchanged=True))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--parent',required=True,type=Path);p.add_argument('--prototype',required=True,type=Path);p.add_argument('--threshold',type=float,default=.5);p.add_argument('--scale',type=float,default=1.);main(**vars(p.parse_args()))
