"""Opt-in observed API; no oracle fields and no historical default changes."""
import torch
from torch import nn
from research_feasible_space_v1.model import load_model
from research_realized_coverage_v1.deployment import RealizedCoveragePlanner,digest
from research_realized_coverage_v1.allocator import SuccessHead
from routeset.observed_probability import load_scored_planner

class FeasibleSpacePlanner(RealizedCoveragePlanner):
    def forward(self,features,current,world_xyz,rgb,uv,depth,valid_mask,return_k=4):
        out=super().forward(features,current,world_xyz,rgb,uv,depth,valid_mask,return_k)
        out['corridor_certificate']='predicted containment only; true obstacle/visibility feasibility unknown'
        return out

def load_planner(generator_checkpoint,scorer_bundle,proposal_checkpoint=None,device='cpu'):
    generator,_=load_model(generator_checkpoint,device);scored=load_scored_planner(scorer_bundle,device)
    if digest(scorer_bundle)!='2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e':raise ValueError('Frozen complete scorer required')
    proposal=None
    if proposal_checkpoint:
        ck=torch.load(proposal_checkpoint,map_location='cpu',weights_only=False)
        if ck['settings']['kind']!='success' or ck['settings']['generator_sha256']!=digest(generator_checkpoint):raise ValueError('Success head must match this exact decoder')
        proposal=SuccessHead();proposal.load_state_dict(ck['model'])
    return FeasibleSpacePlanner(generator,scored,proposal,'success' if proposal else None).to(device).eval().requires_grad_(False)
