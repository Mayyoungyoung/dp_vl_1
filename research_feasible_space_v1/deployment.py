"""Opt-in observed API; no oracle fields and no historical default changes."""
import torch
from torch import nn
from research_feasible_space_v1.model import load_model
from research_realized_coverage_v1.deployment import RealizedCoveragePlanner,digest
from research_realized_coverage_v1.allocator import SuccessHead
from routeset.observed_probability import load_scored_planner
from routeset.observed_probability import route_observation_features,select_route_indices
from research_realized_coverage_v1.allocator import occurrence

class FeasibleSpacePlanner(RealizedCoveragePlanner):
    def forward(self,features,current,world_xyz,rgb,uv,depth,valid_mask,return_k=4):
        if return_k!=4:raise ValueError('Return four from exactly eight')
        inp=dict(features=features,current=current,world_xyz=world_xyz,rgb=rgb,uv=uv,depth=depth,valid_mask=valid_mask)
        context,anchor=self.generator.encode(**inp);logits=self.generator.mode_predictor(context);modes=[]
        for l in logits:
            order=torch.argsort(l,descending=True,stable=True);active=order[l[order]>=0][:8]
            if not len(active):active=order[:1]
            modes.append(active[torch.arange(8,device=l.device)%len(active)])
        m=torch.stack(modes);v=occurrence(m)
        if self.proposal:
            from research_realized_coverage_v1.allocator import allocate
            m,v,cost=allocate('success',self.proposal,context,m,v)
        else:cost=dict(replacements=0,query_sets_evaluated=0)
        paths,events,info=self.generator.decode(context,anchor,current,m,variant_ids=v)
        geometry=self.scored.generator.geometry(**inp,return_point_features=True)
        qc=self.scored.generator.head.feature_encoder(features)+self.scored.generator.head.state_encoder(current)+geometry['context']
        nodes,ctx=route_observation_features(paths,events,current,world_xyz,rgb,valid_mask,geometry['point_features'],qc,geometry['anchor_xyz'])
        q=(self.scored.scorer((nodes-self.scored.nodes_mean)/self.scored.nodes_std,(ctx-self.scored.context_mean)/self.scored.context_std)/self.scored.temperature).sigmoid()
        chosen=select_route_indices(paths,q,4);rows=torch.arange(len(paths),device=paths.device)[:,None]
        return dict(paths=paths,events=events,q=q,mode_ids=m,variant_ids=v,selected_indices=chosen,
            selected_paths=paths[rows,chosen],selected_events=events[rows,chosen],selected_q=q[rows,chosen],proposal_cost=cost,
            corridor_centers=info['centers'],corridor_radii=info['radii'],
            corridor_certificate='predicted containment only; true obstacle/visibility feasibility unknown')

def load_planner(generator_checkpoint,scorer_bundle,proposal_checkpoint=None,device='cpu'):
    generator,_=load_model(generator_checkpoint,device);scored=load_scored_planner(scorer_bundle,device)
    if digest(scorer_bundle)!='2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e':raise ValueError('Frozen complete scorer required')
    proposal=None
    if proposal_checkpoint:
        ck=torch.load(proposal_checkpoint,map_location='cpu',weights_only=False)
        if ck['settings']['kind']!='success' or ck['settings']['generator_sha256']!=digest(generator_checkpoint):raise ValueError('Success head must match this exact decoder')
        proposal=SuccessHead();proposal.load_state_dict(ck['model'])
    return FeasibleSpacePlanner(generator,scored,proposal,'success' if proposal else None).to(device).eval().requires_grad_(False)
