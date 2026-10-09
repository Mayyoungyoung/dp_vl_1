"""Current-observation API: token allocation -> one8-route decode -> fixed4return."""
import hashlib
import torch
from torch import nn
from routeset.mode_geometry import ModeGeometryHead
from routeset.observed_probability import load_scored_planner,route_observation_features,select_route_indices
from research_realized_coverage_v1.allocator import make_allocator,allocate,occurrence

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

class RealizedCoveragePlanner(nn.Module):
    def __init__(self,generator,scored,proposal=None,kind=None,starter=None):
        super().__init__();self.generator=generator;self.scored=scored
        self.proposal=proposal;self.kind=kind;self.starter=starter
    def forward(self,features,current,world_xyz,rgb,uv,depth,valid_mask,return_k=4):
        if return_k!=4:raise ValueError('This protocol returns four from eight')
        inp=dict(features=features,current=current,world_xyz=world_xyz,rgb=rgb,uv=uv,depth=depth,valid_mask=valid_mask)
        context,anchor=self.generator.encode(**inp);logits=self.generator.mode_predictor(context);base=[]
        for l in logits:
            order=torch.argsort(l,descending=True,stable=True);active=order[l[order]>=0][:8]
            if not len(active):active=order[:1]
            base.append(active[torch.arange(8,device=l.device)%len(active)])
        m=torch.stack(base);v=occurrence(m)
        if self.starter:m,v,_=allocate('success',self.starter,context,m,v)
        if self.proposal:m,v,cost=allocate(self.kind,self.proposal,context,m,v)
        else:cost=dict(replacements=0,query_sets_evaluated=0)
        paths,events,_=self.generator.decode(context,anchor,current,m,variant_ids=v)
        geometry=self.scored.generator.geometry(**inp,return_point_features=True)
        qc=self.scored.generator.head.feature_encoder(features)+self.scored.generator.head.state_encoder(current)+geometry['context']
        nodes,ctx=route_observation_features(paths,events,current,world_xyz,rgb,valid_mask,geometry['point_features'],qc,geometry['anchor_xyz'])
        q=(self.scored.scorer((nodes-self.scored.nodes_mean)/self.scored.nodes_std,(ctx-self.scored.context_mean)/self.scored.context_std)/self.scored.temperature).sigmoid()
        chosen=select_route_indices(paths,q,4);rows=torch.arange(len(paths),device=paths.device)[:,None]
        return dict(paths=paths,events=events,q=q,mode_ids=m,variant_ids=v,selected_indices=chosen,
            selected_paths=paths[rows,chosen],selected_events=events[rows,chosen],selected_q=q[rows,chosen],proposal_cost=cost)

def load_planner(generator_checkpoint,scorer_bundle,proposal_checkpoint=None,starter_checkpoint=None,device='cpu'):
    ck=torch.load(generator_checkpoint,map_location='cpu',weights_only=False)
    generator=ModeGeometryHead(**ck['config']['head_options'],conditional=True)
    generator.load_state_dict(ck['model']);scored=load_scored_planner(scorer_bundle,device)
    assert digest(scorer_bundle)=='2d87cb3c92336e224f48ec7888abb5ffa5c648eeaca86102d62780588ed0f72e'
    def head(path):
        if path is None:return None,None
        saved=torch.load(path,map_location='cpu',weights_only=False)
        assert saved['settings']['generator_sha256']==digest(generator_checkpoint),'Stale generator feedback'
        kind=saved['settings']['kind'];model=make_allocator(kind);model.load_state_dict(saved['model']);return model,kind
    proposal,kind=head(proposal_checkpoint);starter,sk=head(starter_checkpoint)
    if starter is not None:assert sk=='success'
    model=RealizedCoveragePlanner(generator,scored,proposal,kind,starter).to(device).eval();model.requires_grad_(False)
    return model
