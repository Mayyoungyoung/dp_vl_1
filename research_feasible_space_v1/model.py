"""Observation-only learned boundaries + peer-aware interior generation."""
import copy
import torch
from torch import nn
from research_feasible_space_v1.mapping import construct,node_radii

class FeasibleSpaceHead(nn.Module):
    def __init__(self,base,kind='bounded',peer_boundaries=False):
        super().__init__();self.base=base;self.kind=kind;self.peer_boundaries=peer_boundaries
        self.base.requires_grad_(False)
        width=base.mode_embedding.embedding_dim
        self.corridor_blocks=copy.deepcopy(base.head.blocks);self.corridor_output=copy.deepcopy(base.head.output)
        self.relative_blocks=copy.deepcopy(base.head.blocks)
        self.corridor_feature=nn.Sequential(nn.Linear(24*4,width),nn.SiLU(),nn.Linear(width,width))
        self.relative_output=nn.Linear(width,24*3);nn.init.zeros_(self.relative_output.weight);nn.init.zeros_(self.relative_output.bias)
        self.radius_output=nn.Linear(width,23);nn.init.zeros_(self.radius_output.weight);nn.init.constant_(self.radius_output.bias,-1.)
        for module in (self.corridor_blocks,self.corridor_output,self.relative_blocks):module.requires_grad_(True)

    @property
    def mode_predictor(self):return self.base.mode_predictor
    def encode(self,**observed):return self.base.encode(**observed)

    def corridor(self,context,anchor,current,modes,variants):
        b,k=modes.shape
        token=context[:,None]+self.base.mode_embedding(modes)+self.base.variant_embedding(variants%2)
        if self.peer_boundaries:
            h=token
            for block in self.corridor_blocks:h=block(h,context)
        else:
            # Each corridor sees only its own mode/variant; attention length is one.
            h=token.reshape(b*k,1,-1);ctx=context[:,None].expand(-1,k,-1).reshape(b*k,-1)
            for block in self.corridor_blocks:h=block(h,ctx)
            h=h.reshape(b,k,-1)
        prediction=self.corridor_output(h).reshape(b,k,23,4)
        t=torch.linspace(0,1,24,device=context.device,dtype=context.dtype)[1:]
        line=current[:,None,None,:3]+t[None,None,:,None]*(anchor-current[:,:3])[:,None,None]
        centers=torch.cat((current[:,None,None,:3].expand(-1,k,1,-1),line[:,:,:-1]+prediction[:,:,:-1,:3],
            anchor[:,None,None]+self.base.endpoint_residual_bound*prediction[:,:,-1:,:3].tanh()),2)
        radii=.06*self.radius_output(h).sigmoid()
        events=torch.cat((current[:,None,None,7].expand(-1,k,1).clamp(0,1),prediction[...,3].sigmoid()),2)
        return centers,radii,events,token

    def decode(self,context,anchor,current,mode_ids,variant_ids=None,kind=None,reference_corridor=None):
        if mode_ids.shape!=(len(context),8):raise ValueError('Exactly eight candidates required')
        if variant_ids is None:variant_ids=torch.zeros_like(mode_ids)
        centers,radii,events,token=self.corridor(context,anchor,current,mode_ids,variant_ids)
        predicted_centers,predicted_radii=centers,radii
        if reference_corridor is not None:centers,radii=reference_corridor
        feature=torch.cat((centers-current[:,None,None,:3],node_radii(radii)[...,None]),-1).flatten(-2)
        token=token+self.corridor_feature(feature)
        for block in self.relative_blocks:token=block(token,context)
        z=self.relative_output(token).reshape(len(context),8,24,3)
        paths=construct(centers,radii,z,kind or self.kind)
        return paths,events,dict(mode_ids=mode_ids,centers=centers,radii=radii,parameters=z,
            predicted_centers=predicted_centers,predicted_radii=predicted_radii)

def load_model(path,device='cuda'):
    from routeset.mode_geometry import ModeGeometryHead
    ck=torch.load(path,map_location='cpu',weights_only=False)
    base=ModeGeometryHead(**ck['config']['head_options'],conditional=True)
    model=FeasibleSpaceHead(base,ck['settings']['arm'],ck['settings']['peer_boundaries'])
    model.load_state_dict(ck['model']);return model.to(device).eval(),ck
