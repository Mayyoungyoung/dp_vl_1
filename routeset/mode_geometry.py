"""Observation-only semantic mode tokens and a shared whole-route decoder.

Mode identities are two-row operational passage words, not homotopy classes.
Only training/evaluation helpers know verified labels or scene correspondences.
"""
import itertools
import torch
from torch import nn
from torch.nn import functional as F
from routeset.observed_probability import ProbabilisticGeometryRouteHead

VOCAB = ['|'.join(x) for x in itertools.product(('gap0','gap1','gap2','over'), repeat=2)]


class ModeGeometryHead(ProbabilisticGeometryRouteHead):
    def __init__(self, *args, conditional=True, decoder_communication=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.conditional = conditional
        self.decoder_communication=decoder_communication
        if not decoder_communication:
            for block in self.head.blocks:
                block.communicate=False
                block.attention.requires_grad_(False)
                block.attention_norm.requires_grad_(False)
        width = self.head.queries.shape[-1]
        self.mode_predictor = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width),
                                            nn.SiLU(), nn.Linear(width, len(VOCAB)))
        self.mode_embedding = nn.Embedding(len(VOCAB), width)
        self.variant_embedding = nn.Embedding(8, width)
        nn.init.normal_(self.mode_embedding.weight, std=.1)
        nn.init.zeros_(self.variant_embedding.weight)
        # Variant zero is the canonical draw; repeated modes can vary geometrically.
        with torch.no_grad(): self.variant_embedding.weight[1:].normal_(std=.02)

    def freeze_encoders(self):
        for module in (self.geometry, self.head.feature_encoder, self.head.state_encoder, self.mode_mass):
            module.requires_grad_(False)
        self.head.queries.requires_grad_(not self.conditional)
        self.mode_embedding.requires_grad_(self.conditional)
        self.variant_embedding.requires_grad_(self.conditional)

    def encode(self, features, current, world_xyz, rgb, uv, depth, valid_mask):
        geometry = self.geometry(features,current,world_xyz,rgb,uv,depth,valid_mask)
        context = self.head.feature_encoder(features)+self.head.state_encoder(current)+geometry['context']
        return context, geometry['anchor_xyz']

    def decode(self, context, anchor, current, mode_ids=None, sampling='adaptive'):
        logits = self.mode_predictor(context)
        if mode_ids is None:
            # Scores allocate a finite proposal budget, NOT calibrated existence.
            if sampling == 'adaptive':
                allocations=[]
                for scores in logits:
                    order=torch.argsort(scores,descending=True,stable=True)
                    active=order[scores[order]>=0][:8]
                    if not len(active):active=order[:1]
                    allocations.append(active[torch.arange(8,device=scores.device)%len(active)])
                mode_ids=torch.stack(allocations)
            elif sampling == 'balanced':
                mode_ids = torch.argsort(logits,dim=-1,descending=True,stable=True)[:,:8]
            elif sampling == 'ordinary':
                mode_ids = torch.multinomial(logits.softmax(-1),8,replacement=True)
            else: raise ValueError(sampling)
        if mode_ids.shape != (len(context),8): raise ValueError('Exactly eight mode queries required')
        if self.conditional:
            occurrence = torch.stack([(mode_ids[:,:i] == mode_ids[:,i:i+1]).sum(-1) for i in range(8)],1)
            tokens = context[:,None]+self.mode_embedding(mode_ids)+self.variant_embedding(occurrence)
        else:
            tokens = context[:,None]+self.head.queries[None]
        for block in self.head.blocks: tokens = block(tokens,context)
        prediction = self.head.output(tokens).reshape(len(context),8,self.head.horizon-1,4)
        t = torch.linspace(0,1,self.head.horizon,device=context.device,dtype=context.dtype)[1:]
        line = current[:,None,None,:3]+t[None,None,:,None]*(anchor-current[:,:3])[:,None,None]
        xyz = torch.cat((current[:,None,None,:3].expand(-1,8,1,-1),
            line[:,:,:-1]+prediction[:,:,:-1,:3],
            anchor[:,None,None]+self.endpoint_residual_bound*prediction[:,:,-1:,:3].tanh()),2)
        event = torch.cat((current[:,None,None,7].expand(-1,8,1).clamp(0,1),prediction[...,3].sigmoid()),2)
        return xyz,event,dict(mode_logits=logits,mode_ids=mode_ids)

    def forward(self, features, current, world_xyz, rgb, uv, depth, valid_mask, k=None,
                sampling='adaptive'):
        if k not in (None,8): raise ValueError('Fixed candidate budget')
        context,anchor = self.encode(features,current,world_xyz,rgb,uv,depth,valid_mask)
        return self.decode(context,anchor,current,sampling=sampling)


def mode_loss(logits, evidence):
    """Positive allocation likelihood plus BCE on *known* status only.

    Unknown=-1 never becomes a negative existence label. The normalized
    positive likelihood learns relative proposal allocation from partial data;
    it cannot certify absence for low ranked/unobserved words.
    """
    positive = (evidence == 1).to(logits.dtype)
    allocation = -(positive/positive.sum(-1,keepdim=True).clamp_min(1)*logits.log_softmax(-1)).sum(-1).mean()
    known = evidence >= 0
    binary = F.binary_cross_entropy_with_logits(logits,evidence.clamp_min(0),reduction='none')
    return allocation+(binary*known).sum()/known.sum().clamp_min(1)


def pair_displacement_loss(paths, target, mask):
    """Adjacent source/destination pairs; target displacement is NOT zero.

    Only witnessed identical words are aligned. Unknown/deleted words are masked.
    Both actual geometry outputs receive gradients, even for unchanged labels.
    """
    error = ((paths[1::2,:,1:]-paths[0::2,:,1:])-(target[1::2,:,1:]-target[0::2,:,1:])).square().mean((-1,-2))
    return (error*mask).sum()/mask.sum().clamp_min(1)


def shared_allocation_loss(logits, evidence):
    """Symmetric KL over witnessed common words only; no unknown absence labels.

    This ordinary correspondence term is NOT a novelty claim. It connects the
    proposal module to selective pair supervision; displacement still trains XYZ.
    """
    common=(evidence[::2]==1)&(evidence[1::2]==1)
    if not common.any(-1).all():raise ValueError('Pair needs a witnessed common mode')
    la=logits[::2].masked_fill(~common,-1e9).log_softmax(-1)
    lb=logits[1::2].masked_fill(~common,-1e9).log_softmax(-1)
    return (.5*(la.exp()-lb.exp())*(la-lb)*common).sum(-1).mean()


class FixedScoredModePlanner(nn.Module):
    """Eight observation-only routes -> four, preserving the complete q encoder."""
    def __init__(self, generator, frozen_scored_planner):
        super().__init__()
        self.generator=generator
        self.fixed_q=frozen_scored_planner
        self.fixed_q.requires_grad_(False)

    def forward(self, features,current,world_xyz,rgb,uv,depth,valid_mask,return_k=4,sampling='adaptive'):
        from routeset.observed_probability import route_observation_features,select_route_indices
        inp=dict(features=features,current=current,world_xyz=world_xyz,rgb=rgb,uv=uv,depth=depth,valid_mask=valid_mask)
        paths,events,details=self.generator(**inp,sampling=sampling)
        qmodel=self.fixed_q
        g=qmodel.generator.geometry(**inp,return_point_features=True)
        context=qmodel.generator.head.feature_encoder(features)+qmodel.generator.head.state_encoder(current)+g['context']
        nodes,ctx=route_observation_features(paths,events,current,world_xyz,rgb,valid_mask,g['point_features'],context,g['anchor_xyz'])
        logits=qmodel.scorer((nodes-qmodel.nodes_mean)/qmodel.nodes_std,(ctx-qmodel.context_mean)/qmodel.context_std)
        q=(logits/qmodel.temperature).sigmoid();selected=select_route_indices(paths,q,return_k)
        rows=torch.arange(len(paths),device=paths.device)[:,None]
        return dict(paths=paths,events=events,q=q,selected_indices=selected,selected_paths=paths[rows,selected],
            selected_events=events[rows,selected],selected_q=q[rows,selected],mode_ids=details['mode_ids'],
            internal_candidates=8,returned_candidates=return_k)


def load_fixed_scored_mode_planner(checkpoint,scorer_bundle,device='cpu'):
    """Load locally trusted artifacts; no parent/source-route input at inference."""
    from routeset.observed_probability import load_scored_planner
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
    generator=ModeGeometryHead(**saved['config']['head_options'],conditional=saved['config']['conditional'],
        decoder_communication=saved['config'].get('decoder_communication',True))
    generator.load_state_dict(saved['model'],strict=True)
    planner=FixedScoredModePlanner(generator,load_scored_planner(scorer_bundle))
    return planner.to(device).eval()
