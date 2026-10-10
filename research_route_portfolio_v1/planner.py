"""Requester-supplied passage preferences, with cached observation encodings.

Preferences constrain proposals; actual path compliance is an evaluated outcome.
No native executor or truth geometry is accepted by this interface.
"""
import torch
from routeset.observed_probability import route_observation_features, select_route_indices


def allocate_preference(logits, allowed):
    if allowed.shape != logits.shape or allowed.dtype != torch.bool:
        raise ValueError('Use one boolean preference mask per observation and mode')
    if not allowed.any(-1).all():
        raise ValueError('Empty preference set')
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite allocation scores')
    allocation=[]
    for scores,mask in zip(logits,allowed):
        order=torch.argsort(scores.masked_fill(~mask,-float('inf')),descending=True,stable=True)
        legal=order[mask[order]]
        active=legal[scores[legal]>=0][:8]
        if len(active)==0:active=legal[:1]
        allocation.append(active[torch.arange(8,device=logits.device)%len(active)])
    return torch.stack(allocation)


class PreferenceRoutePlanner:
    def __init__(self, fixed_planner):
        if not fixed_planner.generator.conditional:
            raise ValueError('Preferences require a semantic conditional decoder')
        self.planner=fixed_planner

    @torch.inference_mode()
    def observe(self, **inp):
        generator=self.planner.generator;q=self.planner.fixed_q
        context,anchor=generator.encode(**inp)
        geo=q.generator.geometry(**inp,return_point_features=True)
        qc=q.generator.head.feature_encoder(inp['features'])+q.generator.head.state_encoder(inp['current'])+geo['context']
        return dict(inputs=inp,context=context,anchor=anchor,q_geometry=geo,q_context=qc,
                    allocation_logits=generator.mode_predictor(context))

    @torch.inference_mode()
    def propose(self, observed, allowed):
        inp=observed['inputs'];qmodel=self.planner.fixed_q
        modes=allocate_preference(observed['allocation_logits'],allowed)
        paths,events,_=self.planner.generator.decode(observed['context'],observed['anchor'],inp['current'],modes)
        geo=observed['q_geometry']
        nodes,context=route_observation_features(paths,events,inp['current'],inp['world_xyz'],inp['rgb'],
            inp['valid_mask'],geo['point_features'],observed['q_context'],geo['anchor_xyz'])
        logits=qmodel.scorer((nodes-qmodel.nodes_mean)/qmodel.nodes_std,(context-qmodel.context_mean)/qmodel.context_std)
        q=(logits/qmodel.temperature).sigmoid()
        return dict(paths=paths,events=events,q=q,mode_ids=modes,
                    selected_indices=select_route_indices(paths,q,4),internal_candidates=8,returned_candidates=4)
