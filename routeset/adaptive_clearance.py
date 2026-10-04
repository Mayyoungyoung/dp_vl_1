"""TRAIN-only projected dual state, absent from generator/deployment."""
import torch
from .segment_clearance import path_segment_clearances


class SegmentDual:
    def __init__(self, requests, modes, segments, expected_visits, geometric_scale, device, quadratic=160.):
        self.weights = torch.zeros(requests, modes, segments, device=device)
        self.quadratic = float(quadratic)
        # Over the planned number of exposures, a persistent violation adds
        # one baseline-sized derivative. No development-set lambda search.
        self.rate = 2*self.quadratic/expected_visits
        self.cap = 2*self.quadratic*float(geometric_scale)

    def loss(self, paths, centers, halves, ids):
        signed = .02-path_segment_clearances(paths, centers, halves)
        violation = signed.clamp_min(0)
        weight = self.weights[ids].detach().clone()
        return (weight*violation+self.quadratic*violation.square()).mean(), signed

    @torch.no_grad()
    def update(self, ids, signed):
        ids = torch.as_tensor(ids, device=self.weights.device, dtype=torch.long)
        unique, inverse, counts = torch.unique(ids, return_inverse=True, return_counts=True)
        summed = torch.zeros(len(unique), *signed.shape[1:], device=signed.device)
        summed.index_add_(0, inverse, signed.detach())
        average = summed/counts[:, None, None]
        self.weights[unique] = (self.weights[unique]+self.rate*average).clamp(0, self.cap)

    def state_dict(self):
        return dict(weights=self.weights, rate=self.rate, cap=self.cap, quadratic=self.quadratic)

    def load_state_dict(self, state):
        self.weights.copy_(state['weights'])
        for key in ('rate','cap','quadratic'): setattr(self, key, state[key])
