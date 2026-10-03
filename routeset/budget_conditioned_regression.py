"""Ordinary K-conditioned set regression; no claimed new mechanism."""
import math

import torch

from .models import SetRegressor


BUDGETS = (1, 2, 4, 8)


class BudgetConditionedRegressor(SetRegressor):
    """A learned condition embedding reads geometry plus explicit log2(K)/3.

    Parent SetRegressor constructs only the requested K tokens. Eight query
    parameters exist; inactive query rows are not generated candidate paths.
    No valid/mode/reference input, previously generated path, or filtering.
    """

    def __init__(self, geometry_dim=34, horizon=24, width=192, depth=3, heads=4):
        super().__init__(geometry_dim + 1, horizon, 8, width, depth, heads)
        self.geometry_dim = geometry_dim

    def forward(self, geometry, k):
        if type(k) is not int or k not in BUDGETS:
            raise ValueError("K must be one of 1/2/4/8")
        if geometry.ndim != 2 or geometry.shape[-1] != self.geometry_dim:
            raise ValueError("only registered geometry condition is accepted")
        budget = geometry.new_full((len(geometry), 1), math.log2(k) / 3.)
        return super().forward(torch.cat((geometry, budget), dim=-1), k)
