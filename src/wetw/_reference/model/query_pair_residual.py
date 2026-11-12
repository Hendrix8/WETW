# coding = utf-8
"""Optional learned residual on predicted WETW distances: d' = d + scale * r(z_i, z_j).

Unlike global affine calibration, r depends on both endpoints and can change within-row
ordering (similarity-search relevant). Disabled when scale=0 or module not used."""

from __future__ import annotations

import torch
from torch import nn


class PairQueryDistanceResidual(nn.Module):
    def __init__(self, dim_emb: int, hidden_dim: int = 128):
        super().__init__()
        h = max(8, int(hidden_dim))
        d = int(dim_emb)
        self.net = nn.Sequential(
            nn.Linear(2 * d, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, 1),
        )
        # Start near zero residual so training begins close to previous behavior
        nn.init.zeros_(self.net[-1].bias)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=1e-4)

    def forward(self, z_left: torch.Tensor, z_right: torch.Tensor) -> torch.Tensor:
        """Return shape (N,) residual; z_left/z_right are (N, dim_emb)."""
        if z_left.shape != z_right.shape:
            raise ValueError(f"z_left {z_left.shape} vs z_right {z_right.shape}")
        x = torch.cat([z_left, z_right], dim=-1)
        r = self.net(x).squeeze(-1)
        return torch.tanh(r)
