# coding = utf-8

import torch
import torch.nn as nn


class LearnedPredDistanceCalib(nn.Module):
    """Train-time map on predicted pairwise distances (after dim scaling in the loss).

    bias — pred' = pred + b
    affine — pred' = pred * exp(s) + b (s = 0, b = 0 at init ⇒ identity)
    """

    def __init__(self, mode: str, bias_init: float = 0.0):
        super().__init__()
        m = (mode or "").lower()
        if m not in ("bias", "affine"):
            raise ValueError(
                "LearnedPredDistanceCalib mode must be bias or affine; "
                f"got {mode!r} (use None on ModelBuilder when disabled)"
            )
        self.mode = m
        self.bias = nn.Parameter(torch.tensor([float(bias_init)], dtype=torch.float32))
        if m == "affine":
            self.log_scale = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        b = self.bias.type_as(pred)
        if self.mode == "bias":
            return pred + b
        s = self.log_scale.type_as(pred)
        return pred * torch.exp(s) + b
