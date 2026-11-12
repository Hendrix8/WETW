"""Deployable models using the paper's encoder and symmetric search MLP."""

import torch
from torch import nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from ._reference.model.ResidualAE import ResidualEncoder
from ._reference.architecture import AblationEncoder


class WETW(nn.Module):
    """s sqrt(sum_k w_k(a,b) (a_k-b_k)^2), with positive symmetric weights.

    ``kind`` also exposes the static measures of Table IV (ed, global_w, lowrank) and the
    direct MLP of Table V (mlp_direct), which predicts the distance itself from the same
    symmetric pair features, starting from the Euclidean distance.
    An epsilon-stabilized sqrt is used during optimization; exact self distances
    are zero in evaluation. No triangle inequality is assumed.
    """

    def __init__(self, dimension=128, hidden=128, kind="mlp_w", rank=64):
        super().__init__()
        self.dimension, self.hidden, self.kind, self.rank = (
            dimension,
            hidden,
            kind,
            rank,
        )
        self.log_s = nn.Parameter(torch.zeros(()))
        if kind in ("mlp_w", "mlp_direct"):
            self.net = nn.Sequential(
                nn.Linear(3 * dimension, hidden),
                nn.ReLU(),
                nn.Linear(hidden, dimension if kind == "mlp_w" else 1),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        elif kind == "global_w":
            self.theta = nn.Parameter(torch.zeros(dimension))
        elif kind == "lowrank":
            self.rank = min(int(rank), dimension)
            self.projection = nn.Parameter(torch.eye(self.rank, dimension))
        elif kind != "ed":
            raise ValueError(f"Unknown head: {kind}")

    def weights(self, a, b):
        features = torch.cat([(a - b).abs(), a * b, a + b], dim=-1)
        return F.softplus(self.net(features) + 0.5413)

    def forward(self, a, b):
        delta = a - b
        if self.kind == "mlp_direct":
            features = torch.cat([delta.abs(), a * b, a + b], dim=-1)
            euclidean = delta.square().sum(-1).clamp_min(0).sqrt()
            return self.log_s.exp() * F.softplus(self.net(features).squeeze(-1) + euclidean)
        if self.kind == "mlp_w":
            squared = (delta.square() * self.weights(a, b)).sum(-1)
        elif self.kind == "global_w":
            squared = (
                delta.square() * (F.softplus(self.theta) / 0.6931471805599453)
            ).sum(-1)
        elif self.kind == "lowrank":
            squared = (delta @ self.projection.T).square().sum(-1)
        else:
            squared = delta.square().sum(-1)
        if self.training:
            return self.log_s.exp() * (squared + 1e-12).sqrt()
        return self.log_s.exp() * squared.clamp_min(0).sqrt()


class WESee(nn.Module):
    """The archived paper encoder, including its multi-scale temporal branch.

    Accepts (batch, length) or (batch, 1, length). Length is fixed per model.
    ``variant``: full, or sfe (no temporal branch, Table VII).
    """

    def __init__(self, config, variant="full", reference_layout=False):
        super().__init__()
        cfg = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
        if reference_layout:
            self.encoder = ResidualEncoder(cfg)
        else:
            cfg.encoder_model.arch_ablation = variant
            self.encoder = AblationEncoder(cfg)
        self.dimension = int(cfg.encoder_model.dim_emb)
        self.length = int(cfg.dataset.dim_series)

    def forward(self, x):
        if x.ndim == 2:
            x = x[:, None, :]
        if x.ndim != 3 or x.shape[1:] != (1, self.length):
            raise ValueError(
                f"Expected (batch,1,{self.length}), received {tuple(x.shape)}"
            )
        return self.encoder(x).reshape(len(x), self.dimension)
