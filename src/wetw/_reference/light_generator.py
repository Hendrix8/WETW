"""The search MLP used as the sole generator during encoder training and search."""
import torch
from torch import nn
import torch.nn.functional as F


class LightGenerator(nn.Module):
    def __init__(self, dimension=128):
        super().__init__()
        self.dimension = dimension
        self.log_s = nn.Parameter(torch.zeros(()))
        self.net = nn.Sequential(nn.Linear(3 * dimension, 128), nn.ReLU(), nn.Linear(128, dimension))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, left, right, **kwargs):
        features = torch.cat([(left-right).abs(), left*right, left+right], dim=1)
        return F.softplus(self.net(features) + .5413)

    def distance(self, left, right, w=None, **kwargs):
        # This is the lightweight search head's exact formula, including its
        # learned global scale and epsilon. Legacy heavy-head weight-sum gauges
        # do not apply. Original loss-level dimension scaling/calibration stays.
        if w is None:
            w = self(left, right)
        return self.log_s.exp() * torch.sqrt(((left-right).square()*w).sum(1) + 1e-12)


def install():
    import wetw._reference.util.train as training
    import wetw._reference.model.loss as loss_module

    def factory(**kwargs):
        generator = LightGenerator(int(kwargs['output_dim']))
        assert sum(p.numel() for p in generator.parameters()) == 65793
        # The legacy trainer expects weights from the module. Both its main
        # loss and triplet paths then call the distance helper. Bind that helper
        # to this generator so the scalar scale remains outside the square root
        # and all paths use precisely the deployed lightweight distance.
        loss_module.weighted_pairwise_distance = generator.distance
        training.weighted_pairwise_distance = generator.distance
        return generator

    training.PairConvModel = factory
