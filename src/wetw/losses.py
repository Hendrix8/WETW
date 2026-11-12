import torch
import torch.nn.functional as F


def warpdistill(predicted, target, delta=0.1, epsilon=1e-6):
    """Huber loss on relative DTW error; zero-distance targets are excluded."""
    keep = target > epsilon
    if not keep.any():
        return predicted.sum() * 0
    relative = (predicted[keep] - target[keep]) / target[keep]
    return F.huber_loss(relative, torch.zeros_like(relative), delta=delta)


def log_ranking(positive, negative, margin=0.1, epsilon=1e-6):
    """Scale-invariant ranking hinge used in the paper's refinement stage."""
    return F.relu(
        margin + positive.clamp_min(epsilon).log() - negative.clamp_min(epsilon).log()
    ).mean()
