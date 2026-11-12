# coding = utf-8

import torch
from torch import nn, Tensor


class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, input: Tensor) -> Tensor:
        return input.view(self.shape)


class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1

    def forward(self, input: Tensor) -> Tensor:
        return torch.transpose(input, self.dim0, self.dim1)


class Permute(nn.Module):
    def __init__(self, dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, input: Tensor) -> Tensor:
        return torch.permute(input, self.dims)


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super(Squeeze, self).__init__()
        self.dim = dim

    def forward(self, input: Tensor) -> Tensor:
        if self.dim is None:
            return input.squeeze()
        else:
            return input.squeeze(dim=self.dim)
