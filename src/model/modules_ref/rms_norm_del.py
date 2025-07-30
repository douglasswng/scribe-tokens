import torch
from torch import Tensor
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, model_dimension: int):
        super().__init__()
        self.g = nn.Parameter(torch.ones(model_dimension))

    def forward(self, x: Tensor) -> Tensor:
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True))
        x = x / rms
        x = x * self.g
        return x