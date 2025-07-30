import torch
from torch import Tensor
import torch.nn as nn


class RotaryPostionalEmbedding(nn.Module):
    def __init__(self, model_dimension: int, max_token_count: int) -> None:
        super().__init__()
        theta = torch.tensor([10000 ** (-2 * i / model_dimension) for i in range(model_dimension // 2)])
        m_theta = torch.stack([m * theta for m in range(1, max_token_count+1)])       
        m_theta = m_theta.repeat_interleave(2, dim=-1)
        self.cos = torch.cos(m_theta)
        self.sin = torch.sin(m_theta)

    def shuffle(self, x: Tensor) -> Tensor:
        x = x.view(x.size(0), x.size(1), x.size(2) // 2, 2)
        x = torch.stack([-x[..., 1], x[..., 0]], dim=-1)
        x = x.view(x.size(0), x.size(1), x.size(2) * 2)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        token_count = x.size(-2)
        x = x * self.cos[:token_count].unsqueeze(0) + self.shuffle(x) * self.sin[:token_count].unsqueeze(0)
        return x