import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward Network as used in LLaMA and other modern transformers"""

    def __init__(self, d_model: int, ffn_factor: float = 8 / 3, dropout: float = 0.1):
        super().__init__()
        hidden_dim = int(d_model * ffn_factor)

        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(self.dropout(gate * up))
