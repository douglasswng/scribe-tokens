import torch
import torch.nn as nn

from ml_model.modules.ffn import FeedForward
from ml_model.modules.mha import MultiHeadAttention


class TransformerDecoderLayer(nn.Module):
    """Single Transformer Decoder Layer with RoPE"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_factor: float = 8 / 3,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout, max_seq_len)
        self.feed_forward = FeedForward(d_model, ffn_factor, dropout)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # Self-attention with residual connection and layer norm
        attn_out, new_kv_cache = self.self_attn(self.norm1(x), mask, start_pos, kv_cache)
        x = x + self.dropout(attn_out)

        # Feed-forward with residual connection and layer norm
        ff_out = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_out)
        return x, new_kv_cache
