import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml_model.modules.rope import RoPEEmbedding


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE"""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, max_seq_len: int = 8192):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # RoPE embedding
        self.rope = RoPEEmbedding(self.head_dim, max_seq_len)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        start_pos: int = 0,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        batch_size, seq_len, _ = x.shape

        # Linear projections
        q = self.q_proj(x)  # (batch_size, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim)

        # Apply RoPE to queries and keys
        q = self.rope.apply_rotary_emb(q, start_pos)
        k = self.rope.apply_rotary_emb(k, start_pos)

        # Transpose for attention computation
        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle KV cache
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            # Concatenate with cached keys and values
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Store updated cache
        new_kv_cache = (k, v)

        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply causal mask
        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        # Apply softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)

        # Reshape and project output
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out, new_kv_cache


if __name__ == "__main__":
    batch_size, seq_len, num_heads, head_dim = 1, 1024, 16, 64
    mha = MultiHeadAttention(head_dim, num_heads)
    x = torch.randn(batch_size, seq_len, head_dim)
    print(mha(x).shape)
