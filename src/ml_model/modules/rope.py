import torch
import torch.nn as nn


class RoPEEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) implementation"""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Precompute frequency matrix
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute cos and sin for efficiency
        self._precompute_freqs_cis(max_seq_len)

    def _precompute_freqs_cis(self, seq_len: int):
        """Precompute cos and sin values for given sequence length"""
        t = torch.arange(seq_len, dtype=torch.float32)
        assert isinstance(self.inv_freq, torch.Tensor)
        freqs = torch.outer(t, self.inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer("freqs_cis", freqs_cis)

    def apply_rotary_emb(self, x: torch.Tensor, start_pos: int | torch.Tensor = 0) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
            start_pos: Position information, can be:
                - int: Global start position
                - 2D tensor [batch_size, seq_len]: Explicit position for each element
        """
        _, seq_len = x.shape[:2]

        # Get the appropriate frequency values
        assert isinstance(self.freqs_cis, torch.Tensor)

        if isinstance(start_pos, int):
            # Scalar: all sequences use same consecutive positions
            freqs_cis = self.freqs_cis[start_pos : start_pos + seq_len]
            freqs_cis = freqs_cis.view(1, seq_len, 1, -1)
        else:
            # 2D tensor: explicit position for each element
            # start_pos: [batch_size, seq_len]
            assert start_pos.ndim == 2, f"start_pos must be int or 2D tensor, got {start_pos.ndim}D"
            freqs_cis = self.freqs_cis[start_pos].unsqueeze(2)  # [batch, seq_len, 1, -1]

        # Reshape x to complex representation
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))

        # Apply rotation
        x_rotated = x_complex * freqs_cis

        # Convert back to real representation
        x_out = torch.view_as_real(x_rotated).flatten(-2)

        return x_out.type_as(x)
