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
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for efficiency
        self._precompute_freqs_cis(max_seq_len)
    
    def _precompute_freqs_cis(self, seq_len: int):
        """Precompute cos and sin values for given sequence length"""
        t = torch.arange(seq_len, dtype=torch.float32)
        assert isinstance(self.inv_freq, torch.Tensor)
        freqs = torch.outer(t, self.inv_freq)
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
        self.register_buffer('freqs_cis', freqs_cis)
    
    def apply_rotary_emb(self, x: torch.Tensor, start_pos: int = 0) -> torch.Tensor:
        """
        Apply rotary embedding to input tensor
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_heads, head_dim)
            start_pos: Starting position for the sequence (useful for inference)
        """
        seq_len = x.shape[1]
        
        # Get the appropriate frequency values
        assert isinstance(self.freqs_cis, torch.Tensor)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seq_len]
        
        # Reshape x to complex representation
        x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Apply rotation
        freqs_cis = freqs_cis.view(1, seq_len, 1, -1)
        x_rotated = x_complex * freqs_cis
        
        # Convert back to real representation
        x_out = torch.view_as_real(x_rotated).flatten(-2)
        
        return x_out.type_as(x)
    

if __name__ == "__main__":
    batch_size, seq_len, num_heads, head_dim = 1, 1024, 16, 64
    rope = RoPEEmbedding(head_dim, max_seq_len=seq_len)
    x = torch.randn(batch_size, seq_len, num_heads, head_dim)
    print(rope.apply_rotary_emb(x).shape)