import torch
import torch.nn as nn
from torch import Tensor

from constants import DROPOUT, FFN_FACTOR, HIDDEN_DIM, MAX_LEN, NUM_HEADS, NUM_LAYERS
from ml_model.modules.decoder_layer import TransformerDecoderLayer


class TransformerDecoder(nn.Module):
    """Complete Transformer Decoder with RoPE"""

    def __init__(
        self,
        d_model: int = HIDDEN_DIM,
        n_heads: int = NUM_HEADS,
        n_layers: int = NUM_LAYERS,
        ffn_factor: float = FFN_FACTOR,
        max_seq_len: int = MAX_LEN,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(d_model, n_heads, ffn_factor, dropout, max_seq_len)
                for _ in range(n_layers)
            ]
        )

        # Final layer norm
        self.norm = nn.RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """Create causal (lower triangular) mask"""
        ones = torch.ones(seq_len, seq_len, device=device)
        causal_mask = torch.triu(ones, diagonal=1).bool()
        return causal_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)

    # TODO: add kv cache
    def forward(self, x: Tensor, start_pos: int = 0) -> Tensor:
        # Create causal mask
        seq_len = x.shape[1]
        mask = self._create_causal_mask(seq_len, x.device)

        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask, start_pos)

        # Final layer norm
        x = self.norm(x)
        return x


if __name__ == "__main__":
    decoder = TransformerDecoder(
        d_model=128, n_heads=4, n_layers=1, ffn_factor=8 / 3, dropout=0.1, max_seq_len=1024
    )
    x = torch.randn(2, 1024, 128)
    x = decoder(x)
    print(x.shape)
    print(torch.any(torch.isnan(x)))
