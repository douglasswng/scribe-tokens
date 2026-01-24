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

    def forward(
        self,
        x: Tensor,
        start_pos: int = 0,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
        use_cache: bool = False,
    ) -> Tensor | tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        mask: Tensor | None = None
        if kv_caches is None:
            seq_len = x.shape[1]
            mask = self._create_causal_mask(seq_len, x.device)

        new_caches: list[tuple[Tensor, Tensor]] = []
        for i, layer in enumerate(self.layers):
            current_cache = kv_caches[i] if kv_caches is not None and i < len(kv_caches) else None
            x, layer_cache = layer(x, mask, start_pos, current_cache)

            if use_cache:
                new_caches.append(layer_cache)

        x = self.norm(x)
        return (x, new_caches) if use_cache else x
