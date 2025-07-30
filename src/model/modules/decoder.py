import torch
import torch.nn as nn

from model.modules.decoder_layer import TransformerDecoderLayer


class TransformerDecoder(nn.Module):
    """Complete Transformer Decoder with RoPE"""
    
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_factor: float = 8/3,
        dropout: float = 0.1,
        max_seq_len: int = 8192
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Decoder layers
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, ffn_factor, dropout, max_seq_len)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.norm = nn.RMSNorm(d_model)

        self.dropout = nn.Dropout(dropout)

         # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal (lower triangular) mask"""
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(
        self, 
        x: torch.Tensor, 
        start_pos: int = 0
    ) -> torch.Tensor:
        # Create causal mask
        seq_len = x.shape[1]
        mask = self.create_causal_mask(seq_len, x.device)
        
        # Pass through decoder layers
        for layer in self.layers:
            x = layer(x, mask, start_pos)
        
        # Final layer norm 
        x = self.norm(x)
        return x


if __name__ == "__main__":
    batch_size, seq_len, num_heads, head_dim = 1, 1024, 16, 64
    decoder = TransformerDecoder(head_dim, num_heads, head_dim * 4)
    x = torch.randn(batch_size, seq_len, head_dim)
    print(decoder(x).shape)