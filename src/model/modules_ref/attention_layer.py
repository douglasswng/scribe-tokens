import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import RMSNorm

from model.modules.rope import RotaryPostionalEmbedding


class GroupedKeyValue(nn.Module):
    def __init__(self, model_dimension: int,
                 group_size: int) -> None:
        super().__init__()
        self.group_size = group_size
        self.W_grouped = nn.Linear(model_dimension, model_dimension // self.group_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.W_grouped(x)
        x = x.repeat_interleave(self.group_size, dim=-1)
        return x


class GroupedQueryAttention(nn.Module):
    def __init__(self, model_dimension: int,
                 attention_heads: int,
                 key_value_heads: int,
                 positional_embedding: RotaryPostionalEmbedding) -> None:
        super().__init__()
        self.model_dimension = model_dimension
        self.attention_heads = attention_heads
        self.head_dimension = model_dimension // self.attention_heads
        group_size = self.attention_heads // key_value_heads
        self.W_q = nn.Linear(model_dimension, model_dimension, bias=False)
        self.W_k = GroupedKeyValue(model_dimension, group_size)
        self.W_v = GroupedKeyValue(model_dimension, group_size)
        self.positional_embedding = positional_embedding

    def padding_mask(self, x: Tensor) -> tuple[Tensor, Tensor]:
        # Return the mask instead of modifying in-place
        padding_mask = torch.isnan(x)
        x = x.masked_fill(padding_mask, 0)
        # Create attention mask (True means positions to mask out)
        attn_mask = padding_mask.any(dim=-1)  # Shape: (batch, seq_len)
        return x, attn_mask
    
    def forward(self, x: Tensor) -> Tensor:
        x, attn_mask = self.padding_mask(x)
        
        Q_matrix = self.W_q(x)
        K_matrix = self.W_k(x)
        V_matrix = self.W_v(x)

        Q_matrix = self.positional_embedding(Q_matrix)
        K_matrix = self.positional_embedding(K_matrix)
        
        # Reshape for multi-head attention
        batch_size, seq_len = x.shape[:2]
        Q_matrix = Q_matrix.view(batch_size, seq_len, self.attention_heads, self.head_dimension).transpose(1, 2)
        K_matrix = K_matrix.view(batch_size, seq_len, self.attention_heads, self.head_dimension).transpose(1, 2)
        V_matrix = V_matrix.view(batch_size, seq_len, self.attention_heads, self.head_dimension).transpose(1, 2)
        
        # Use PyTorch's scaled dot product attention
        attention = F.scaled_dot_product_attention(
            Q_matrix, K_matrix, V_matrix,
            attn_mask=attn_mask,
            is_causal=True,
            dropout_p=0.0 if not self.training else 0.0  # Add dropout if needed
        )
        
        # Reshape back
        attention = attention.transpose(1, 2).contiguous().view(batch_size, seq_len, self.model_dimension)
        return attention
    

class AttentionLayer(nn.Module):
    def __init__(self, model_dimension: int,
                 attention_heads: int,
                 key_value_heads: int,
                 positional_embedding: RotaryPostionalEmbedding) -> None:
        super().__init__()
        self.norm = RMSNorm(model_dimension)
        self.grouped_query_attention = GroupedQueryAttention(model_dimension,
                                                             attention_heads,
                                                             key_value_heads,
                                                             positional_embedding)
    
    def forward(self, x: Tensor) -> Tensor:
        x_skip = x.clone()     
        x = self.norm(x)
        x = self.grouped_query_attention(x)
        return x_skip + x

    
