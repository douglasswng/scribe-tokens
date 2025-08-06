from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from core.constants import HIDDEN_DIM, NUM_CHARS, DROPOUT, VOCAB_SIZE, NUM_MIXTURES, UNKNOWN_TOKEN_RATE


type MDNOutput = tuple[Tensor, Tensor, Tensor, Tensor, Tensor]  # (mixtures, means, stds, rhos, pen_states)


class Embedder(nn.Module, ABC):
    @abstractmethod
    def embed(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def unembed(self, x: Tensor) -> Tensor | MDNOutput: ...


class VectorEmbedder(Embedder):
    def __init__(self, input_dim: int):
        super().__init__()
        self._embedding = nn.Linear(input_dim, HIDDEN_DIM)
        
        self._mixture_proj = nn.Linear(HIDDEN_DIM, NUM_MIXTURES)
        self._mean_proj = nn.Linear(HIDDEN_DIM, NUM_MIXTURES * 2)
        self._std_proj = nn.Linear(HIDDEN_DIM, NUM_MIXTURES * 2)
        self._rho_proj = nn.Linear(HIDDEN_DIM, NUM_MIXTURES)
        self._pen_state_proj = nn.Linear(HIDDEN_DIM, 3)

        self._dropout = nn.Dropout(DROPOUT)

    def embed(self, x: Tensor) -> Tensor:
        x = self._embedding(x)
        return self._dropout(x)
    
    def unembed(self, x: Tensor) -> MDNOutput:
        mixtures = self._mixture_proj(x)
        means = self._mean_proj(x)
        stds = self._std_proj(x)
        rhos = self._rho_proj(x)
        pen_states = self._pen_state_proj(x)

        mixtures = torch.softmax(mixtures, dim=-1)
        means = means.view(*means.size()[:-1], NUM_MIXTURES, 2)
        stds = F.softplus(stds.view(*stds.size()[:-1], NUM_MIXTURES, 2)) + 1e-3
        rhos = torch.tanh(rhos) * 0.99

        return mixtures, means, stds, rhos, pen_states


class TokenEmbedder(Embedder):
    def __init__(self, unk_token_id: int | None):
        super().__init__()
        self._unk_token_id = unk_token_id

        self._embedding = nn.Embedding(VOCAB_SIZE+1, HIDDEN_DIM, padding_idx=0)
        self._dropout = nn.Dropout(DROPOUT)

    def _add_unk_token(self, x: Tensor) -> Tensor:
        assert self._unk_token_id is not None

        dropout_mask = torch.rand(x.size(0), x.size(1)) < UNKNOWN_TOKEN_RATE
        dropout_mask = dropout_mask.to(x.device)
        x = x.masked_fill(dropout_mask, self._unk_token_id)
        return x

    def embed(self, x: Tensor) -> Tensor:
        if self._unk_token_id is not None and self.training:
            x = self._add_unk_token(x)
        x = self._embedding(x)
        return self._dropout(x)
    
    def unembed(self, x: Tensor) -> Tensor:
        logits = torch.matmul(x, self._embedding.weight.transpose(0, 1))  # parameter sharing
        return logits
    

class CharEmbedder(Embedder):
    def __init__(self):
        super().__init__()
        self._embedding = nn.Embedding(NUM_CHARS+3, HIDDEN_DIM, padding_idx=0)  # +2 for pad, bos, eos
        self._unembedding = nn.Linear(HIDDEN_DIM, NUM_CHARS+1) # +1 for blank
        self._dropout = nn.Dropout(DROPOUT)

    def embed(self, x: Tensor) -> Tensor:
        x = self._embedding(x)
        return self._dropout(x)
    
    def unembed(self, x: Tensor) -> Tensor:
        logits = self._unembedding(x)
        return logits