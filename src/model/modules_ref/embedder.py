from abc import ABC, abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from core.constants import HIDDEN_DIM, NUM_CHARS, DROPOUT, VOCAB_SIZE


class Embedder(nn.Module, ABC):
    @abstractmethod
    def embed(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def unembed(self, x: Tensor) -> Tensor | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]: ...


class TokenEmbedder(Embedder):
    def __init__(self):
        super().__init__()
        self._embedding = nn.Embedding(VOCAB_SIZE+1, HIDDEN_DIM, padding_idx=0)
        self._dropout = nn.Dropout(DROPOUT)

    def embed(self, x: Tensor) -> Tensor:
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