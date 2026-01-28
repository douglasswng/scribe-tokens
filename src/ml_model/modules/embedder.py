from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from constants import (
    DROPOUT,
    HIDDEN_DIM,
    MDN_RHO_MAX,
    MDN_STD_MIN,
    NUM_CHARS,
    NUM_MIXTURES,
    UNKNOWN_TOKEN_RATE,
    VOCAB_SIZE,
)

type MDNOutput = tuple[
    Tensor, Tensor, Tensor, Tensor, Tensor
]  # (mixtures, means, stds, rhos, pen_states)


class Embedder(nn.Module, ABC):
    @abstractmethod
    def embed(self, x: Tensor) -> Tensor: ...

    @abstractmethod
    def unembed(self, x: Tensor) -> Tensor | MDNOutput: ...


class VectorEmbedder(Embedder):
    def __init__(self, input_dim: int, with_unembed: bool = True):
        super().__init__()
        self._embedding = nn.Linear(input_dim, HIDDEN_DIM)
        self._dropout = nn.Dropout(DROPOUT)

        # Only create unembed layers if needed (for HTG/NTP tasks)
        if with_unembed:
            self._mixture_proj = nn.Linear(HIDDEN_DIM, NUM_MIXTURES)
            self._mean_proj = nn.Linear(HIDDEN_DIM, NUM_MIXTURES * 2)
            self._std_proj = nn.Linear(HIDDEN_DIM, NUM_MIXTURES * 2)
            self._rho_proj = nn.Linear(HIDDEN_DIM, NUM_MIXTURES)
            self._pen_state_proj = nn.Linear(HIDDEN_DIM, 3)
        else:
            self._mixture_proj = None
            self._mean_proj = None
            self._std_proj = None
            self._rho_proj = None
            self._pen_state_proj = None

    def embed(self, x: Tensor) -> Tensor:
        x = self._embedding(x)
        return self._dropout(x)

    def unembed(self, x: Tensor) -> MDNOutput:
        if (
            self._mixture_proj is None
            or self._mean_proj is None
            or self._std_proj is None
            or self._rho_proj is None
            or self._pen_state_proj is None
        ):
            raise RuntimeError(
                "Unembed layers not initialized. Create VectorEmbedder with with_unembed=True."
            )

        mixtures = self._mixture_proj(x)
        means = self._mean_proj(x)
        stds = self._std_proj(x)
        rhos = self._rho_proj(x)
        pen_states = self._pen_state_proj(x)

        mixtures = torch.softmax(mixtures, dim=-1)
        means = means.view(*means.size()[:-1], NUM_MIXTURES, 2)
        stds = F.softplus(stds.view(*stds.size()[:-1], NUM_MIXTURES, 2)) + MDN_STD_MIN
        rhos = torch.tanh(rhos) * MDN_RHO_MAX

        return mixtures, means, stds, rhos, pen_states

    def strip_unembed(self) -> None:
        self._mixture_proj = None
        self._mean_proj = None
        self._std_proj = None
        self._rho_proj = None
        self._pen_state_proj = None


class TokenEmbedder(Embedder):
    def __init__(self, unk_token_id: int | None):
        super().__init__()
        self._unk_token_id = unk_token_id

        self._embedding = nn.Embedding(VOCAB_SIZE + 1, HIDDEN_DIM, padding_idx=0)  # + 1 for pad
        self._dropout = nn.Dropout(DROPOUT)

    def _add_unk_token(self, x: Tensor) -> Tensor:
        assert self._unk_token_id is not None

        shape = x.size(0) if x.dim() == 1 else (x.size(0), x.size(1))
        dropout_mask = torch.rand(shape) < UNKNOWN_TOKEN_RATE
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
        self._embedding = nn.Embedding(
            NUM_CHARS + 3, HIDDEN_DIM, padding_idx=0
        )  # +3 for pad, bos, eos
        self._dropout = nn.Dropout(DROPOUT)

    def embed(self, x: Tensor) -> Tensor:
        x = self._embedding(x)
        return self._dropout(x)

    def unembed(self, x: Tensor) -> Tensor:
        logits = torch.matmul(x, self._embedding.weight.transpose(0, 1))  # parameter sharing
        return logits
