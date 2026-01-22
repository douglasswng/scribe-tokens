from typing import Self

import torch
from torch import Tensor

from ink_repr.repr import InkRepr
from ink_tokeniser.tokeniser import Tokeniser
from ink_tokeniser.tokens import Token
from schemas.ink import DigitalInk


class TokenRepr(InkRepr):
    def __init__(self, tokens: list[Token], tokeniser: Tokeniser):
        self._tokens = tokens
        self._tokeniser = tokeniser

    def __str__(self) -> str:
        return "\n".join([str(token) for token in self._tokens])

    @classmethod
    def from_ink(cls, ink: DigitalInk, tokeniser: Tokeniser) -> Self:
        tokens = tokeniser.tokenise(ink)
        return cls(tokens, tokeniser)

    @classmethod
    def from_tensor(cls, tensor: Tensor, tokeniser: Tokeniser) -> Self:
        tokens = tokeniser.convert_ids_to_tokens(tensor.tolist())
        return cls(tokens, tokeniser)

    def to_ink(self) -> DigitalInk:
        return self._tokeniser.detokenise(self._tokens)

    def to_tensor(self) -> Tensor:
        token_ids = self._tokeniser.convert_tokens_to_ids(self._tokens)
        return torch.tensor(token_ids, dtype=torch.long)
