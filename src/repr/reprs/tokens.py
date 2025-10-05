from typing import Self

import torch
from torch import Tensor

from core.repr import Repr, TokenReprId
from core.tokeniser import Token, Tokeniser
from core.data_schema import DigitalInk
from tokeniser.factory import DefaultTokeniserFactory


class TokenRepr(Repr):
    def __init__(self, tokens: list[Token]):
        self._tokens = tokens

    def __str__(self) -> str:
        return '\n'.join([str(token) for token in self._tokens])
    
    @classmethod
    def _get_tokeniser(cls, id: TokenReprId) -> Tokeniser:
        return DefaultTokeniserFactory.create(id)
    
    @classmethod
    def from_ink(cls, id: TokenReprId, ink: DigitalInk) -> Self:
        tokeniser = cls._get_tokeniser(id)
        tokens = tokeniser.tokenise(ink)
        return cls(tokens)

    @classmethod
    def from_tensor(cls, id: TokenReprId, tensor: Tensor) -> Self:
        tokeniser = cls._get_tokeniser(id)
        tokens = tokeniser.convert_ids_to_tokens(tensor.tolist())
        return cls(tokens)

    def to_ink(self, id: TokenReprId) -> DigitalInk:
        tokeniser = self._get_tokeniser(id)
        return tokeniser.detokenise(self._tokens)

    def to_tensor(self, id: TokenReprId) -> Tensor:
        tokeniser = self._get_tokeniser(id)
        token_ids = tokeniser.convert_tokens_to_ids(self._tokens)
        return torch.tensor(token_ids, dtype=torch.long)
    
    
if __name__ == "__main__":
    from core.data_schema import Parsed
    from tokeniser.factory import DefaultTokeniserFactory
    from core.repr import TokenReprId, TokenReprType
    
    parsed = Parsed.load_random()
    ink = parsed.ink
    ink.visualise(name="(Original) Text: " + parsed.text)

    for delta in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        token_repr_id = TokenReprId(delta=delta, type=TokenReprType.SCRIBE)
        tokeniser = DefaultTokeniserFactory.create(token_repr_id)
        tokens = tokeniser.tokenise(ink)
        print(tokeniser._preprocessor)
        ink = tokeniser.detokenise(tokens)
        ink.visualise(name=f"(Delta {delta}) Text: " + parsed.text)
