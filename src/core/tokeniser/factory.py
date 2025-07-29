from typing import Protocol

from core.repr.id import TokenReprId
from core.tokeniser.tokeniser import Tokeniser


class TokeniserFactory(Protocol):
    @classmethod
    def create(cls, id: TokenReprId) -> Tokeniser: ...