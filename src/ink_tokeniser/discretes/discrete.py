from typing import Protocol

from ink_tokeniser.tokens import SpecialToken, SpecialTokenType, Token
from schemas.ink import DigitalInk


class DiscreteTokeniser(Protocol):
    def tokenise(self, ink: DigitalInk[int]) -> list[Token]: ...
    def detokenise(self, tokens: list[Token]) -> DigitalInk[int]: ...

    @property
    def unknown_token(self) -> SpecialToken:
        return SpecialToken(type=SpecialTokenType.UNKNOWN)

    @property
    def start_token(self) -> SpecialToken:
        return SpecialToken(type=SpecialTokenType.START)

    @property
    def down_token(self) -> SpecialToken:
        return SpecialToken(type=SpecialTokenType.DOWN)

    @property
    def up_token(self) -> SpecialToken:
        return SpecialToken(type=SpecialTokenType.UP)

    @property
    def end_token(self) -> SpecialToken:
        return SpecialToken(type=SpecialTokenType.END)
