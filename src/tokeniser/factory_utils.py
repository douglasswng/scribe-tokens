from core.tokeniser.tokens import Token, SpecialToken, RegularToken


class TokenParser:
    @classmethod
    def _try_parse_special(cls, s: str) -> Token | None:
        try:
            return SpecialToken.from_str(s)
        except ValueError:
            return None
    
    @classmethod
    def _try_parse_regular(cls, s: str) -> Token | None:
        try:
            return RegularToken.from_str(s)
        except ValueError:
            return None
    
    @classmethod
    def from_str(cls, s: str) -> Token:
        token = cls._try_parse_special(s) or cls._try_parse_regular(s)
        if token is None:
            raise ValueError(f"Unknown token: {s}")
        return token