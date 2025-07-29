from typing import Iterator

from core.tokeniser.tokens import RegularToken, RegularTokenType


def get_unicode_iterator() -> Iterator[str]:
    for i in range(0x10FFFF + 1):
        # Skip surrogate code points (U+D800 through U+DFFF)
        if 0xD800 <= i <= 0xDFFF:
            continue
        yield chr(i)


class LazyBiDict[T]:
    def __init__(self, value_iterator: Iterator[str]=get_unicode_iterator()):
        self.forward: dict[T, str] = {}
        self.backward: dict[str, T] = {}
        self.value_iterator = value_iterator

    def __getitem__(self, key: T) -> str:
        if key not in self.forward:
            self[key] = next(self.value_iterator)
        return self.forward[key]
    
    def __setitem__(self, key: T, value: str) -> None:
        self.forward[key] = value
        self.backward[value] = key

    def inverse(self, value: str) -> T:
        return self.backward[value]


class TokenUnicodeBidict:
    def __init__(self):
        self._token_type: RegularTokenType | None = None
        self._value_is_str: bool | None = None
        self._bidict = LazyBiDict[RegularToken]()

    def token_to_unicode(self, token: RegularToken) -> str:
        if self._token_type is None:
            self._token_type = token.type
        if self._value_is_str is None:
            self._value_is_str = isinstance(token.values, str)
        if isinstance(token.values, str):
            return token.values

        base_tokens = token.split()
        unicode_chars = [self._bidict[token] for token in base_tokens]
        return ''.join(unicode_chars)
    
    def unicode_to_token(self, unicode_str: str) -> RegularToken:
        if self._token_type is None:
            raise ValueError("Token type not set")
        if self._value_is_str is None:
            raise ValueError("Token value type not set")
        if self._value_is_str:
            return RegularToken(type=self._token_type, values=unicode_str)

        base_tokens = [self._bidict.inverse(char) for char in unicode_str]
        token = sum(base_tokens[1:], start=base_tokens[0])
        return token