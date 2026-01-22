from typing import Iterator, Protocol

from ink_tokeniser.tokens import RegularToken, RegularTokenType
from tokenizers.models import BPE


class Merger(Protocol):
    def merge(self, tokens: list[RegularToken]) -> list[RegularToken]: ...


class HFMerger(Merger):
    def __init__(
        self, vocab: dict[RegularToken, int], merges: list[tuple[RegularToken, RegularToken]]
    ):
        self.token_unicode_bidict = TokenUnicodeBidict()
        self.tokenizer = self._init_tokenizer(vocab, merges)

    def _init_tokenizer(
        self, vocab: dict[RegularToken, int], merges: list[tuple[RegularToken, RegularToken]]
    ) -> BPE:
        unicode_vocab = {
            self.token_unicode_bidict.token_to_unicode(token): vocab[token] for token in vocab
        }
        unicode_merges = [
            (
                self.token_unicode_bidict.token_to_unicode(token1),
                self.token_unicode_bidict.token_to_unicode(token2),
            )
            for token1, token2 in merges
        ]
        return BPE(vocab=unicode_vocab, merges=unicode_merges)

    def merge(self, tokens: list[RegularToken]) -> list[RegularToken]:
        tokens_unicode = "".join(
            [self.token_unicode_bidict.token_to_unicode(token) for token in tokens]
        )
        tokenised = self.tokenizer.tokenize(tokens_unicode)
        token_strs: list[str] = [token.value for token in tokenised]
        merged_tokens = [
            self.token_unicode_bidict.unicode_to_token(token_str) for token_str in token_strs
        ]
        return merged_tokens


def get_unicode_iterator() -> Iterator[str]:
    for i in range(0x10FFFF + 1):
        # Skip surrogate code points (U+D800 through U+DFFF)
        if 0xD800 <= i <= 0xDFFF:
            continue
        yield chr(i)


class LazyBiDict[T]:
    def __init__(self, value_iterator: Iterator[str] | None = None):
        self.forward: dict[T, str] = {}
        self.backward: dict[str, T] = {}
        self.value_iterator = (
            value_iterator if value_iterator is not None else get_unicode_iterator()
        )

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
        return "".join(unicode_chars)

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
