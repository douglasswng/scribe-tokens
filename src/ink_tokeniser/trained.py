from functools import cached_property
from itertools import groupby

from ink_tokeniser.tokens import RegularToken, SpecialToken, SpecialTokenType, Token

from ink_tokeniser.merger import HFMerger, Merger


class TrainedTokeniser:
    def __init__(self, vocab: dict[Token, int], merges: list[tuple[RegularToken, RegularToken]]):
        self._vocab = vocab
        self._merger = self._init_merger(vocab, merges)

    @cached_property
    def _reverse_vocab(self) -> dict[int, Token]:
        return {v: k for k, v in self._vocab.items()}

    @cached_property
    def unk_token_id(self) -> int | None:
        unk_token = SpecialToken(SpecialTokenType.UNKNOWN)
        return self._vocab.get(unk_token, None)

    def _init_merger(
        self, vocab: dict[Token, int], merges: list[tuple[RegularToken, RegularToken]]
    ) -> Merger:
        regular_vocab = {token: vocab[token] for token in vocab if isinstance(token, RegularToken)}
        return HFMerger(vocab=regular_vocab, merges=merges)

    def _is_merge_eligible(self, token: Token) -> bool:
        return token.is_regular and token in self._vocab

    def merge(self, tokens: list[Token]) -> list[Token]:
        merged_tokens = []
        for is_merge_eligible, group in groupby(tokens, self._is_merge_eligible):
            if is_merge_eligible:
                regular_group = [token for token in group if isinstance(token, RegularToken)]
                merged_tokens.extend(self._merger.merge(regular_group))
            else:
                merged_tokens.extend(group)
        return merged_tokens

    def split(self, tokens: list[Token]) -> list[Token]:
        split_tokens = []
        for is_regular, group in groupby(tokens, lambda t: t.is_regular):
            if is_regular:
                regular_group = [token for token in group if isinstance(token, RegularToken)]
                for regular_token in regular_group:
                    split_tokens.extend(regular_token.split())
            else:
                split_tokens.extend(group)
        return split_tokens

    def convert_tokens_to_ids(self, tokens: list[Token]) -> list[int]:
        if self.unk_token_id is None:
            return [self._vocab[token] for token in tokens]
        else:
            return [self._vocab.get(token, self.unk_token_id) for token in tokens]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[Token]:
        return [self._reverse_vocab[id] for id in ids]
