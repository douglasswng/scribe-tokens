from typing import Protocol

from tokenizers.models import BPE

from core.tokeniser.tokens import RegularToken
from core.tokeniser.merger_utils import TokenUnicodeBidict


class Merger(Protocol):
    def merge(self, tokens: list[RegularToken]) -> list[RegularToken]: ...
    

class HFMerger(Merger):
    def __init__(self,
                 vocab: dict[RegularToken, int],
                 merges: list[tuple[RegularToken, RegularToken]]):
        self.token_unicode_bidict = TokenUnicodeBidict()
        self.tokenizer = self._init_tokenizer(vocab, merges)

    def _init_tokenizer(self,
                        vocab: dict[RegularToken, int],
                        merges: list[tuple[RegularToken, RegularToken]]) -> BPE:
        unicode_vocab = {self.token_unicode_bidict.token_to_unicode(token): vocab[token]
                         for token in vocab}
        unicode_merges = [(self.token_unicode_bidict.token_to_unicode(token1),
                           self.token_unicode_bidict.token_to_unicode(token2))
                          for token1, token2 in merges]
        return BPE(vocab=unicode_vocab, merges=unicode_merges)
        
    def merge(self, tokens: list[RegularToken]) -> list[RegularToken]:
        tokens_unicode = ''.join([self.token_unicode_bidict.token_to_unicode(token)
                                  for token in tokens])
        tokenised = self.tokenizer.tokenize(tokens_unicode)
        token_strs: list[str] = [token.value for token in tokenised]
        merged_tokens = [self.token_unicode_bidict.unicode_to_token(token_str)
                         for token_str in token_strs]
        return merged_tokens