from typing import Iterator
from itertools import groupby
import ujson as json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from core.tokeniser.merger_utils import TokenUnicodeBidict
from tokeniser.factory import DefaultTokeniserFactory
from core.data_schema import DigitalInk
from core.repr.id import TokenReprId, TokenReprType
from core.tokeniser.tokens import RegularToken, SpecialToken, SpecialTokenType


class InkBpeTrainer:
    def __init__(self,
                 tokeniser_id: TokenReprId,
                 vocab_size: int):
        self.tokeniser_id = tokeniser_id
        self.vocab_size = vocab_size
        self.token_unicode_bidict = TokenUnicodeBidict()

    def _get_regular_tokens_iterator(self,
                                     ink_iterator: Iterator[DigitalInk]
                                     ) -> Iterator[list[RegularToken]]:
        tokeniser = DefaultTokeniserFactory.create(self.tokeniser_id)
        for ink in ink_iterator:
            tokens = tokeniser.tokenise(ink)
            for is_regular, token_group in groupby(tokens, lambda x: isinstance(x, RegularToken)):
                if is_regular:
                    yield list(token for token in token_group if isinstance(token, RegularToken))

    def _get_str_iterator(self, ink_iterator: Iterator[DigitalInk]) -> Iterator[str]:
        for regular_tokens in self._get_regular_tokens_iterator(ink_iterator):
            yield ''.join(self.token_unicode_bidict.token_to_unicode(token)
                          for token in regular_tokens)

    def _get_vocab_and_merges(self,
                              hf_vocab: dict[str, int],
                              hf_merges: list[tuple[str, str]]
                              ) -> tuple[dict[RegularToken, int], list[tuple[RegularToken, RegularToken]]]:
        vocab = {}
        for token_str, token_id in hf_vocab.items():
            token = self.token_unicode_bidict.unicode_to_token(token_str)
            vocab[token] = token_id

        merges = [] 
        for merge in hf_merges:
            token_merge = (self.token_unicode_bidict.unicode_to_token(merge[0]), 
                           self.token_unicode_bidict.unicode_to_token(merge[1]))
            merges.append(token_merge)
        return vocab, merges

    def _extend_vocab(self, vocab: dict[RegularToken, int]) -> dict[SpecialToken | RegularToken, int]:
        special_tokens = [SpecialToken(type=SpecialTokenType.START),
                          SpecialToken(type=SpecialTokenType.END),
                          SpecialToken(type=SpecialTokenType.UP)]
        if self.tokeniser_id.type == TokenReprType.ABS \
            or self.tokeniser_id.type == TokenReprType.REL:
             special_tokens.append(SpecialToken(type=SpecialTokenType.UNKNOWN))
        if self.tokeniser_id.type == TokenReprType.SCRIBE:
            special_tokens.append(SpecialToken(type=SpecialTokenType.DOWN))
        
        extended_vocab = {}
        for i, token in enumerate(special_tokens + list(vocab.keys()), start=1):
            extended_vocab[token] = i
        return extended_vocab

    def _save_tokenizer(self,
                        extended_vocab: dict[SpecialToken | RegularToken, int],
                        merges: list[tuple[RegularToken, RegularToken]]) -> None:
        self.tokeniser_id.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tokeniser_id.vocab_path, 'w') as f:
            str_vocab = {str(token): idx for token, idx in extended_vocab.items()}
            json.dump(str_vocab, f, indent=4, ensure_ascii=False)

        self.tokeniser_id.merges_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.tokeniser_id.merges_path, 'w') as f:
            for left, right in merges:
                f.write(f"{left} {right}\n")

    def _process_tokenizer(self, tokenizer: Tokenizer) -> None:
        tokenizer_data = json.loads(tokenizer.to_str())
        vocab, merges = self._get_vocab_and_merges(tokenizer_data['model']['vocab'],
                                                    tokenizer_data['model']['merges'])
        extended_vocab = self._extend_vocab(vocab)
        if not merges:
            print("This tokenizer is merge-ineligible, skipping...")
        else:
            self._save_tokenizer(extended_vocab, merges)
            print(f"Saved tokenizer {self.tokeniser_id!s} to {self.tokeniser_id.tokeniser_path}")

    def train_from_iterator(self, ink_iterator: Iterator[DigitalInk]) -> None:
        hf_tokenizer = Tokenizer(BPE())
        trainer = BpeTrainer(vocab_size=self.vocab_size)  # type: ignore
        hf_tokenizer.train_from_iterator(self._get_str_iterator(ink_iterator), trainer=trainer)
        self._process_tokenizer(hf_tokenizer)