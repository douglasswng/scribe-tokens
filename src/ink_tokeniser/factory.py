from functools import lru_cache
from pathlib import Path

import ujson as json

from constants import SCRIBE_DOWNSAMPLE_FACTOR
from ink_tokeniser.discretes.abs import AbsTokeniser
from ink_tokeniser.discretes.discrete import DiscreteTokeniser
from ink_tokeniser.discretes.rel import RelTokeniser
from ink_tokeniser.discretes.scribe import ScribeTokeniser
from ink_tokeniser.discretes.text import TextTokeniser
from ink_tokeniser.id import TokeniserId, TokenType
from ink_tokeniser.preprocessor import DeltaSmoothPreprocessor
from ink_tokeniser.tokeniser import Tokeniser
from ink_tokeniser.tokens import RegularToken, Token, TokenParser
from ink_tokeniser.trained import TrainedTokeniser
from utils.distributed_context import distributed_context


class DiscreteFactory:
    @classmethod
    def create_discrete_tokeniser(cls, id: TokeniserId) -> DiscreteTokeniser:
        match id.type:
            case TokenType.ABS:
                return AbsTokeniser()
            case TokenType.REL:
                return RelTokeniser()
            case TokenType.SCRIBE:
                return ScribeTokeniser()
            case TokenType.TEXT:
                return TextTokeniser()
            case _:
                raise ValueError(f"Unknown tokeniser type: {id.type}")


class TrainedFactory:
    @classmethod
    def _load_vocab(cls, vocab_path: Path) -> dict[Token, int]:
        vocab: dict[Token, int] = {}
        with open(vocab_path, "r") as f:
            data = json.load(f)
        for k, v in data.items():
            token = TokenParser.from_str(k)
            vocab[token] = int(v)
        return vocab

    @classmethod
    def _load_merges(cls, merges_path: Path) -> list[tuple[RegularToken, RegularToken]]:
        merges: list[tuple[RegularToken, RegularToken]] = []
        with open(merges_path, "r") as f:
            for line in f.readlines():
                token1_str, token2_str = line.split()
                token1 = TokenParser.from_str(token1_str)
                token2 = TokenParser.from_str(token2_str)
                assert isinstance(token1, RegularToken) and isinstance(token2, RegularToken)
                merges.append((token1, token2))
        return merges

    @classmethod
    def _prune_vocab_and_merges(
        cls,
        vocab: dict[Token, int],
        merges: list[tuple[RegularToken, RegularToken]],
        vocab_size: int,
    ) -> tuple[dict[Token, int], list[tuple[RegularToken, RegularToken]]]:
        current_size = len(vocab)
        if current_size < vocab_size:
            raise ValueError(f"Target vocab size too small: {current_size} < {vocab_size}")

        pruned_vocab = {k: v for i, (k, v) in enumerate(vocab.items()) if i < vocab_size}

        reduce_count = current_size - vocab_size
        pruned_merges = merges[:-reduce_count] if reduce_count > 0 else merges

        return pruned_vocab, pruned_merges

    @classmethod
    def from_pretrained(cls, id: TokeniserId) -> TrainedTokeniser | None:
        if not id.tokeniser_path.exists():
            if distributed_context.is_master:
                print(f"Warning: {id.tokeniser_path} does not exist")
            return None
        if id.vocab_size is None:
            if distributed_context.is_master:
                print(f"Warning: {id} has no vocab size")
            return None

        vocab = cls._load_vocab(id.vocab_path)
        merges = cls._load_merges(id.merges_path)
        vocab, merges = cls._prune_vocab_and_merges(vocab, merges, id.vocab_size)
        return TrainedTokeniser(vocab=vocab, merges=merges)


class TokeniserFactory:
    @classmethod
    @lru_cache(maxsize=128)
    def create(cls, id: TokeniserId) -> Tokeniser:
        downsample_factor = SCRIBE_DOWNSAMPLE_FACTOR if id.is_scribe() else 1
        preprocessor = DeltaSmoothPreprocessor(delta=id.delta, downsample_factor=downsample_factor)
        discrete_tokeniser = DiscreteFactory.create_discrete_tokeniser(id)
        trained_tokeniser = TrainedFactory.from_pretrained(id)
        return Tokeniser(preprocessor, discrete_tokeniser, trained_tokeniser)


if __name__ == "__main__":
    from schemas.parsed import Parsed

    ink = Parsed.load_random().ink
    ink.visualise()
    print(f"DigitalInk length: {len(ink)}")

    for id in TokeniserId.create_defaults():
        tokeniser = TokeniserFactory.create(id)
        tokens = tokeniser.tokenise(ink)
        print(f"{id} length: {len(tokens)}")

        ink = tokeniser.detokenise(tokens)
        ink.visualise()
