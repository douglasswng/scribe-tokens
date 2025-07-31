from pathlib import Path
from functools import lru_cache

import ujson as json

from core.utils import distributed_context
from core.repr import TokenReprId, TokenReprType
from core.tokeniser import DiscreteTokeniser, TrainedTokeniser, DeltaPreprocessor, DeltaSmoothPreprocessor, RegularToken, Token, Tokeniser, TokeniserFactory
from tokeniser.tokenisers.abs import AbsTokeniser
from tokeniser.tokenisers.rel import RelTokeniser
from tokeniser.tokenisers.scribe import ScribeTokeniser
from tokeniser.tokenisers.text import TextTokeniser
from tokeniser.factory_utils import TokenParser
from core.constants import SCRIBE_DOWNSAMPLE_FACTOR


class DiscreteFactory:
    @classmethod
    def create_discrete_tokeniser(cls, id: TokenReprId) -> DiscreteTokeniser:
        match id.type:
            case TokenReprType.ABS:
                return AbsTokeniser()
            case TokenReprType.REL:
                return RelTokeniser()
            case TokenReprType.SCRIBE:
                return ScribeTokeniser()
            case TokenReprType.TEXT:
                return TextTokeniser()
            case _:
                raise ValueError(f"Unknown tokeniser type: {id.type}")
    

class TrainedFactory:
    @classmethod
    def _load_vocab(cls, vocab_path: Path) -> dict[Token, int]:
        vocab: dict[Token, int] = {}
        with open(vocab_path, 'r') as f:
            data = json.load(f)
        for k, v in data.items():
            token = TokenParser.from_str(k)
            vocab[token] = int(v)
        return vocab

    @classmethod
    def _load_merges(cls, merges_path: Path) -> list[tuple[RegularToken, RegularToken]]:
        merges: list[tuple[RegularToken, RegularToken]] = []
        with open(merges_path, 'r') as f:
            for line in f.readlines():
                token1_str, token2_str = line.split()
                token1 = TokenParser.from_str(token1_str)
                token2 = TokenParser.from_str(token2_str)
                assert isinstance(token1, RegularToken) and isinstance(token2, RegularToken)
                merges.append((token1, token2))
        return merges
    
    @classmethod
    def _prune_vocab_and_merges(cls,
                                vocab: dict[Token, int],
                                merges: list[tuple[RegularToken, RegularToken]],
                                vocab_size: int
                                ) -> tuple[dict[Token, int],
                                           list[tuple[RegularToken, RegularToken]]]:
        current_size = len(vocab)
        if current_size < vocab_size:
            raise ValueError(f"Target vocab size too small: {current_size} < {vocab_size}")
        
        pruned_vocab = {k: v for i, (k, v) in enumerate(vocab.items()) if i < vocab_size}
        
        reduce_count = current_size - vocab_size
        pruned_merges = merges[:-reduce_count] if reduce_count > 0 else merges

        return pruned_vocab, pruned_merges

    @classmethod
    def from_pretrained(cls, id: TokenReprId) -> TrainedTokeniser | None:
        if not id.tokeniser_path.exists():
            if distributed_context.is_master:
                print(f"Warning: {id.tokeniser_path} does not exist")
            return None
        if id.vocab_size is None:
            if distributed_context.is_master:
                print(f"Warning: {id} has no vocab size")
            return None

        vocab_path = id.tokeniser_path / 'vocab.json'
        vocab = cls._load_vocab(vocab_path)

        merges_path = id.tokeniser_path / 'merges.txt'
        merges = cls._load_merges(merges_path)

        vocab, merges = cls._prune_vocab_and_merges(vocab, merges, id.vocab_size)
        return TrainedTokeniser(vocab=vocab, merges=merges)
    

class DefaultTokeniserFactory(TokeniserFactory):
    @classmethod
    @lru_cache(maxsize=128)
    def create(cls, id: TokenReprId) -> Tokeniser:
        if id.is_scribe:
            preprocessor = DeltaSmoothPreprocessor(delta=id.delta, downsample_factor=SCRIBE_DOWNSAMPLE_FACTOR)
        else:
            preprocessor = DeltaSmoothPreprocessor(delta=id.delta, downsample_factor=1)
        discrete_tokeniser = DiscreteFactory.create_discrete_tokeniser(id)
        trained_tokeniser = TrainedFactory.from_pretrained(id)
        return Tokeniser(preprocessor, discrete_tokeniser, trained_tokeniser)
    

if __name__ == "__main__":
    from core.data_schema import Parsed

    ink = Parsed.load_random().ink
    # ink = Parsed.from_path('/Users/douglasswang/Desktop/TCL/writing_beaut/ScribeTokens0728/data/parsed/iam/n09-105z-06.json').ink
    ink = Parsed.from_path('/data/doug/ScribeTokens0728/data/parsed/iam/a10-673z-05.json').ink
    ink.visualise()

    #for id in TokenReprId.create_defaults():
    for id in [TokenReprId.create_scribe()]:
        tokeniser = DefaultTokeniserFactory.create(id)
        tokens = tokeniser.tokenise(ink)

        print(f"Tokeniser length: {len(tokens)}")
        print(f"DigitalInk length: {len(ink)}")
        
        ink = tokeniser.detokenise(tokens)
        ink.visualise()