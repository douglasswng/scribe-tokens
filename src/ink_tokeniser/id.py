from enum import StrEnum
from pathlib import Path
from typing import Self

from constants import DELTA, TOKENISERS_DIR, VOCAB_SIZE
from utils.distributed_context import distributed_context


class TokenType(StrEnum):
    SCRIBE = "ScribeTokens"
    ABS = "AbsTokens"
    REL = "RelTokens"
    TEXT = "TextTokens"


class TokeniserId:
    def __init__(self, type: TokenType, delta: int | float, vocab_size: int | None = None):
        self.type = type
        self.delta = delta
        self.vocab_size = vocab_size

    def __str__(self) -> str:
        return f"{self.type}-{self.delta} (vocab_size: {self.vocab_size})"

    @property
    def tokeniser_path(self) -> Path:
        return TOKENISERS_DIR / self.type.value / f"{self.type.value}-{self.delta}"

    @property
    def vocab_path(self) -> Path:
        return self.tokeniser_path / "vocab.json"

    @property
    def merges_path(self) -> Path:
        return self.tokeniser_path / "merges.txt"

    @classmethod
    def create_defaults(cls) -> list[Self]:
        ids = []
        for type in TokenType:
            id = cls(type=type, delta=DELTA, vocab_size=VOCAB_SIZE)
            if not id.tokeniser_path.exists():
                if distributed_context.is_master:
                    print(f"Warning: {id} does not exist")
                continue
            ids.append(id)
        return ids

    @classmethod
    def create_scribe(cls) -> Self:
        return cls(type=TokenType.SCRIBE, delta=DELTA, vocab_size=VOCAB_SIZE)

    @classmethod
    def create_rel(cls) -> Self:
        return cls(type=TokenType.REL, delta=DELTA, vocab_size=VOCAB_SIZE)

    @classmethod
    def create_text(cls) -> Self:
        return cls(type=TokenType.TEXT, delta=DELTA, vocab_size=VOCAB_SIZE)

    def is_scribe(self) -> bool:
        return self.type == TokenType.SCRIBE
