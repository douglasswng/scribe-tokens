from typing import Protocol, Self
from pathlib import Path
from enum import Enum

import ujson as json

from core.utils import distributed_context
from core.constants import TOKENISERS_DIR, VOCAB_SIZE, DELTA


class ReprId(Protocol):
    def __str__(self) -> str: ...

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)
    
    @property
    def type(self) -> Enum: ...

    @property
    def is_vector(self) -> bool: ...

    @property
    def has_oov(self) -> bool: ...

    @property
    def is_token(self) -> bool:
        return not self.is_vector
    

class VectorReprType(Enum):
    POINT3 = "Point-3"
    POINT5 = "Point-5"
    

class VectorReprId(ReprId):
    def __init__(self, type: VectorReprType):
        self._type = type

    def __str__(self) -> str:
        return self._type.value
    
    @classmethod
    def create_defaults(cls) -> list[Self]:
        return [cls(type) for type in VectorReprType]
    
    @classmethod
    def create_point3(cls) -> Self:
        return cls(VectorReprType.POINT3)
    
    @classmethod
    def create_point5(cls) -> Self:
        return cls(VectorReprType.POINT5)
    
    @property
    def type(self) -> VectorReprType:
        return self._type
    
    @property
    def is_vector(self) -> bool:
        return True

    @property
    def has_oov(self) -> bool:
        return False

    @property
    def dim(self) -> int:
        match self.type:
            case VectorReprType.POINT3:
                return 3
            case VectorReprType.POINT5:
                return 5
            case _:
                raise ValueError(f"Unknown vector repr type: {self.type}")

class TokenReprType(Enum):
    SCRIBE = 'ScribeTokens'
    ABS = 'AbsTokens'
    REL = 'RelTokens'
    TEXT = 'TextTokens'
    

class TokenReprId(ReprId):
    def __init__(self, 
                 type: TokenReprType,
                 delta: int | float,
                 vocab_size: int | None = None):
        self._type = type
        self._delta = delta
        self._vocab_size = vocab_size

    def __str__(self) -> str:
        return f'{self.type.value}-{self._delta} (vocab_size: {self._vocab_size})'
    
    @classmethod
    def create_defaults(cls) -> list[ReprId]:
        ids = []
        for type in TokenReprType:
            id = cls(type, DELTA, VOCAB_SIZE)
            if not id.tokeniser_path.exists():
                if distributed_context.is_master:
                    print(f"Warning: {id} does not exist")
                continue
            ids.append(id)
        return ids
    
    @classmethod
    def create_scribe(cls) -> Self:
        return cls(TokenReprType.SCRIBE, DELTA, VOCAB_SIZE)
    
    @property
    def tokeniser_path(self) -> Path:
        return TOKENISERS_DIR / self.type.value / f'{self.type.value}-{self._delta}'
    
    @property
    def vocab_path(self) -> Path:
        return self.tokeniser_path / 'vocab.json'
    
    @property
    def merges_path(self) -> Path:
        return self.tokeniser_path / 'merges.txt'
    
    @property
    def type(self) -> TokenReprType:
        return self._type
    
    @property
    def delta(self) -> int | float:
        return self._delta
    
    @property
    def vocab_size(self) -> int | None:
        return self._vocab_size
    
    @property
    def is_vector(self) -> bool:
        return False
    
    @property
    def is_scribe(self) -> bool:
        return self.type == TokenReprType.SCRIBE

    @property
    def has_oov(self) -> bool:
        return self.type in {TokenReprType.REL, TokenReprType.ABS}
    
    @property
    def has_trained(self) -> bool:
        # Check if tokenizer directory exists
        if not self.tokeniser_path.exists():
            return False
        
        # If no vocab_size is specified, we can't verify the tokenizer is trained for it
        if self.vocab_size is None:
            return False
        
        with open(self.vocab_path, 'r') as f:
            vocab_data = json.load(f)
        actual_vocab_size = len(vocab_data)
        
        # Vocab must be large enough
        if actual_vocab_size < self.vocab_size:
            return False
        
        # Count the number of merges
        with open(self.merges_path, 'r') as f:
            num_merges = sum(1 for line in f if line.strip())
        
        # Calculate how many merges would remain after pruning
        # reduce_count = actual_vocab_size - vocab_size
        # pruned_merges = merges[:-reduce_count] if reduce_count > 0 else merges
        reduce_count = actual_vocab_size - self.vocab_size
        remaining_merges = num_merges - reduce_count if reduce_count > 0 else num_merges
        
        # We need at least one merge for the trained tokenizer to be useful
        return remaining_merges > 0