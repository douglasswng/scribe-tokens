from typing import Protocol, Self
from pathlib import Path
from enum import Enum

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