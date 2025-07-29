from typing import Protocol, Self, Sequence
from enum import Enum


class Token(Protocol):
    def __str__(self) -> str: ...

    def __hash__(self) -> int:
        return hash(str(self))
    
    def __eq__(self, other: object) -> bool:
        return str(self) == str(other)
    
    @classmethod
    def from_str(cls, s: str) -> Self: ...
    
    @property
    def is_special(self) -> bool: ...

    @property
    def is_regular(self) -> bool:
        return not self.is_special
    

class SpecialTokenType(Enum):
    UNKNOWN = 'UNKNOWN'
    START = 'START'
    DOWN = 'DOWN'
    UP = 'UP'
    END = 'END'
    

class SpecialToken(Token):
    def __init__(self, type: SpecialTokenType):
        self._type = type

    def __str__(self) -> str:
        return f'[{self.type.value}]'
    
    @classmethod
    def from_str(cls, s: str) -> Self:
        s = s.strip('[]')
        type = SpecialTokenType(s)
        return cls(type=type)
    
    @property
    def type(self) -> SpecialTokenType:
        return self._type

    @property
    def is_special(self) -> bool:
        return True
    

class RegularTokenType(Enum):
    ABS = 'ABS'
    REL = 'REL'
    TEXT = 'TEXT'
    SCRIBE = 'SCRIBE'
    

class Value(Protocol):
    def __str__(self) -> str: ...

    @classmethod
    def from_str(cls, s: str) -> Self: ...


class Coord(Value):
    def __init__(self, x: int, y: int):
        self._x = x
        self._y = y

    def __str__(self) -> str:
        return f'({self._x},{self._y})'
    
    @classmethod
    def from_str(cls, s: str) -> Self:
        s = s.strip('()')
        x, y = s.split(',')
        return cls(x=int(x), y=int(y))
    
    @property
    def x(self) -> int:
        return self._x
    
    @property
    def y(self) -> int:
        return self._y
    

class RegularToken(Token):
    def __init__(self,
                 type: RegularTokenType,
                 values: Sequence[Value] | str):
        self._type = type
        self._values = values

    def __str__(self) -> str:
        if isinstance(self.values, str):
            values_str = self.values
        else:
            values_str = ';'.join([str(v) for v in self.values])
        return f'[{self.type.value}:{values_str}]'
    
    def __add__(self, other: Self) -> Self:
        if self.type != other.type:
            raise ValueError(f'Cannot add tokens of different types: {self.type} and {other.type}')
        if isinstance(self.values, str) and isinstance(other.values, str):
            return type(self)(type=self.type, values=self.values + other.values)
        elif isinstance(self.values, list) and isinstance(other.values, list):
            return type(self)(type=self.type, values=self.values + other.values)
        else:
            raise ValueError(f'Cannot add tokens of different types: {self.type} and {other.type}')
    
    @classmethod
    def from_str(cls, s: str) -> Self:
        s = s.strip('[]')
        type_str, values_str = s.split(':')
        type = RegularTokenType(type_str)

        if type in {RegularTokenType.TEXT, RegularTokenType.SCRIBE}:
            return cls(type=type, values=values_str)
        elif type in {RegularTokenType.ABS, RegularTokenType.REL}:
            values: list[Value] = [Coord.from_str(v.strip())
                                   for v in values_str.split(';')]
            return cls(type=type, values=values)
        else:
            raise ValueError(f'{cls} has no value type')
    
    @property
    def type(self) -> RegularTokenType:
        return self._type
    
    @property
    def values(self) -> Sequence[Value] | str:
        return self._values

    @property
    def is_special(self) -> bool:
        return False
    
    def split(self) -> list[Self]:
        if isinstance(self.values, str):
            return [type(self)(type=self.type, values=v) for v in self.values]
        else:
            return [type(self)(type=self.type, values=[v]) for v in self.values]